"""Pseudo-bag replay *training* adapters (PMIL, ProDiv, PseMix, MMIL).

The selection-only adapters keep a patch subset per slide and store the group
IDs of the retained patches (``pseudo_bag_ids``).  This module consumes those
IDs in the replay loss: each replay update splits (or mixes) the replayed
slide(s) into pseudo-bags, forwards every pseudo-bag through the unchanged
backbone and aggregates the backbone's own supervised loss.  The splitting /
mixing rule is the only difference between methods; the protocol (one optimizer
step per replay update, 78 updates per epoch, ER) is unchanged.

These are replay adapters of the published *training-time* mechanisms, not
re-implementations of the original models; see PSEUDOBAG_PBTRAIN_SPEC.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

METHODS = ("pmil", "prodiv", "psemix", "mmil")


def _slide_id(sample) -> str:
    value = sample.get("slide_id", "")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def _label(sample) -> int:
    return int(torch.as_tensor(sample["label"]).reshape(-1)[0].item())


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def pmil_stage_count(m_max: int, step: int) -> list[int]:
    """Official PMIL rounds: range(1, M_max + 1, step), then M_max."""
    if int(step) >= int(m_max):
        return [int(m_max)]
    stages = list(range(1, int(m_max) + 1, int(step)))
    if stages[-1] != int(m_max):
        stages.append(int(m_max))
    return stages


def pmil_stage_for_epoch(stages: list[int], epoch: int, epochs: int) -> int:
    """Spread the progressive rounds evenly over the epochs of one task."""
    index = min(len(stages) - 1, int(epoch) * len(stages) // max(1, int(epochs)))
    return stages[index]


def prodiv_pseudo_bags(strata: torch.Tensor, groups: int, rng: np.random.Generator):
    """Official ProDiv rule: shuffle each stratum, split it into ``groups``
    equal chunks, and pseudo-bag g is the union of chunk g over strata."""
    bags = [[] for _ in range(groups)]
    for stratum in torch.unique(strata).tolist():
        members = np.flatnonzero((strata == stratum).cpu().numpy())
        rng.shuffle(members)
        for g, chunk in enumerate(np.array_split(members, groups)):
            bags[g].extend(chunk.tolist())
    return [torch.as_tensor(sorted(b), dtype=torch.long) for b in bags if len(b)]


def psemix_assignment(phenotypes: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Official PseMix ``uniform_assign``: each phenotype cluster is spread
    uniformly at random over the ``n`` pseudo-bags."""
    generator = torch.Generator().manual_seed(seed)
    out = torch.zeros_like(phenotypes.cpu(), dtype=torch.long)
    for cluster in torch.unique(phenotypes).tolist():
        mask = (phenotypes.cpu() == cluster)
        size = int(mask.sum())
        labels = torch.randperm(size, generator=generator) % n
        permuted = torch.randperm(n, generator=generator)
        out[mask] = permuted[labels]
    return out


class PseudoBagReplayEpoch:
    """Per-epoch pseudo-bag replay loss; deterministic given seed/fold/task/epoch."""

    def __init__(self, args, buffer, *, fold: int, task: int, epoch: int, log_path: Path | None):
        self.method = str(args.replay_pseudobag_train)
        if self.method not in METHODS:
            raise ValueError(f"unknown replay_pseudobag_train: {self.method}")
        if str(args.cl_method) != "ER":
            raise ValueError("pseudo-bag replay training is defined for plain ER only")
        self.args, self.buffer = args, buffer
        self.fold, self.task, self.epoch = int(fold), int(task), int(epoch)
        self.min_bag = int(args.pbtrain_min_bag)
        seed = (int(args.seed) + 7_000_003 * self.fold + 10_007 * self.task + 101 * self.epoch) % (2**32)
        self.rng = np.random.default_rng(seed)
        self.log_path = log_path
        self.update = 0
        if self.method == "pmil":
            self.stages = pmil_stage_count(int(args.pbtrain_pmil_m_max), int(args.pbtrain_pmil_step))
            self.m = (self.stages[-1] if not bool(args.pbtrain_pmil_progressive)
                      else pmil_stage_for_epoch(self.stages, self.epoch, int(args.epochs)))
        self._partners = None

    # ---- partition construction -------------------------------------------------
    def _groups(self, sample) -> list[torch.Tensor]:
        """Index sets (into the stored, selector-ordered patches) of the pseudo-bags."""
        n = int(sample["features"].shape[0])
        ids = sample.get("pseudo_bag_ids")
        if ids is None or int(ids.numel()) != n:
            raise RuntimeError("pseudo-bag training needs stored pseudo_bag_ids aligned with the stored patches")
        ids = ids.detach().cpu().long()
        if self.method == "pmil":
            # Stored order is the attention ranking; pseudo-bag i is ranking[i::M].
            position = torch.arange(n) % self.m
            if self.m == int(torch.unique(ids).numel()) and not torch.equal(position, ids):
                raise RuntimeError("stored PMIL pseudo-bag IDs are not the rank-stride partition")
            return [torch.nonzero(position == g).reshape(-1) for g in range(self.m)]
        if self.method == "prodiv":
            return prodiv_pseudo_bags(ids, int(self.args.pbtrain_prodiv_groups), self.rng)
        if self.method == "mmil":
            return [torch.nonzero(ids == g).reshape(-1) for g in torch.unique(ids).tolist()]
        raise AssertionError  # psemix builds bags in _psemix_terms

    def _partner(self, sample):
        if self._partners is None:
            self._partners = [s for s in self._buffer_samples()]
        pool = [s for s in self._partners if _slide_id(s) != _slide_id(sample)] or self._partners
        return pool[int(self.rng.integers(len(pool)))]

    def _buffer_samples(self):
        if hasattr(self.buffer, "_all_samples"):
            return list(self.buffer._all_samples())
        return list(self.buffer.buffer)

    # ---- terms: (features, [(label, weight), ...], weight, tag) ---------------------
    def _terms(self, sample, device):
        feats, label = sample["features"], sample["label"].reshape(-1)
        if self.method == "psemix":
            return self._psemix_terms(sample, device)
        groups = [g for g in self._groups(sample) if int(g.numel()) >= self.min_bag]
        terms = []
        if self.method == "mmil":
            terms.append((feats, [(label, 1.0)], float(self.args.pbtrain_mmil_full_weight), "full"))
            if groups:
                w = (1.0 - float(self.args.pbtrain_mmil_full_weight)) / len(groups)
                terms += [(feats[g.to(device)], [(label, 1.0)], w, f"sub{k}") for k, g in enumerate(groups)]
            else:
                terms[0] = (feats, [(label, 1.0)], 1.0, "full")
            return terms
        if not groups or (self.method == "pmil" and self.m == 1):
            return [(feats, [(label, 1.0)], 1.0, "full")]
        w = 1.0 / len(groups)
        return [(feats[g.to(device)], [(label, 1.0)], w, f"pb{k}") for k, g in enumerate(groups)]

    def _psemix_terms(self, sample, device):
        n = int(self.args.pbtrain_psemix_n)
        partner = self._partner(sample)

        def bags(s):
            assign = psemix_assignment(s["pseudo_bag_ids"], n, _stable_seed(self.args.seed, "psemix", _slide_id(s)))
            return assign

        def fetch(s, k):
            if k <= 0:
                return None
            assign = bags(s)
            chosen = torch.as_tensor(self.rng.permutation(n)[:k])
            keep = torch.isin(assign, chosen).nonzero().reshape(-1)
            return s["features"].to(device)[keep.to(device)] if keep.numel() else None

        lam = float(self.rng.beta(float(self.args.pbtrain_psemix_alpha), float(self.args.pbtrain_psemix_alpha)))
        lam_i = int(((lam if lam != 1.0 else lam - 1e-5)) * (n + 1))
        bag_a, bag_b = fetch(sample, lam_i), fetch(partner, n - lam_i)
        ya, yb = sample["label"].reshape(-1), partner["label"].reshape(-1).to(device)
        ratio = lam_i / n
        if bag_a is None:
            feats, ratio, tag = bag_b, 0.0, "partner_only"
        elif bag_b is None:
            feats, ratio, tag = bag_a, 1.0, "anchor_only"
        elif float(self.rng.random()) <= float(self.args.pbtrain_psemix_prob):
            feats, tag = torch.cat([bag_a, bag_b], dim=0), "mixed"
        else:
            feats, ratio, tag = bag_a, 1.0, "anchor_subset"
        self._mix = dict(lam=lam, lam_pseudo_bags=lam_i, ratio=ratio, partner=_slide_id(partner),
                         partner_label=_label(partner), n_pseudo_bags=n)
        labels = [(ya, ratio), (yb, 1.0 - ratio)]
        return [(feats, [(y, w) for y, w in labels if w > 0], 1.0, tag)]

    # ---- loss ----------------------------------------------------------------------
    def loss(self, old_batch, term_loss: Callable[[torch.Tensor, list], torch.Tensor]):
        self._mix = None
        device = old_batch["features"].device
        terms = self._terms(old_batch, device)
        total, rows = 0.0, []
        for feats, labels, weight, tag in terms:
            value = term_loss(feats, labels)
            total = total + weight * value
            rows.append(dict(tag=tag, patches=int(feats.shape[0]), weight=round(float(weight), 6),
                             loss=float(value.detach())))
        self.update += 1
        if self.log_path is not None:
            record = dict(method=self.method, fold=self.fold, task=self.task, epoch=self.epoch, update=self.update,
                          slide=_slide_id(old_batch), label=_label(old_batch),
                          stored_patches=int(old_batch["features"].shape[0]), terms=rows,
                          total_loss=float(total.detach()))
            if self.method == "pmil":
                record["pseudo_bags_M"] = self.m
            if self._mix:
                record.update(self._mix)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return total
