"""Model-specific KEM v2 replay selectors.

CLAM uses a true virtual-update/full-bag outer objective.  TransMIL uses
token-gradient matching because a masked full grid is not equivalent to the
compact token sequence used during replay.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .adapters import ReplayModelAdapter
from . import mambamil_adapter as _mambamil_adapter  # register model hooks
from .bilevel_selector import BilevelConfig
from .candidates import _cosine_kmeans, build_candidate_pool, stratified_attention_indices
from .losses import replay_distillation_loss
from .region_supervision import (
    RegionBatchCLAMAdapter,
    RegionBatchTransMILAdapter,
    RegionConfig,
    RegionSupervisor,
    weighted_region_kl,
    pad_regions,
)
from .mambamil_adapter import RegionBatchMambaMILAdapter
from .model_registry import get_replay_adapter, get_training_loss
from .solvers import _grads, _soft_exact_budget


@dataclass(frozen=True)
class KemV2Config:
    variant: str = "anchored"
    anchor_ratio: float = 0.5
    anchor_method: str = "maxminrand"
    candidate_method: str = "maxminrand"
    candidate_ratio: float = 4.0
    backend: str = "auto"
    outer_steps: int = 15
    inner_steps: int = 1
    proxy_lr: float = 1e-3
    weight_lr: float = 3e-2
    temperature_start: float = 0.1
    temperature_end: float = 0.01
    gradient_lambda: float = 1.0
    diversity_lambda: float = 0.05
    candidate_max: int = 2048
    num_groups: int = 256
    cluster_dim: int = 64
    cluster_steps: int = 5
    proxy_scope: str = "aggregator_head"
    seed: int = 1

    @classmethod
    def from_args(cls, args):
        def value(name, default):
            return getattr(args, name, default)

        return cls(
            variant=value("kem_variant", "anchored"),
            anchor_ratio=float(value("kem_anchor_ratio", 0.5)),
            anchor_method=value("kem_anchor_method", "maxminrand"),
            candidate_method=value("kem_candidate_method", "maxminrand"),
            candidate_ratio=float(value("kem_candidate_ratio", 4.0)),
            backend=value("kem_backend", "auto"),
            outer_steps=int(value("kem_outer_steps", 15)),
            inner_steps=int(value("bilevel_inner_steps", 1)),
            proxy_lr=float(value("bilevel_proxy_lr", 1e-3)),
            weight_lr=float(value("kem_weight_lr", 3e-2)),
            temperature_start=float(value("kem_temperature_start", 0.1)),
            temperature_end=float(value("kem_temperature_end", 0.01)),
            gradient_lambda=float(value("kem_gradient_lambda", 1.0)),
            diversity_lambda=float(value("kem_diversity_lambda", 0.05)),
            candidate_max=int(value("bilevel_candidate_max", 2048)),
            num_groups=int(value("bilevel_num_groups", 256)),
            cluster_dim=int(value("bilevel_cluster_dim", 64)),
            cluster_steps=int(value("bilevel_cluster_steps", 5)),
            proxy_scope=value("bilevel_proxy_scope", "aggregator_head"),
            seed=int(value("seed", 1)),
        )

    def __post_init__(self):
        if self.variant not in {"pure", "anchored"}:
            raise ValueError(f"Unsupported kem_variant={self.variant!r}")
        if self.candidate_method != "maxminrand":
            raise ValueError("KEM v2 currently requires maxminrand candidates")
        if not 0 <= self.anchor_ratio < 1:
            raise ValueError("kem_anchor_ratio must be in [0, 1)")
        if self.anchor_method not in {"maxminrand", "random"}:
            raise ValueError("kem_anchor_method must be maxminrand or random")
        if self.candidate_ratio < 1 or self.outer_steps < 1:
            raise ValueError("Invalid KEM v2 candidate ratio or iteration count")


def _active_logits(logits, seen_classes):
    if seen_classes is None:
        return logits
    seen = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
    unseen = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
    unseen[seen] = False
    return logits.masked_fill(unseen.unsqueeze(0), -100.0)


def _forward(model, model_name, features, seen_classes, params=None, patch_mask=None, detailed=False):
    if model_name.startswith("clam_"):
        kwargs = {
            "return_features": True,
            "seen_classes": seen_classes,
            "patch_mask": patch_mask,
        }
    elif model_name == "acmil":
        kwargs = {"patch_mask": patch_mask, "seen_classes": seen_classes}
    elif model_name == "transmil":
        kwargs = {"patch_mask": patch_mask}
        if detailed:
            kwargs["return_attn"] = True
    elif model_name == "mambamil":
        if patch_mask is not None:
            raise ValueError("MambaMIL KEM must use gradient matching, not patch_mask")
        kwargs = {}
    else:
        raise ValueError(f"No explicit KEM forward contract for {model_name!r}")
    output = functional_call(model, params, (features,), kwargs) if params is not None else model(features, **kwargs)
    output = dict(output)
    if "A" not in output and "A_raw" in output:
        output["A"] = output["A_raw"]
    output["logits"] = _active_logits(output["logits"], seen_classes)
    return output


def _detach_output(output):
    return {key: value.detach() for key, value in output.items() if torch.is_tensor(value)}


def _preservation_loss(output, teacher, label, model_name):
    if model_name == "acmil":
        loss, _ = get_training_loss("acmil").compute(output, label)
    else:
        loss = F.cross_entropy(output["logits"], label)
    distillation, _ = replay_distillation_loss(model_name, output, teacher)
    return loss + distillation


def _standardized(values):
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _anchor_indices(scores, patch_count, budget, config, device):
    if config.variant != "anchored" or budget <= 1:
        return torch.empty(0, dtype=torch.long, device=device)
    count = min(budget - 1, int(round(budget * config.anchor_ratio)))
    if config.anchor_method == "random":
        generator = torch.Generator(device=device).manual_seed(config.seed)
        return torch.randperm(patch_count, device=device, generator=generator)[:count]
    return stratified_attention_indices(
        scores, patch_count, count, "maxminrand", device, seed=config.seed
    )


def _fallback_indices(scores, patch_count, budget, config, device):
    """Return the same-budget baseline associated with the configured anchor."""
    if config.anchor_method == "random":
        generator = torch.Generator(device=device).manual_seed(config.seed)
        return torch.randperm(patch_count, device=device, generator=generator)[:budget]
    return stratified_attention_indices(
        scores, patch_count, budget, "maxminrand", device, seed=config.seed
    )


def _merge_selection(anchor, learned, patch_count, budget, device, seed=1):
    selected = torch.unique(torch.cat([anchor.to(device), learned.to(device)]), sorted=False)
    if selected.numel() < budget:
        available = torch.ones(patch_count, dtype=torch.bool, device=device)
        available[selected] = False
        pool = torch.nonzero(available, as_tuple=False).flatten()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        selected = torch.cat([selected, pool[torch.randperm(
            pool.numel(), device=device, generator=generator
        )[:budget-selected.numel()]]])
    return selected[:budget]


class KemV2Selector:
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.config = KemV2Config.from_args(args)
        backend = self.config.backend
        if backend == "auto":
            backend = "bilevel" if model_name.startswith("clam_") or model_name == "acmil" else "gradient_match"
        if backend not in {"bilevel", "gradient_match"}:
            raise ValueError(f"Unsupported kem_backend={backend!r}")
        self.backend = backend
        self.adapter = ReplayModelAdapter(model_name)
        self.region_config = RegionConfig.from_args(args)
        self.region_supervisor = RegionSupervisor(self.region_config, args) if self.region_config.enabled else None
        self._region_occurrences = {}

    def select(self, model, features, label, seen_classes, budget, scores,
               coords=None, slide_id=None, task=None):
        if model is None or label is None:
            raise ValueError("kem_v2 selection requires model and label")
        if scores is None:
            raise ValueError("kem_v2 requires attention scores; refusing random fallback")
        if self.backend == "bilevel":
            return self._select_clam(model, features, label, seen_classes, budget, scores,
                                     coords=coords, slide_id=slide_id, task=task)
        if self.region_config.enabled and self.model_name.startswith("clam_"):
            return self._select_clam(model, features, label, seen_classes, budget, scores,
                                     coords=coords, slide_id=slide_id, task=task)
        return self._select_transmil(
            model, features, label, seen_classes, budget, scores,
            coords=coords, slide_id=slide_id, task=task,
        )

    def _candidate_pool(self, features, scores, budget):
        legacy = BilevelConfig(
            prefilter="maxminrand",
            coarse_ratio=self.config.candidate_ratio,
            candidate_max=self.config.candidate_max,
            num_groups=self.config.num_groups,
            cluster_dim=self.config.cluster_dim,
            cluster_steps=self.config.cluster_steps,
            seed=self.config.seed,
        )
        return build_candidate_pool(features, scores, budget, legacy)

    def _select_clam(self, model, features, label, seen_classes, budget, scores,
                     coords=None, slide_id=None, task=None):
        started = time.perf_counter()
        device = next(model.parameters()).device
        features = features.detach().to(device)
        scores = scores.detach().reshape(-1).to(device)
        label = label.detach().long().view(-1).to(device)
        budget = min(max(int(budget), 1), features.shape[0])
        region_data = None
        if self.region_supervisor is not None:
            if slide_id is None or task is None:
                raise ValueError("region supervision requires slide_id and task")
            region_data = self.region_supervisor.prepare(
                model, features, coords, seen_classes, str(slide_id), int(task), self.model_name
            )
        anchor = _anchor_indices(scores, features.shape[0], budget, self.config, device)
        learned_budget = budget - anchor.numel()
        pool = self._candidate_pool(features, scores, max(learned_budget, 1))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            teacher_full = _detach_output(_forward(model, self.model_name, features, seen_classes, detailed=True))
            teacher_candidate = _detach_output(_forward(model, self.model_name, pool.features, seen_classes, detailed=True))

        proxy = copy.deepcopy(model).to(device).eval()
        self.adapter.configure_proxy_parameters(proxy, self.config.proxy_scope)
        base_params = OrderedDict((name, value) for name, value in proxy.named_parameters() if value.requires_grad)
        reference_out = _forward(
            proxy, self.model_name, features, seen_classes,
            params=base_params, detailed=True,
        )
        if self.model_name == "acmil":
            reference_loss, _ = get_training_loss("acmil").compute(reference_out, label)
        else:
            reference_loss = F.cross_entropy(reference_out["logits"], label)
        reference_grads = tuple(
            gradient.detach() for gradient in _grads(reference_loss, base_params)
        )
        flat_reference_grad = torch.cat([gradient.reshape(-1) for gradient in reference_grads])
        del reference_out, reference_loss

        if pool.mode == "patch":
            prior = scores.index_select(0, pool.source_indices)
        else:
            local_scores = scores.index_select(0, pool.source_indices)
            prior = torch.zeros(pool.features.shape[0], device=device)
            prior.index_add_(0, pool.assignments.to(device), local_scores)
            counts = torch.bincount(pool.assignments.to(device), minlength=pool.features.shape[0]).clamp_min(1)
            prior = prior / counts
        generator = torch.Generator(device=device).manual_seed(self.config.seed)
        initial = 0.01 * _standardized(prior) + 1e-3 * torch.randn(prior.shape, device=device, generator=generator)
        weights = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([weights], lr=self.config.weight_lr)
        last_inner = last_outer = last_grad = last_gradient_match = last_soft = None

        for step in range(self.config.outer_steps):
            ratio = step / max(self.config.outer_steps - 1, 1)
            temperature = self.config.temperature_start * (1-ratio) + self.config.temperature_end * ratio
            soft = _soft_exact_budget(weights, min(learned_budget, weights.numel()), temperature)
            last_soft = soft
            hard = torch.zeros_like(weights)
            hard[torch.topk(weights, min(learned_budget, weights.numel())).indices] = 1.0
            mask = hard + soft - soft.detach()
            params = base_params
            for _ in range(self.config.inner_steps):
                candidate_out = _forward(
                    proxy, self.model_name, pool.features, seen_classes,
                    params=params, patch_mask=mask, detailed=True,
                )
                inner = _preservation_loss(candidate_out, teacher_candidate, label, self.model_name)
                grads = _grads(inner, params, create_graph=True)
                params = OrderedDict(
                    (name, value - self.config.proxy_lr * grad)
                    for (name, value), grad in zip(params.items(), grads)
                )
            full_out = _forward(proxy, self.model_name, features, seen_classes, params=params, detailed=True)
            outer = _preservation_loss(full_out, teacher_full, label, self.model_name)
            flat_candidate_grad = torch.cat([gradient.reshape(-1) for gradient in grads])
            gradient_match = 1.0 - F.cosine_similarity(
                flat_candidate_grad, flat_reference_grad, dim=0, eps=1e-8
            )
            probabilities = soft / soft.sum().clamp_min(1e-8)
            normalized_candidates = F.normalize(pool.features.float(), dim=1)
            diversity = (probabilities @ normalized_candidates).square().sum()
            outer = outer + self.config.gradient_lambda * gradient_match
            outer = outer + self.config.diversity_lambda * diversity
            if region_data is not None:
                registered_region = get_replay_adapter(self.model_name).region_forward
                if registered_region is not None:
                    region_logits = registered_region(
                        proxy, region_data["padded"], region_data["mask"],
                        seen_classes, params=params,
                    )
                else:
                    adapter = (RegionBatchTransMILAdapter if self.model_name == "transmil"
                               else RegionBatchCLAMAdapter)
                    region_logits = adapter.forward(proxy, region_data["padded"],
                                                    region_data["mask"], seen_classes,
                                                    params=params)
                region_loss = weighted_region_kl(
                    region_logits, region_data["teacher_probs"],
                    region_data["confidence"], seen_classes,
                    self.region_config.teacher_temperature,
                )
                outer = outer + self.region_config.loss_lambda * region_loss
            else:
                region_loss = None
            gradient = torch.autograd.grad(outer, weights)[0]
            optimizer.zero_grad(set_to_none=True)
            weights.grad = gradient
            optimizer.step()
            last_inner, last_outer, last_grad = inner, outer, gradient
            last_gradient_match = gradient_match

        learned = pool.finalize(weights.detach(), learned_budget).to(device)
        selected = _merge_selection(
            anchor, learned, features.shape[0], budget, device, self.config.seed
        )
        fallback = False
        baseline = _fallback_indices(
            scores, features.shape[0], budget, self.config, device
        )
        if self.config.variant == "anchored":
            learned_outer = self._hard_subset_outer(
                proxy, base_params, model, features, selected, label,
                seen_classes, teacher_full,
            )
            baseline_outer = self._hard_subset_outer(
                proxy, base_params, model, features, baseline, label,
                seen_classes, teacher_full,
            )
            if learned_outer > baseline_outer:
                selected = baseline
                fallback = True
        if was_training:
            model.train()
        diagnostics = {
            "kem_v2/backend_bilevel": 1.0,
            "kem_v2/anchor_count": float(anchor.numel()),
            "kem_v2/learned_count": float(budget-anchor.numel()),
            "kem_v2/candidate_count": float(pool.source_indices.numel()),
            "kem_v2/group_mode": float(pool.mode == "group"),
            "kem_v2/inner_loss": float(last_inner.detach()),
            "kem_v2/outer_loss": float(last_outer.detach()),
            "kem_v2/gradient_norm": float(last_grad.detach().norm()),
            "kem_v2/gradient_cosine": float((1.0-last_gradient_match).detach()),
            "kem_v2/weight_std": float(weights.detach().std(unbiased=False)),
            "kem_v2/fallback_to_baseline": float(fallback),
            "kem_v2/random_anchor": float(self.config.anchor_method == "random"),
            "kem_v2/selection_time_seconds": time.perf_counter()-started,
        }
        if self.model_name == "acmil" and "A_tokens_normalized" in teacher_full:
            token_attention = teacher_full["A_tokens_normalized"].squeeze(0)
            token_pairs = []
            for left in range(token_attention.shape[0]):
                for right in range(left + 1, token_attention.shape[0]):
                    token_pairs.append(F.cosine_similarity(
                        token_attention[left], token_attention[right], dim=0
                    ))
            diagnostics.update({
                "acmil/token_attention_std": float(token_attention.std(unbiased=False)),
                "acmil/token_attention_pair_cosine": float(
                    torch.stack(token_pairs).mean() if token_pairs
                    else token_attention.new_zeros(())
                ),
            })
        if region_data is not None:
            selected_regions = region_data["membership"].index_select(0, selected.cpu())
            counts = torch.bincount(selected_regions, minlength=len(region_data["regions"]))
            soft_patch_mass = torch.zeros(features.shape[0], dtype=last_soft.dtype)
            source_indices = pool.source_indices.cpu()
            if pool.mode == "patch":
                soft_patch_mass.index_add_(0, source_indices, last_soft.detach().cpu())
            else:
                assignments = pool.assignments.cpu()
                capacities = torch.bincount(assignments, minlength=last_soft.numel()).clamp_min(1)
                distributed = last_soft.detach().cpu().index_select(0, assignments) / capacities.index_select(0, assignments)
                soft_patch_mass.index_add_(0, source_indices, distributed)
            region_soft_mass = torch.zeros(len(region_data["regions"]), dtype=last_soft.dtype)
            region_soft_mass.index_add_(0, region_data["membership"], soft_patch_mass)
            probs = region_data["teacher_probs"]
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            active_classes = torch.as_tensor(seen_classes, device=probs.device, dtype=torch.long)
            teacher_classes = active_classes.index_select(0, probs.argmax(dim=-1))
            label_agreement = (teacher_classes == int(label.item())).float().mean()
            diagnostics.update({
                "region/loss": float(region_loss.detach()),
                "region/count": float(len(region_data["regions"])),
                "region/effective_selected": float((counts > 0).sum()),
                "region/teacher_entropy": float(entropy.mean()),
                "region/teacher_confidence": float(region_data["confidence"].mean()),
                "region/label_agreement": float(label_agreement),
                "region/cache_hit": float(region_data["cache_hit"]),
                "region/prepare_seconds": float(region_data["prepare_seconds"]),
                "region/max_gpu_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
            })
            occurrence_key = (int(task), str(slide_id))
            occurrence = self._region_occurrences.get(occurrence_key, 0)
            self._region_occurrences[occurrence_key] = occurrence + 1
            artifact = (Path(self.region_config.cache_dir) / "selections" / f"task_{int(task)}"
                        / f"{slide_id}__occ{occurrence:03d}.pt")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".tmp")
            artifact_payload = {
                "slide_id": str(slide_id), "task": int(task),
                "occurrence": occurrence,
                "partition": self.region_config.partition,
                "model": self.model_name,
                "membership": region_data["membership"],
                "regions": region_data["regions"],
                "teacher_logits": region_data["teacher_logits"].cpu(),
                "teacher_probs": region_data["teacher_probs"].cpu(),
                "confidence": region_data["confidence"].cpu(),
                "label": int(label.item()),
                "coords": region_data["coords"],
                "selected_indices": selected.cpu(),
                "region_selected_count": counts.cpu(),
                "region_soft_mass": region_soft_mass.cpu(),
            }
            if self.model_name == "acmil" and "A_tokens_normalized" in teacher_full:
                full_token_attention = teacher_full["A_tokens_normalized"].cpu()
                artifact_payload.update({
                    "teacher_token_attention": full_token_attention,
                    "selected_token_attention": full_token_attention.index_select(
                        -1, selected.cpu()
                    ),
                })
            torch.save(artifact_payload, temporary)
            temporary.replace(artifact)
        return selected.cpu(), diagnostics

    def _hard_subset_outer(
        self, proxy, base_params, teacher_model, full_features, indices,
        label, seen_classes, teacher_full,
    ):
        subset = full_features.index_select(0, indices)
        with torch.no_grad():
            subset_teacher = _detach_output(
                _forward(teacher_model, self.model_name, subset, seen_classes, detailed=True)
            )
        params = base_params
        for _ in range(self.config.inner_steps):
            output = _forward(
                proxy, self.model_name, subset, seen_classes,
                params=params, detailed=True,
            )
            inner = _preservation_loss(output, subset_teacher, label, self.model_name)
            gradients = _grads(inner, params)
            params = OrderedDict(
                (name, value - self.config.proxy_lr * gradient)
                for (name, value), gradient in zip(params.items(), gradients)
            )
        with torch.no_grad():
            full_output = _forward(
                proxy, self.model_name, full_features, seen_classes,
                params=params, detailed=True,
            )
            return float(_preservation_loss(
                full_output, teacher_full, label, self.model_name
            ).detach())

    def _select_transmil(self, model, features, label, seen_classes, budget, scores,
                         coords=None, slide_id=None, task=None):
        started = time.perf_counter()
        device = next(model.parameters()).device
        features = features.detach().to(device)
        scores = scores.detach().reshape(-1).to(device)
        label = label.detach().long().view(-1).to(device)
        budget = min(max(int(budget), 1), features.shape[0])
        anchor = _anchor_indices(scores, features.shape[0], budget, self.config, device)
        learned_budget = budget - anchor.numel()
        candidate_count = min(features.shape[0], max(budget, int(round(budget*self.config.candidate_ratio))))
        candidate_indices = stratified_attention_indices(
            scores, features.shape[0], candidate_count, "maxminrand", device,
            seed=self.config.seed,
        )

        was_training = model.training
        model.eval()
        region_data = None
        if self.region_supervisor is not None:
            if self.model_name != "mambamil":
                raise ValueError(
                    "gradient-based region supervision is currently implemented for MambaMIL only"
                )
            if slide_id is None or task is None:
                raise ValueError("region supervision requires slide_id and task")
            region_data = self.region_supervisor.prepare(
                model, features, coords, seen_classes, str(slide_id), int(task)
            )
        x = features.detach().clone().requires_grad_(True)
        output = _forward(model, self.model_name, x, seen_classes, detailed=True)
        full_loss = F.cross_entropy(output["logits"], label)
        input_gradient = torch.autograd.grad(full_loss, x)[0]
        combined_gradient = input_gradient
        region_loss = None
        region_gradient = None
        if region_data is not None:
            region_padded, region_mask = pad_regions(x, region_data["regions"])
            region_logits = RegionBatchMambaMILAdapter.forward(
                model, region_padded, region_mask, seen_classes
            )
            region_loss = weighted_region_kl(
                region_logits, region_data["teacher_probs"],
                region_data["confidence"], seen_classes,
                self.region_config.teacher_temperature,
                student_temperature=1.0,
            )
            region_gradient = torch.autograd.grad(region_loss, x)[0]
            combined_gradient = input_gradient + self.region_config.loss_lambda * region_gradient
        signatures = (combined_gradient * x).detach()
        saliency = signatures.norm(dim=1)
        if signatures.shape[1] > self.config.cluster_dim:
            generator = torch.Generator(device=device).manual_seed(self.config.seed)
            projection = torch.randn(
                signatures.shape[1], self.config.cluster_dim, device=device, generator=generator
            ) / self.config.cluster_dim ** 0.5
            signatures = signatures @ projection
        signatures = F.normalize(signatures, dim=1)
        target = F.normalize(signatures.mean(dim=0), dim=0)

        candidate_features = features.index_select(0, candidate_indices)
        groups = min(self.config.num_groups, candidate_features.shape[0])
        assignments, prototypes = _cosine_kmeans(
            candidate_features, groups, self.config.cluster_dim,
            self.config.cluster_steps, self.config.seed,
        )
        candidate_signatures = signatures.index_select(0, candidate_indices)
        group_signatures = torch.zeros(groups, candidate_signatures.shape[1], device=device)
        group_signatures.index_add_(0, assignments, candidate_signatures)
        capacities = torch.bincount(assignments, minlength=groups)
        group_signatures = F.normalize(group_signatures / capacities.clamp_min(1).unsqueeze(1), dim=1)
        candidate_attention = scores.index_select(0, candidate_indices)
        candidate_saliency = saliency.index_select(0, candidate_indices)
        group_attention = torch.zeros(groups, device=device).index_add_(0, assignments, candidate_attention)
        group_saliency = torch.zeros(groups, device=device).index_add_(0, assignments, candidate_saliency)
        group_attention = _standardized(group_attention / capacities.clamp_min(1))
        group_saliency = _standardized(group_saliency / capacities.clamp_min(1))

        anchor_mask = torch.isin(candidate_indices, anchor)
        available_capacity = capacities.clone()
        if anchor_mask.any():
            available_capacity.index_add_(0, assignments[anchor_mask], -torch.ones_like(assignments[anchor_mask]))
        quotas = torch.zeros(groups, dtype=torch.long, device=device)
        selected_signature = torch.zeros_like(target)
        for _ in range(learned_budget):
            residual = F.normalize(target - selected_signature / max(learned_budget, 1), dim=0)
            match = group_signatures @ residual
            group_score = match + 0.15*group_attention + 0.15*group_saliency
            group_score[quotas >= available_capacity] = -torch.inf
            group = int(group_score.argmax())
            if not torch.isfinite(group_score[group]):
                break
            quotas[group] += 1
            selected_signature += group_signatures[group]

        learned_local = []
        normalized_attention = _standardized(candidate_attention)
        normalized_saliency = _standardized(candidate_saliency)
        normalized_features = F.normalize(candidate_features.float(), dim=1)
        for group in range(groups):
            quota = int(quotas[group])
            if quota == 0:
                continue
            members = torch.nonzero((assignments == group) & ~anchor_mask, as_tuple=False).flatten()
            if members.numel() == 0:
                continue
            representative = normalized_features[members] @ F.normalize(prototypes[group].float(), dim=0)
            local_score = normalized_attention[members] + normalized_saliency[members]
            local_score = local_score + self.config.diversity_lambda * representative
            learned_local.append(members[torch.topk(local_score, min(quota, members.numel())).indices])
        learned_local = torch.cat(learned_local) if learned_local else torch.empty(0, dtype=torch.long, device=device)
        learned = candidate_indices.index_select(0, learned_local)
        selected = _merge_selection(
            anchor, learned, features.shape[0], budget, device, self.config.seed
        )
        if self.model_name == "mambamil":
            # Replay must preserve the original WSI patch sequence.  Returning
            # selection-score order would change Mamba state propagation.
            selected = selected.sort().values
        selected_signature_mean = F.normalize(signatures.index_select(0, selected).mean(dim=0), dim=0)
        gradient_cosine = F.cosine_similarity(selected_signature_mean, target, dim=0)
        attention_mass = scores.softmax(dim=0).index_select(0, selected).sum()
        if was_training:
            model.train()
        diagnostics = {
            "kem_v2/backend_gradient_match": 1.0,
            "kem_v2/anchor_count": float(anchor.numel()),
            "kem_v2/learned_count": float(budget-anchor.numel()),
            "kem_v2/candidate_count": float(candidate_indices.numel()),
            "kem_v2/group_count": float(groups),
            "kem_v2/gradient_cosine": float(gradient_cosine.detach()),
            "kem_v2/attention_mass": float(attention_mass.detach()),
            "kem_v2/selection_time_seconds": time.perf_counter()-started,
        }
        if region_data is not None:
            selected_regions = region_data["membership"].index_select(0, selected.cpu())
            counts = torch.bincount(selected_regions, minlength=len(region_data["regions"]))
            gradient_mass = (region_gradient * x).detach().norm(dim=1).cpu()
            region_gradient_mass = torch.zeros(len(region_data["regions"]))
            region_gradient_mass.index_add_(0, region_data["membership"], gradient_mass)
            probabilities = region_data["teacher_probs"]
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            active_classes = torch.as_tensor(seen_classes, device=probabilities.device, dtype=torch.long)
            teacher_classes = active_classes.index_select(0, probabilities.argmax(dim=-1))
            diagnostics.update({
                "region/loss": float(region_loss.detach()),
                "region/count": float(len(region_data["regions"])),
                "region/effective_selected": float((counts > 0).sum()),
                "region/teacher_entropy": float(entropy.mean()),
                "region/teacher_confidence": float(region_data["confidence"].mean()),
                "region/label_agreement": float((teacher_classes == int(label.item())).float().mean()),
                "region/cache_hit": float(region_data["cache_hit"]),
                "region/prepare_seconds": float(region_data["prepare_seconds"]),
                "region/full_gradient_norm": float(input_gradient.detach().norm()),
                "region/region_gradient_norm": float(region_gradient.detach().norm()),
                "region/combined_gradient_norm": float(combined_gradient.detach().norm()),
                "region/gradient_cosine": float(F.cosine_similarity(
                    input_gradient.detach().reshape(-1),
                    region_gradient.detach().reshape(-1), dim=0, eps=1e-8,
                )),
            })
            occurrence_key = (int(task), str(slide_id))
            occurrence = self._region_occurrences.get(occurrence_key, 0)
            self._region_occurrences[occurrence_key] = occurrence + 1
            artifact = (Path(self.region_config.cache_dir) / "selections" / f"task_{int(task)}"
                        / f"{slide_id}__occ{occurrence:03d}.pt")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".tmp")
            torch.save({
                "schema_version": 2,
                "slide_id": str(slide_id), "task": int(task), "occurrence": occurrence,
                "partition": self.region_config.partition, "model": self.model_name,
                "ordering_policy": region_data["ordering_policy"],
                "ordering_hash": region_data["ordering_hash"],
                "membership": region_data["membership"], "regions": region_data["regions"],
                "teacher_logits": region_data["teacher_logits"].cpu(),
                "teacher_probs": region_data["teacher_probs"].cpu(),
                "confidence": region_data["confidence"].cpu(), "label": int(label.item()),
                "coords": region_data["coords"], "selected_indices": selected.cpu(),
                "region_selected_count": counts.cpu(),
                "region_gradient_mass": region_gradient_mass,
                "full_gradient_norm": float(input_gradient.detach().norm()),
                "region_gradient_norm": float(region_gradient.detach().norm()),
                "combined_gradient_norm": float(combined_gradient.detach().norm()),
            }, temporary)
            temporary.replace(artifact)
        return selected.cpu(), diagnostics
