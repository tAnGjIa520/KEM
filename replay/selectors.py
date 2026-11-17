"""Unified replay patch selector API."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch

from .bilevel_selector import BilevelConfig, BilevelReplaySelector
from .kem_v2 import KemV2Selector
from .kem_v4 import KemV4Selector
from .pseudobag_mmil import build_mmil_pseudobag_selector
from .pseudobag_pmil import PMILSelector
from .pseudobag_prodiv import ProDivSelector
from .pseudobag_psemix import PseMixSelector


@dataclass(frozen=True)
class SelectionResult:
    indices: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)
    partition_ids: torch.Tensor | None = None


class ReplaySelector:
    def __init__(self, method: str, model_name: str, args):
        self.method = method
        self.bilevel = None
        self.kem_v2 = None
        self.kem_v4 = None
        self.pseudobag = None
        if method == "kem":
            config = BilevelConfig(
                solver=args.bilevel_solver,
                proxy_scope=args.bilevel_proxy_scope,
                prefilter=args.bilevel_prefilter,
                coarse_ratio=args.bilevel_coarse_ratio,
                candidate_max=args.bilevel_candidate_max,
                num_groups=args.bilevel_num_groups,
                cluster_dim=args.bilevel_cluster_dim,
                cluster_steps=args.bilevel_cluster_steps,
                outer_steps=args.bilevel_outer_steps,
                inner_steps=args.bilevel_inner_steps,
                proxy_lr=args.bilevel_proxy_lr,
                weight_lr=args.bilevel_weight_lr,
                distill_lambda=args.bilevel_distill_lambda,
                feature_lambda=args.bilevel_feature_lambda,
                temperature=args.bilevel_temperature,
                neumann_steps=args.bilevel_neumann_steps,
                damping=args.bilevel_damping,
                nystrom_rank=args.bilevel_nystrom_rank,
                bome_eta=args.bilevel_bome_eta,
                seed=args.seed,
            )
            self.bilevel = BilevelReplaySelector(model_name, config)
        elif method == "kem_v2":
            self.kem_v2 = KemV2Selector(model_name, args)
        elif method == "kem_v4":
            self.kem_v4 = KemV4Selector(model_name, args)
        elif method in {"spatial", "mmil_random", "mmil_embed"}:
            self.pseudobag = build_mmil_pseudobag_selector(
                method,
                num_groups=int(getattr(args, "mmil_num_subbags", 4)),
                seed=int(getattr(args, "seed", 1)),
                kmeans_steps=int(getattr(args, "mmil_kmeans_steps", 8)),
                cluster_dim=int(getattr(args, "mmil_cluster_dim", 64)),
            )
        elif method == "prodiv_selector":
            self.pseudobag = ProDivSelector(
                group_count=int(getattr(args, "prodiv_num_strata", 4)),
            )
        elif method == "psemix_selector":
            self.pseudobag = PseMixSelector(
                group_count=int(getattr(args, "psemix_num_phenotypes", 8)),
                refinement_steps=int(getattr(args, "psemix_refine_steps", 8)),
            )
        elif method in {"pmil_attention", "pmil_shapley"}:
            self.pseudobag = PMILSelector.from_args(method, args)

    def select(self, features, budget, scores=None, model=None, label=None,
               seen_classes=None, coords=None, slide_id=None, task=None):
        if features.ndim != 2:
            raise ValueError(f"Expected [patches, features], got {tuple(features.shape)}")
        budget = min(max(int(budget), 1), features.shape[0])
        if self.method == "kem":
            if model is None or label is None:
                raise ValueError("kem selection requires model and label")
            indices, diagnostics = self.bilevel.select(
                model, features, label, seen_classes, budget, scores
            )
            return SelectionResult(indices, diagnostics)
        if self.method == "kem_v2":
            indices, diagnostics = self.kem_v2.select(
                model, features, label, seen_classes, budget, scores,
                coords=coords, slide_id=slide_id, task=task,
            )
            return SelectionResult(indices, diagnostics)
        if self.method == "kem_v4":
            indices, diagnostics = self.kem_v4.select(
                model, features, label, seen_classes, budget, scores=scores,
                coords=coords, slide_id=slide_id, task=task,
            )
            return SelectionResult(indices, diagnostics)
        if self.pseudobag is not None:
            started = time.perf_counter()
            if self.method in {"spatial", "mmil_random", "mmil_embed"}:
                result = self.pseudobag.select(
                    features, budget, coords=coords, slide_id=slide_id, task=task,
                )
            else:
                result = self.pseudobag.select(
                    features, budget, scores=scores, model=model, label=label,
                    seen_classes=seen_classes, coords=coords, slide_id=slide_id, task=task,
                )
            diagnostics = dict(result.diagnostics)
            diagnostics["selector/selection_time_seconds"] = time.perf_counter() - started
            for source in (
                "pseudobag/partition_count", "prodiv/group_count",
                "psemix/group_count", "partition_count",
            ):
                if source in diagnostics:
                    diagnostics["selector/partition_count"] = float(diagnostics[source])
                    break
            for source in (
                "pseudobag/partition_coverage", "prodiv/group_coverage",
                "psemix/group_coverage", "partition_coverage",
            ):
                if source in diagnostics:
                    diagnostics["selector/partition_coverage"] = float(diagnostics[source])
                    break
            if "effective_rank" in diagnostics:
                diagnostics["selector/selected_effective_rank"] = float(
                    diagnostics["effective_rank"]
                )
            return SelectionResult(
                result.indices, diagnostics,
                partition_ids=getattr(result, "partition_ids", None),
            )

        flat_scores = None if scores is None else scores.detach().reshape(-1).cpu()
        patch_count = features.shape[0]
        if self.method == "random":
            indices = torch.randperm(patch_count)[:budget]
        elif self.method == "max":
            self._require_scores(flat_scores)
            indices = torch.topk(flat_scores, budget).indices
        elif self.method in {"maxmin", "maxminrand", "maxrand"}:
            self._require_scores(flat_scores)
            indices = self._mixed_indices(flat_scores, patch_count, budget)
        else:
            raise ValueError(f"Unknown replay patch selector: {self.method!r}")
        return SelectionResult(indices)

    @staticmethod
    def _require_scores(scores):
        if scores is None:
            raise ValueError("Attention scores are required for this selector")

    def _mixed_indices(self, scores, patch_count, budget):
        if self.method == "maxrand":
            top_count = budget // 2
            random_count = budget - top_count
            groups = [torch.topk(scores, top_count).indices]
        elif self.method == "maxmin":
            top_count = budget // 2
            bottom_count = budget - top_count
            random_count = 0
            groups = [torch.topk(scores, top_count).indices,
                      torch.topk(-scores, bottom_count).indices]
        else:
            top_count = budget // 4
            bottom_count = budget // 4
            random_count = budget - top_count - bottom_count
            groups = [torch.topk(scores, top_count).indices,
                      torch.topk(-scores, bottom_count).indices]
        used = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)
        used = torch.unique(used)
        if random_count or used.numel() < budget:
            available = torch.ones(patch_count, dtype=torch.bool)
            available[used] = False
            pool = torch.nonzero(available, as_tuple=False).flatten()
            need = budget - used.numel()
            random_indices = pool[torch.randperm(pool.numel())[:need]]
            used = torch.cat([used, random_indices])
        return used[:budget]


def build_replay_selector(method: str, model_name: str, args) -> ReplaySelector:
    return ReplaySelector(method, model_name, args)
