"""Fast hierarchical replay selection.

KEM v4 deliberately avoids higher-order bilevel derivatives.  A single
input-gradient pass supplies patch influence, while low-rank leverage,
region reliability and farthest-point coverage protect morphological
diversity.  The same analysis also exposes a concave utility curve used by
the global patch-budget allocator.

The ``logdet`` rule is the coefficient-free alternative. It treats region
reliability as observation precision and greedily maximizes D-optimal
information gain, replacing separately weighted leverage/diversity terms.
"""

from __future__ import annotations

import math
import time
import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .adapters import ReplayModelAdapter
from .candidates import stratified_attention_indices
from .model_registry import get_replay_adapter
from .region_supervision import (
    RegionBatchCLAMAdapter,
    RegionBatchTransMILAdapter,
    RegionConfig,
    pad_regions,
    partition_regions,
)


def _standardize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)


def _random_projection(value: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    if value.shape[-1] <= output_dim:
        return value.float()
    generator = torch.Generator(device=value.device).manual_seed(int(seed))
    matrix = torch.randn(
        value.shape[-1], output_dim, device=value.device, generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(output_dim)
    return value.float() @ matrix


def _effective_rank(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return float(features.shape[0])
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if float(total) <= 0:
        return 0.0
    probabilities = singular / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


@dataclass(frozen=True)
class KemV4Config:
    candidate_max: int = 2048
    projection_dim: int = 64
    max_regions: int = 32
    min_region_size: int = 16
    patch_dropout: float = 0.2
    coverage_ratio: float = 0.7
    influence_weight: float = 0.3
    leverage_weight: float = 0.25
    diversity_weight: float = 0.35
    region_weight: float = 0.20
    attention_weight: float = 0.10
    candidate_pool: str = "maxminrand"
    selection_rule: str = "weighted_sum"
    seed: int = 1

    @classmethod
    def from_args(cls, args):
        return cls(
            candidate_max=int(getattr(args, "kem_v4_candidate_max", 2048)),
            projection_dim=int(getattr(args, "kem_v4_projection_dim", 64)),
            max_regions=int(getattr(args, "kem_v4_max_regions", 32)),
            min_region_size=int(getattr(args, "kem_v4_min_region_size", 16)),
            patch_dropout=float(getattr(args, "kem_v4_patch_dropout", 0.2)),
            coverage_ratio=float(getattr(args, "kem_v4_coverage_ratio", 0.7)),
            influence_weight=float(getattr(args, "kem_v4_influence_weight", 0.3)),
            leverage_weight=float(getattr(args, "kem_v4_leverage_weight", 0.25)),
            diversity_weight=float(getattr(args, "kem_v4_diversity_weight", 0.35)),
            region_weight=float(getattr(args, "kem_v4_region_weight", 0.20)),
            attention_weight=float(getattr(args, "kem_v4_attention_weight", 0.10)),
            candidate_pool=str(getattr(args, "kem_v4_candidate_pool", "maxminrand")),
            selection_rule=str(getattr(args, "kem_v4_selection_rule", "weighted_sum")),
            seed=int(getattr(args, "seed", 1)),
        )

    def __post_init__(self):
        if min(self.candidate_max, self.projection_dim, self.max_regions, self.min_region_size) < 1:
            raise ValueError("KEM v4 size settings must be positive")
        if not 0 <= self.patch_dropout < 0.9:
            raise ValueError("kem_v4_patch_dropout must be in [0, 0.9)")
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError("kem_v4_coverage_ratio must be in [0, 1]")
        if self.candidate_pool not in {"maxminrand", "random"}:
            raise ValueError("kem_v4_candidate_pool must be maxminrand or random")
        if self.selection_rule not in {"weighted_sum", "logdet"}:
            raise ValueError("kem_v4_selection_rule must be weighted_sum or logdet")


@dataclass
class KemV4Analysis:
    ranking: torch.Tensor
    marginal_gains: torch.Tensor
    bag_signature: torch.Tensor
    bag_uncertainty: float
    diagnostics: dict[str, float]
    marginal_gains_are_positive: bool = False

    def allocation_utilities(self, levels) -> tuple[tuple[int, ...], tuple[float, ...]]:
        valid = sorted({min(max(int(level), 1), int(self.ranking.numel())) for level in levels})
        if not valid:
            raise ValueError("allocation levels must not be empty")
        gains = (
            self.marginal_gains.float().clamp_min(0)
            if self.marginal_gains_are_positive
            else F.softplus(self.marginal_gains.float())
        )
        cumulative = gains.cumsum(dim=0)
        denominator = cumulative[-1].clamp_min(1e-8)
        base = 1.0 + float(self.bag_uncertainty) + 0.1 * float(self.bag_signature.norm())
        utilities = []
        for level in valid:
            fidelity = float(cumulative[level - 1] / denominator)
            # The square root creates diminishing marginal returns so that the
            # global allocator can trade WSI coverage against deeper bags.
            utilities.append(base + math.sqrt(max(fidelity, 0.0)))
        return tuple(valid), tuple(utilities)


class KemV4Selector:
    def __init__(self, model_name: str, args):
        self.model_name = model_name
        self.config = KemV4Config.from_args(args)
        self.adapter = ReplayModelAdapter(model_name)

    def _regions(self, features, coords):
        target = max(self.config.min_region_size, math.ceil(features.shape[0] / self.config.max_regions))
        config = RegionConfig(
            enabled=True,
            partition="spatial_balanced_100" if coords is not None else "feature_balanced_100",
            target_patches=target,
            fixed_count=self.config.max_regions,
            min_size=self.config.min_region_size,
            seed=self.config.seed,
        )
        regions = partition_regions(features.detach().cpu(), coords, config)
        if self.model_name == "mambamil":
            regions = [region.sort().values for region in regions]
        return regions

    def _region_forward(self, model, padded, mask, seen_classes):
        hook = get_replay_adapter(self.model_name).region_forward
        if hook is not None:
            return hook(model, padded, mask, seen_classes)
        if self.model_name == "transmil":
            return RegionBatchTransMILAdapter.forward(model, padded, mask, seen_classes)
        return RegionBatchCLAMAdapter.forward(model, padded, mask, seen_classes)

    def _region_reliability(self, model, features, regions, seen_classes):
        if len(regions) == 1 or len(seen_classes) <= 1:
            return features.new_ones(len(regions))
        views = []
        for view in range(2):
            view_regions = []
            for region_index, region in enumerate(regions):
                count = max(1, round(region.numel() * (1.0 - self.config.patch_dropout)))
                generator = torch.Generator().manual_seed(
                    self.config.seed + 104729 * view + 1009 * region_index
                )
                chosen = region[torch.randperm(region.numel(), generator=generator)[:count]].sort().values
                view_regions.append(chosen)
            padded, mask = pad_regions(features, view_regions)
            with torch.no_grad():
                logits = self._region_forward(model, padded, mask, seen_classes)
            active = logits[:, list(seen_classes)]
            views.append(F.softmax(active, dim=-1))
        first, second = views
        mean = (first + second) / 2
        entropy = -(mean * mean.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = 1 - entropy / math.log(mean.shape[-1])
        js = 0.5 * (
            F.kl_div(mean.clamp_min(1e-12).log(), first, reduction="none").sum(dim=-1)
            + F.kl_div(mean.clamp_min(1e-12).log(), second, reduction="none").sum(dim=-1)
        )
        return confidence.clamp(0, 1) * torch.exp(-js.clamp_min(0))

    def _bag_signature(self, output, label, seen_classes):
        active = torch.as_tensor(seen_classes, device=output.logits.device, dtype=torch.long)
        probabilities = F.softmax(output.logits.index_select(1, active), dim=-1).squeeze(0)
        target = torch.zeros_like(probabilities)
        local = torch.nonzero(active == int(label.item()), as_tuple=False).flatten()
        if local.numel() != 1:
            raise ValueError("label must occur exactly once in seen_classes")
        target[local] = 1
        residual = probabilities - target
        embedding = output.features.float().reshape(output.features.shape[0], -1).mean(dim=0)
        signature = torch.outer(residual, F.normalize(embedding, dim=0)).reshape(1, -1)
        signature = _random_projection(signature, self.config.projection_dim, self.config.seed + 17).squeeze(0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        uncertainty = float((entropy / math.log(max(probabilities.numel(), 2))).detach())
        return F.normalize(signature, dim=0).detach(), uncertainty

    @staticmethod
    def _logdet_ranking(projected, reliability, candidate_regions, region_count, budget):
        """Greedy reliability-weighted D-optimal design with region coverage.

        With A_S = I + Σⱼ∈S rⱼ φⱼ φⱼᵀ, the matrix determinant lemma gives
        the exact marginal gain log(1 + rⱼ φⱼᵀ A_S⁻¹ φⱼ). Sherman–Morrison
        updates keep the greedy step inexpensive. Until all represented
        regions are covered, uncovered regions form a hard constraint.
        """
        if projected.ndim != 2 or reliability.shape != (projected.shape[0],):
            raise ValueError("invalid log-det candidate tensors")
        device, dtype = projected.device, projected.dtype
        eps = torch.finfo(dtype).eps
        precision = reliability.to(dtype=dtype).clamp(min=eps, max=1.0)
        vectors = projected * precision.sqrt().unsqueeze(1)
        inverse = torch.eye(projected.shape[1], dtype=dtype, device=device)
        scores = (vectors * vectors).sum(dim=1).clamp_min(0)
        selected_mask = torch.zeros(projected.shape[0], dtype=torch.bool, device=device)
        region_counts = torch.zeros(region_count, dtype=torch.long, device=device)
        represented = torch.unique(candidate_regions)
        selected_local, marginal_gains = [], []

        for _ in range(min(int(budget), projected.shape[0])):
            eligible = ~selected_mask
            uncovered = represented[region_counts.index_select(0, represented) == 0]
            if uncovered.numel():
                eligible &= torch.isin(candidate_regions, uncovered)
            value = scores.masked_fill(~eligible, -torch.inf)
            local = int(value.argmax())
            if not torch.isfinite(value[local]):
                break
            vector = vectors[local]
            inverse_vector = inverse @ vector
            denominator = (1 + vector @ inverse_vector).clamp_min(eps)
            selected_local.append(local)
            marginal_gains.append(torch.log(denominator).detach())
            selected_mask[local] = True
            region_counts[candidate_regions[local]] += 1

            correlations = vectors @ inverse_vector
            scores = (scores - correlations.square() / denominator).clamp_min(0)
            inverse = inverse - torch.outer(inverse_vector, inverse_vector) / denominator
            scores[selected_mask] = -torch.inf

        return selected_local, marginal_gains, region_counts

    def analyze(self, model, features, label, seen_classes, max_budget, scores=None,
                coords=None, deterministic_key=""):
        started = time.perf_counter()
        device = next(model.parameters()).device
        features = features.detach().to(device)
        label = label.detach().long().reshape(-1).to(device)
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("KEM v4 expects non-empty [patches, features]")
        budget = min(max(int(max_budget), 1), features.shape[0])
        flat_scores = None if scores is None else scores.detach().reshape(-1).to(device)
        candidate_count = min(features.shape[0], max(budget, self.config.candidate_max))
        if self.config.candidate_pool == "random":
            payload = f"{self.config.seed}|{deterministic_key}".encode("utf-8")
            local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
            generator = torch.Generator(device=device).manual_seed(local_seed)
            candidates = torch.randperm(
                features.shape[0], generator=generator, device=device,
            )[:candidate_count]
        else:
            candidates = stratified_attention_indices(
                flat_scores, features.shape[0], candidate_count,
                method="maxminrand", device=device, seed=self.config.seed,
            )

        was_training = model.training
        model.eval()
        if self.config.selection_rule == "logdet":
            with torch.no_grad():
                output = self.adapter.forward(model, features, seen_classes)
            input_gradient = None
            influence = features.new_zeros(features.shape[0])
        else:
            x = features.detach().clone().requires_grad_(True)
            output = self.adapter.forward(model, x, seen_classes)
            loss = F.cross_entropy(output.logits, label)
            input_gradient = torch.autograd.grad(loss, x)[0]
            influence = (input_gradient * x).norm(dim=1).detach()
        bag_signature, uncertainty = self._bag_signature(output, label, seen_classes)

        regions = self._regions(features, coords)
        membership = torch.empty(features.shape[0], dtype=torch.long, device=device)
        for index, region in enumerate(regions):
            membership[region.to(device)] = index
        reliability = self._region_reliability(model, features, regions, seen_classes).to(device)

        candidate_features = features.index_select(0, candidates).float()
        projected = F.normalize(
            _random_projection(candidate_features, self.config.projection_dim, self.config.seed + 31), dim=1
        )
        candidate_regions = membership.index_select(0, candidates)
        if self.config.selection_rule == "logdet":
            candidate_reliability = reliability.index_select(0, candidate_regions)
            selected_local, marginal_gains, region_counts = self._logdet_ranking(
                projected, candidate_reliability, candidate_regions,
                len(regions), budget,
            )
            coverage_steps = budget
        else:
            covariance = projected.T @ projected / max(projected.shape[0], 1)
            covariance = covariance + 0.1 * torch.eye(covariance.shape[0], device=device)
            inverse = torch.linalg.inv(covariance)
            leverage = (projected @ inverse * projected).sum(dim=1)
            candidate_influence = _standardize(influence.index_select(0, candidates))
            candidate_leverage = _standardize(leverage)
            candidate_attention = (
                torch.zeros_like(candidate_influence) if flat_scores is None
                else _standardize(flat_scores.index_select(0, candidates))
            )
            candidate_reliability = _standardize(reliability.index_select(0, candidate_regions))
            static_coverage = (
                self.config.leverage_weight * candidate_leverage
                + self.config.region_weight * candidate_reliability
                + self.config.attention_weight * candidate_attention
            )
            static_influence = static_coverage + self.config.influence_weight * candidate_influence

            selected_local = []
            marginal_gains = []
            selected_mask = torch.zeros(candidate_count, dtype=torch.bool, device=device)
            region_counts = torch.zeros(len(regions), device=device)
            min_distance = torch.ones(candidate_count, device=device)
            coverage_steps = round(budget * self.config.coverage_ratio)
            for step in range(budget):
                region_bonus = (1 + region_counts.index_select(0, candidate_regions)).rsqrt()
                static = static_coverage if step < coverage_steps else static_influence
                value = static + self.config.diversity_weight * min_distance
                value = value + self.config.region_weight * region_bonus
                value[selected_mask] = -torch.inf
                local = int(value.argmax())
                if not torch.isfinite(value[local]):
                    break
                selected_local.append(local)
                marginal_gains.append(value[local].detach())
                selected_mask[local] = True
                region_counts[candidate_regions[local]] += 1
                distance = 1 - projected @ projected[local]
                min_distance = torch.minimum(min_distance, distance.clamp_min(0))

        local = torch.as_tensor(selected_local, device=device, dtype=torch.long)
        selected = candidates.index_select(0, local)
        if selected.numel() < budget:
            available = torch.ones(features.shape[0], dtype=torch.bool, device=device)
            available[selected] = False
            fill = torch.nonzero(available, as_tuple=False).flatten()[:budget - selected.numel()]
            selected = torch.cat([selected, fill])
            marginal_gains.extend([features.new_zeros(())] * fill.numel())
        if self.model_name == "mambamil":
            selected = selected.sort().values
        selected_projected = _random_projection(
            features.index_select(0, selected).float(), self.config.projection_dim, self.config.seed + 31
        )
        if was_training:
            model.train()
        diagnostics = {
            "kem_v4/selection_time_seconds": time.perf_counter() - started,
            "kem_v4/candidate_count": float(candidate_count),
            "kem_v4/region_count": float(len(regions)),
            "kem_v4/region_reliability": float(reliability.mean()),
            "kem_v4/bag_uncertainty": float(uncertainty),
            "kem_v4/input_gradient_norm": (
                0.0 if input_gradient is None else float(input_gradient.detach().norm())
            ),
            "kem_v4/selected_effective_rank": _effective_rank(selected_projected),
            "kem_v4/coverage_count": float(min(coverage_steps, budget)),
            "kem_v4/influence_count": float(max(0, budget - coverage_steps)),
            "kem_v4/logdet_rule": float(self.config.selection_rule == "logdet"),
            "kem_v4/covered_regions": float((region_counts > 0).sum()),
        }
        return KemV4Analysis(
            ranking=selected.detach().cpu(),
            marginal_gains=torch.stack(marginal_gains).detach().cpu(),
            bag_signature=bag_signature.detach().cpu(),
            bag_uncertainty=uncertainty,
            diagnostics=diagnostics,
            marginal_gains_are_positive=self.config.selection_rule == "logdet",
        )

    def select(self, model, features, label, seen_classes, budget, scores=None,
               coords=None, slide_id=None, task=None):
        analysis = self.analyze(
            model, features, label, seen_classes, budget, scores=scores, coords=coords,
            deterministic_key=f"{task}|{slide_id}",
        )
        return analysis.ranking[:int(budget)], analysis.diagnostics

    def build_region_targets(self, model, features, seen_classes, coords=None):
        """Create compact, frozen region targets for later replay."""
        device = next(model.parameters()).device
        features = features.detach().to(device)
        regions = self._regions(features, coords)
        reliability = self._region_reliability(model, features, regions, seen_classes)
        padded, mask = pad_regions(features, regions)
        with torch.no_grad():
            logits = self._region_forward(model, padded, mask, seen_classes)
            probabilities = F.softmax(logits[:, list(seen_classes)], dim=-1)
        membership = torch.empty(features.shape[0], dtype=torch.long)
        for index, region in enumerate(regions):
            membership[region] = index
        return {
            "region_membership": membership,
            "region_teacher_probs": probabilities.detach(),
            "region_reliability": reliability.detach(),
            "region_stored_classes": torch.as_tensor(seen_classes, dtype=torch.long),
        }

    def region_replay_loss(self, model, features, batch):
        membership = batch["region_membership"].detach().cpu().long()
        region_count = int(membership.max()) + 1
        regions = [
            torch.nonzero(membership == index, as_tuple=False).flatten()
            for index in range(region_count)
        ]
        if self.model_name == "mambamil":
            regions = [region.sort().values for region in regions]
        padded, mask = pad_regions(features, regions)
        classes = batch["region_stored_classes"].long().reshape(-1).tolist()
        was_training = model.training
        model.eval()
        logits = self._region_forward(model, padded, mask, classes)
        if was_training:
            model.train()
        student = F.log_softmax(logits[:, classes], dim=-1)
        teacher = batch["region_teacher_probs"].to(logits.device)
        reliability = batch["region_reliability"].to(logits.device).clamp_min(0)
        per_region = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
        return (per_region * reliability).sum() / reliability.sum().clamp_min(1e-8)
