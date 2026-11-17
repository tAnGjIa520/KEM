"""Candidate-restricted unified bag--instance replay allocation.

The allocator maximizes class-conditioned facility-location coverage in a
fixed-dimensional projected classifier-gradient space.  It is deliberately a
surrogate and a deterministic greedy approximation, not an exact solver of
future forgetting.  A non-empty option jointly chooses a slide, a nested patch
trajectory, and a depth under one global patch budget.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .adapters import ReplayModelAdapter
from .global_budget_buffer import GlobalPatchBudgetBuffer
from .unified_objectives import deterministic_random_option_score
from .unified_objectives_information import (
    ObjectivePoint as InformationPoint,
    evaluate_objective as evaluate_information_objective,
)


_PROJECTION_CACHE: dict[tuple, torch.Tensor] = {}


def model_state_fingerprint(model):
    """Hash the exact trained parameters used to construct replay views."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stable_seed(seed: int, slide_id: str, suffix: str = "") -> int:
    payload = f"{int(seed)}|{slide_id}|{suffix}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _project_embedding(embedding: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    embedding = embedding.float().reshape(-1)
    key = (
        embedding.device.type, embedding.device.index, embedding.numel(),
        int(output_dim), int(seed),
    )
    projection = _PROJECTION_CACHE.get(key)
    if projection is None:
        generator = torch.Generator(device=embedding.device).manual_seed(int(seed))
        projection = torch.randn(
            embedding.numel(), int(output_dim), generator=generator,
            device=embedding.device, dtype=torch.float32,
        ) / math.sqrt(max(int(output_dim), 1))
        _PROJECTION_CACHE[key] = projection
    return embedding @ projection


def projected_classifier_gradient_signature(
    output, label, seen_classes, n_classes: int, projection_dim: int, seed: int,
) -> torch.Tensor:
    """Exact analytic last-layer gradient after a fixed feature projection.

    The class axis always has ``n_classes`` entries, so signatures created
    before class expansion remain comparable.  Only the representation axis
    is projected.  This corresponds to outer(p-y, projected_embedding).
    """
    logits = output.logits.float()
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != int(n_classes):
        raise ValueError("unified signature expects one bag and fixed global class logits")
    active = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
    if active.numel() < 1 or torch.unique(active).numel() != active.numel():
        raise ValueError("seen_classes must be non-empty and unique")
    target_class = int(torch.as_tensor(label).reshape(-1)[0])
    if int((active == target_class).sum()) != 1:
        raise ValueError("label must occur exactly once in seen_classes")
    active_probabilities = F.softmax(logits.index_select(1, active), dim=1).squeeze(0)
    residual = logits.new_zeros(int(n_classes))
    residual.index_copy_(0, active, active_probabilities)
    residual[target_class] -= 1.0
    embedding = output.features.float().reshape(output.features.shape[0], -1).mean(dim=0)
    projected = _project_embedding(embedding, projection_dim, seed)
    signature = torch.outer(residual, projected).reshape(-1)
    return F.normalize(signature, dim=0, eps=1e-12).detach().cpu()


def signature_for_features(
    model, model_name: str, features: torch.Tensor, label, seen_classes,
    n_classes: int, projection_dim: int, seed: int,
) -> torch.Tensor:
    adapter = ReplayModelAdapter(model_name)
    device = next(model.parameters()).device
    with torch.no_grad():
        output = adapter.forward(model, features.detach().to(device), seen_classes)
    return projected_classifier_gradient_signature(
        output, label, seen_classes, n_classes, projection_dim, seed,
    )


def _fixed_probabilities(output, seen_classes, n_classes):
    active = torch.as_tensor(seen_classes, device=output.logits.device, dtype=torch.long)
    probabilities = output.logits.new_zeros(int(n_classes), dtype=torch.float32)
    probabilities.index_copy_(
        0, active, F.softmax(output.logits.float().index_select(1, active), dim=1).squeeze(0),
    )
    return probabilities


def frozen_feature_signature(features, projection_dim, seed, mode="mean_std"):
    frozen = features.float()
    mean = frozen.mean(dim=0)
    std = frozen.std(dim=0, unbiased=False)
    if mode == "mean_std":
        value = torch.cat([mean, std])
    elif mode == "mean":
        value = torch.cat([mean, torch.zeros_like(std)])
    elif mode == "std":
        value = torch.cat([torch.zeros_like(mean), std])
    else:
        raise ValueError("frozen_signature_mode must be mean_std, mean, or std")
    return F.normalize(
        _project_embedding(value, projection_dim, seed + 991), dim=0, eps=1e-12,
    )


def replay_option_views(
    model, model_name: str, features: torch.Tensor, label, seen_classes,
    n_classes: int, projection_dim: int, seed: int,
    frozen_signature_mode: str = "mean_std",
) -> dict[str, torch.Tensor]:
    """Compute all objective views in one no-grad model forward."""
    adapter = ReplayModelAdapter(model_name)
    device = next(model.parameters()).device
    features = features.detach().to(device)
    with torch.no_grad():
        output = adapter.forward(model, features, seen_classes)
    probabilities = _fixed_probabilities(output, seen_classes, n_classes)
    active = torch.as_tensor(seen_classes, device=device, dtype=torch.long)
    active_probabilities = probabilities.index_select(0, active)
    entropy = -(active_probabilities * active_probabilities.clamp_min(1e-12).log()).sum()
    entropy = entropy / math.log(max(active.numel(), 2))
    embedding = output.features.float().reshape(output.features.shape[0], -1).mean(dim=0)
    projected_embedding = F.normalize(
        _project_embedding(embedding, projection_dim, seed + 313), dim=0, eps=1e-12,
    )
    # Component ablation: replace the coverage signature only.  Trajectory,
    # facility objective and assignment-mass weighting retain their base rules.
    # Historical signatures remain stored at admission time, as in the base.
    if frozen_signature_mode == "trainable_embedding":
        frozen_signature = projected_embedding
    else:
        frozen_signature = frozen_feature_signature(
            features, projection_dim, seed, frozen_signature_mode,
        )
    gradient = projected_classifier_gradient_signature(
        output, label, seen_classes, n_classes, projection_dim, seed,
    )
    gradient_information = F.normalize(
        _project_embedding(gradient.to(device), projection_dim, seed + 1613),
        dim=0, eps=1e-12,
    )
    return {
        "gradient": gradient,
        "gradient_information": gradient_information.detach().cpu(),
        "probability": probabilities.detach().cpu(),
        "embedding": projected_embedding.detach().cpu(),
        "frozen_feature": frozen_signature.detach().cpu(),
        "boundary": entropy.detach().cpu().reshape(1),
    }


def _tensor_views(views: Mapping[str, torch.Tensor] | None):
    return {
        str(key): torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1).clone()
        for key, value in (views or {}).items()
    }


def deterministic_alternative_trajectory(
    features: torch.Tensor, primary: torch.Tensor, scores: torch.Tensor | None,
    maximum: int, seed: int, slide_id: str,
) -> torch.Tensor:
    """Build a second nested trajectory without another backward pass."""
    count = int(features.shape[0])
    maximum = min(max(int(maximum), 1), count)
    local_seed = _stable_seed(seed, slide_id, "trajectory-b")
    generator = torch.Generator().manual_seed(local_seed)
    if scores is None:
        ordered = torch.randperm(count, generator=generator)
    else:
        flat = scores.detach().float().reshape(-1).cpu()
        if flat.numel() != count:
            raise ValueError("attention scores and patches must align")
        high_count = maximum // 3
        low_count = maximum // 3
        high = torch.argsort(flat, descending=True, stable=True)[:high_count]
        low = torch.argsort(flat, descending=False, stable=True)[:low_count]
        used = torch.zeros(count, dtype=torch.bool)
        used[high] = True
        used[low] = True
        remaining = torch.nonzero(~used, as_tuple=False).flatten()
        random_count = maximum - high.numel() - low.numel()
        random_part = remaining[torch.randperm(remaining.numel(), generator=generator)[:random_count]]
        # Interleaving high/low/random gives useful prefixes at every depth.
        groups = [high, low, random_part]
        values = []
        for offset in range(max(len(group) for group in groups) if groups else 0):
            for group in groups:
                if offset < len(group):
                    values.append(int(group[offset]))
        ordered = torch.as_tensor(values, dtype=torch.long)
    merged = []
    seen = set()
    for value in torch.cat([ordered, primary.detach().cpu().long(), torch.arange(count)]).tolist():
        if value not in seen:
            seen.add(value)
            merged.append(value)
        if len(merged) == maximum:
            break
    return torch.as_tensor(merged, dtype=torch.long)


@dataclass(frozen=True)
class ReplayOption:
    slide_id: str
    label: int
    source_task: int
    trajectory: str
    patch_indices: torch.Tensor
    signature: torch.Tensor
    dataset_index: int | None = None
    is_existing: bool = False
    views: Mapping[str, torch.Tensor] | None = None

    def __post_init__(self):
        object.__setattr__(self, "patch_indices", torch.as_tensor(
            self.patch_indices, dtype=torch.long,
        ).detach().cpu().reshape(-1).clone())
        object.__setattr__(self, "signature", torch.as_tensor(
            self.signature, dtype=torch.float32,
        ).detach().cpu().reshape(-1).clone())
        materialized = _tensor_views(self.views)
        materialized.setdefault("gradient", self.signature)
        object.__setattr__(self, "views", materialized)

    @property
    def patch_count(self) -> int:
        return int(self.patch_indices.numel())


@dataclass(frozen=True)
class ReplaySlideCandidate:
    slide_id: str
    label: int
    source_task: int
    original_patch_count: int
    reference_signature: torch.Tensor
    options: tuple[ReplayOption, ...]
    dataset_index: int | None = None
    is_existing: bool = False
    reference_views: Mapping[str, torch.Tensor] | None = None

    def __post_init__(self):
        object.__setattr__(self, "reference_signature", torch.as_tensor(
            self.reference_signature, dtype=torch.float32,
        ).detach().cpu().reshape(-1).clone())
        materialized = _tensor_views(self.reference_views)
        materialized.setdefault("gradient", self.reference_signature)
        object.__setattr__(self, "reference_views", materialized)


@dataclass(frozen=True)
class SelectedReplayOption:
    option: ReplayOption
    reference_signature: torch.Tensor
    reference_views: Mapping[str, torch.Tensor]
    prefix_options: tuple[ReplayOption, ...]
    original_patch_count: int
    assignment_count: int = 0
    assignment_mass: float = 0.0
    marginal_coverage: float = 0.0

    @property
    def slide_id(self):
        return self.option.slide_id

    @property
    def label(self):
        return self.option.label

    @property
    def source_task(self):
        return self.option.source_task

    @property
    def dataset_index(self):
        return self.option.dataset_index

    @property
    def patch_budget(self):
        return self.option.patch_count

    @property
    def selected_indices(self):
        return self.option.patch_indices.clone()


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first.float(), dim=-1, eps=1e-12)
    second = F.normalize(second.float(), dim=-1, eps=1e-12)
    return (first @ second.T).clamp_min(0)


def _view_similarity(references: torch.Tensor, option: torch.Tensor, mode: str) -> torch.Tensor:
    references = references.float()
    option = option.float().reshape(1, -1)
    if mode == "gradient":
        return _cosine(references, option).squeeze(1)
    if mode in {"embedding", "frozen_feature"}:
        cosine = F.normalize(references, dim=1, eps=1e-12) @ F.normalize(
            option, dim=1, eps=1e-12
        ).T
        return ((cosine.squeeze(1).clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
    if mode == "probability":
        first = references.clamp_min(0)
        first = first / first.sum(dim=1, keepdim=True).clamp_min(1e-12)
        second = option.clamp_min(0)
        second = second / second.sum(dim=1, keepdim=True).clamp_min(1e-12)
        midpoint = 0.5 * (first + second)
        first_kl = torch.where(
            first > 0, first * (first.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
            torch.zeros_like(first),
        ).sum(dim=1)
        second_kl = torch.where(
            second > 0, second * (second.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()),
            torch.zeros_like(second),
        ).sum(dim=1)
        return (1 - 0.5 * (first_kl + second_kl) / math.log(2)).clamp(0, 1)
    raise ValueError(f"unsupported unified facility view: {mode}")


class UnifiedGradientCoverageBuffer(GlobalPatchBudgetBuffer):
    """Greedy multiple-choice facility-location allocator under a patch cap."""

    def __init__(
        self, patch_budget: int, seed: int, fold: int, levels=(64, 128, 256, 512),
        tolerance: float = 0.0, n_classes: int = 1, projection_dim: int = 64,
        device: str = "cpu", assignment_weighted_replay: bool = False,
        objective: str = "class_gradient_coverage",
        worst_class_lexicographic: bool = False,
        trajectory_mode: str = "ab",
        assignment_view: str = "auto",
        class_conditioned: bool = True,
        fixed_depth64: bool = False,
    ):
        super().__init__(
            ratio=1.0, seed=seed, fold=fold, floor=min(levels), tolerance=tolerance,
            device=device, allow_capacity_limited=False,
        )
        self.patch_budget = int(patch_budget)
        self.levels = tuple(sorted({int(value) for value in levels}))
        self.n_classes_total = int(n_classes)
        self.projection_dim = int(projection_dim)
        self.assignment_weighted_replay = bool(assignment_weighted_replay)
        self.objective = str(objective)
        self.worst_class_lexicographic = bool(worst_class_lexicographic)
        self.trajectory_mode = str(trajectory_mode).lower()
        self.assignment_view = str(assignment_view)
        self.class_conditioned = bool(class_conditioned)
        self.fixed_depth64 = bool(fixed_depth64)
        if self.patch_budget < 1 or not self.levels or self.levels[0] < 1:
            raise ValueError("unified patch budget and levels must be positive")
        if self.n_classes_total < 1 or self.projection_dim < 1:
            raise ValueError("unified signature dimensions must be positive")
        if self.trajectory_mode not in {"a", "b", "ab"}:
            raise ValueError("trajectory_mode must be one of: a, b, ab")
        if self.assignment_view not in {"auto", "gradient", "frozen_feature"}:
            raise ValueError("assignment_view must be auto, gradient, or frozen_feature")
        self._pending_reference_views = None

    @property
    def signature_size(self):
        return self.n_classes_total * self.projection_dim

    def existing_slide_candidates(self):
        result = []
        for sample in self._all_samples():
            slide_id = self._sample_slide_id(sample)
            label = int(torch.as_tensor(sample["label"]).reshape(-1)[0])
            source_task = int(torch.as_tensor(sample.get("source_task", -1)).item())
            reference = torch.as_tensor(sample["reference_gradient_signature"]).float().cpu().reshape(-1)
            signatures = sample.get("unified_option_signatures", {})
            option_views = sample.get("unified_option_views", {})
            indices_by_depth = sample.get("unified_option_indices", {})
            trajectory = str(sample.get("unified_trajectory", "a"))
            current_indices = torch.as_tensor(sample["selected_indices"]).cpu().long().reshape(-1)
            options = []
            for key, value in signatures.items():
                depth = int(key)
                if depth > current_indices.numel():
                    continue
                indices = torch.as_tensor(indices_by_depth.get(str(depth), current_indices[:depth])).long()
                if indices.numel() != depth or not torch.equal(indices, current_indices[:depth]):
                    raise AssertionError(f"old option is not a retained prefix: {slide_id}/{depth}")
                signature = torch.as_tensor(value).float().cpu().reshape(-1)
                options.append(ReplayOption(
                    slide_id, label, source_task, trajectory, indices,
                    signature, is_existing=True,
                    views=option_views.get(str(depth), {"gradient": signature}),
                ))
            if not options:
                signature = torch.as_tensor(sample["unified_selected_signature"]).float().cpu().reshape(-1)
                options = [ReplayOption(
                    slide_id, label, source_task, trajectory,
                    current_indices, signature, is_existing=True,
                    views=sample.get("unified_selected_views", {"gradient": signature}),
                )]
            result.append(ReplaySlideCandidate(
                slide_id, label, source_task,
                int(torch.as_tensor(sample["original_patch_count"]).item()),
                reference, tuple(sorted(options, key=lambda item: item.patch_count)),
                is_existing=True,
                reference_views=sample.get("unified_reference_views", {"gradient": reference}),
            ))
        return result

    @staticmethod
    def _validate_candidates(candidates, signature_size):
        slide_ids = [item.slide_id for item in candidates]
        duplicates = sorted(key for key, value in Counter(slide_ids).items() if value > 1)
        if duplicates:
            raise ValueError(f"duplicate slide IDs in unified admission pool: {duplicates[:5]}")
        for item in candidates:
            if item.reference_signature.numel() != signature_size:
                raise ValueError(f"reference signature dimension mismatch for {item.slide_id}")
            if not item.options:
                raise ValueError(f"slide has no non-empty replay option: {item.slide_id}")
            trajectories = defaultdict(list)
            for option in item.options:
                if option.slide_id != item.slide_id or option.label != item.label:
                    raise ValueError("option ownership mismatch")
                if option.signature.numel() != signature_size:
                    raise ValueError(f"option signature dimension mismatch for {item.slide_id}")
                if option.patch_count < 1 or option.patch_count > item.original_patch_count:
                    raise ValueError(f"invalid option depth for {item.slide_id}")
                trajectories[option.trajectory].append(option)
            for options in trajectories.values():
                options.sort(key=lambda value: value.patch_count)
                previous = None
                for option in options:
                    if previous is not None and not torch.equal(
                        option.patch_indices[:previous.numel()], previous
                    ):
                        raise ValueError(f"trajectory is not nested for {item.slide_id}")
                    previous = option.patch_indices

    @staticmethod
    def _objective(references, selected, view_name="gradient"):
        labels = sorted(references)
        if not labels:
            return 0.0, {}, {}
        per_class, errors = {}, {}
        for label in labels:
            refs = references[label]
            options = selected.get(label, [])
            if not options:
                coverage = refs.new_zeros(refs.shape[0])
            else:
                matrix = torch.stack([
                    _view_similarity(refs, item[1], view_name) for item in options
                ], dim=1)
                coverage = matrix.max(dim=1).values
            per_class[label] = float(coverage.mean())
            errors[label] = float((1 - coverage).mean())
        return sum(per_class.values()) / len(labels), per_class, errors

    def _greedy_facility(self, candidates, target, view_name="gradient"):
        if not self.class_conditioned:
            # A true global facility problem: references and prototypes may
            # cross class boundaries. Labels remain attached for reporting.
            remapped = []
            for item in candidates:
                remapped_options = tuple(ReplayOption(
                    option.slide_id, 0, option.source_task, option.trajectory,
                    option.patch_indices, option.signature, option.dataset_index,
                    option.is_existing, option.views,
                ) for option in item.options)
                remapped.append(ReplaySlideCandidate(
                    item.slide_id, 0, item.source_task, item.original_patch_count,
                    item.reference_signature, remapped_options, item.dataset_index,
                    item.is_existing, item.reference_views,
                ))
            chosen, used, value, per_class, marginal = self._greedy_facility(
                remapped, target, view_name,
            ) if self.class_conditioned else self._greedy_facility_global(
                remapped, target, view_name,
            )
            originals = {
                option.slide_id: option
                for item in candidates for option in item.options
            }
            restored = {}
            for slide_id, option in chosen.items():
                restored[slide_id] = next(
                    candidate for candidate in candidates
                    for candidate in candidate.options
                    if candidate.slide_id == slide_id
                    and candidate.trajectory == option.trajectory
                    and candidate.patch_count == option.patch_count
                )
            return restored, used, value, per_class, marginal
        by_id = {item.slide_id: item for item in candidates}
        combined = view_name == "frozen_gradient_geomean"
        if combined:
            references = {
                label: (
                    torch.stack([item.reference_views["frozen_feature"]
                                 for item in candidates if item.label == label]),
                    torch.stack([item.reference_views["gradient"]
                                 for item in candidates if item.label == label]),
                ) for label in sorted({item.label for item in candidates})
            }
        else:
            references = {
                label: torch.stack([
                    item.reference_views[view_name] for item in candidates if item.label == label
                ])
                for label in sorted({item.label for item in candidates})
            }
        trajectories = {}
        for item in candidates:
            grouped = defaultdict(list)
            for option in item.options:
                if "exact_closure_only" in option.views:
                    continue
                grouped[option.trajectory].append(option)
            trajectories[item.slide_id] = {
                key: sorted(value, key=lambda option: option.patch_count)
                for key, value in grouped.items()
            }

        signature_cache = {}
        similarity_cache = {}
        for item in candidates:
            for option in item.options:
                key = (option.slide_id, option.trajectory, option.patch_count)
                signature = (
                    (option.views["frozen_feature"], option.views["gradient"])
                    if combined else option.views[view_name]
                )
                signature_cache[key] = signature
                if combined:
                    frozen_similarity = _view_similarity(
                        references[option.label][0], signature[0], "frozen_feature",
                    )
                    gradient_similarity = _view_similarity(
                        references[option.label][1], signature[1], "gradient",
                    )
                    similarity_cache[key] = torch.sqrt(
                        frozen_similarity.clamp_min(1e-12)
                        * gradient_similarity.clamp_min(1e-12)
                    )
                else:
                    similarity_cache[key] = _view_similarity(
                        references[option.label], signature, view_name,
                    )

        chosen = {}
        selected_similarities = defaultdict(dict)
        current_best = {
            label: torch.zeros((refs[0] if combined else refs).shape[0])
            for label, refs in references.items()
        }
        current_objective = 0.0
        used = 0
        chosen_marginal = {}
        while used < target:
            # Top-2 caches make a same-slide upgrade an exact replacement in
            # O(number of reference WSI), instead of recomputing the complete
            # facility-location objective for every possible move.
            top_cache = {}
            for label, refs in references.items():
                base_refs = refs[0] if combined else refs
                items = sorted(selected_similarities[label].items())
                if not items:
                    top_cache[label] = (
                        None, base_refs.new_zeros(base_refs.shape[0]),
                        base_refs.new_zeros(base_refs.shape[0]),
                    )
                    continue
                matrix = torch.stack([value for _, value in items], dim=1)
                if matrix.shape[1] == 1:
                    top_cache[label] = (
                        [items[0][0]] * base_refs.shape[0], matrix[:, 0],
                        base_refs.new_zeros(base_refs.shape[0])
                    )
                else:
                    values, indices = torch.topk(matrix, 2, dim=1)
                    owners = [items[int(index)][0] for index in indices[:, 0].tolist()]
                    top_cache[label] = (owners, values[:, 0], values[:, 1])
            moves = []
            for slide_id, item in by_id.items():
                if slide_id not in chosen:
                    for trajectory, options in trajectories[slide_id].items():
                        option = options[0]
                        moves.append((slide_id, None, option, option.patch_count))
                else:
                    current = chosen[slide_id]
                    options = trajectories[slide_id][current.trajectory]
                    larger = [value for value in options if value.patch_count > current.patch_count]
                    if larger:
                        option = larger[0]
                        moves.append((slide_id, current, option, option.patch_count - current.patch_count))
            best = None
            for slide_id, previous, option, incremental_cost in moves:
                if incremental_cost < 1 or used + incremental_cost > target:
                    continue
                label = option.label
                key = (option.slide_id, option.trajectory, option.patch_count)
                similarity = similarity_cache[key]
                if previous is None:
                    updated = torch.maximum(current_best[label], similarity)
                else:
                    owners, top, second = top_cache[label]
                    without = torch.stack([
                        second[index] if owners[index] == slide_id else top[index]
                        for index in range(top.numel())
                    ])
                    updated = torch.maximum(without, similarity)
                class_delta = float(updated.mean() - current_best[label].mean())
                marginal = class_delta / max(len(references), 1)
                objective = current_objective + marginal
                score = marginal / incremental_cost
                if self.worst_class_lexicographic:
                    trial_per_class = {
                        group: float(current_best[group].mean()) for group in references
                    }
                    trial_per_class[label] = float(updated.mean())
                    key = (
                        -min(trial_per_class.values()),
                        -sum(trial_per_class.values()) / len(trial_per_class),
                        incremental_cost, slide_id, option.trajectory, option.patch_count,
                    )
                else:
                    key = (-score, -marginal, incremental_cost, slide_id, option.trajectory, option.patch_count)
                if best is None or key < best[0]:
                    best = (key, slide_id, previous, option, incremental_cost, objective, marginal)
            if best is None:
                break
            _, slide_id, previous, option, cost, objective, marginal = best
            option_key = (option.slide_id, option.trajectory, option.patch_count)
            selected_similarities[option.label][slide_id] = similarity_cache[option_key]
            if previous is None:
                current_best[option.label] = torch.maximum(
                    current_best[option.label], similarity_cache[option_key]
                )
            else:
                owners, top, second = top_cache[option.label]
                without = torch.stack([
                    second[index] if owners[index] == slide_id else top[index]
                    for index in range(top.numel())
                ])
                current_best[option.label] = torch.maximum(
                    without, similarity_cache[option_key]
                )
            chosen[slide_id] = option
            chosen_marginal[slide_id] = chosen_marginal.get(slide_id, 0.0) + marginal
            used += cost
            current_objective = objective
        selected_by_class = defaultdict(list)
        for slide_id, option in chosen.items():
            key = (option.slide_id, option.trajectory, option.patch_count)
            selected_by_class[option.label].append((slide_id, signature_cache[key]))
        if combined:
            per_class = {
                label: float(values.mean()) for label, values in current_best.items()
            }
            objective = sum(per_class.values()) / max(len(per_class), 1)
        else:
            objective, per_class, _ = self._objective(
                references, selected_by_class, view_name=view_name,
            )
        return chosen, used, objective, per_class, chosen_marginal

    def _greedy_facility_global(self, candidates, target, view_name):
        # Reuse the class-conditioned implementation after assigning every
        # reference and option to a single synthetic class.
        original = self.class_conditioned
        self.class_conditioned = True
        try:
            return self._greedy_facility(candidates, target, view_name)
        finally:
            self.class_conditioned = original

    def _greedy_additive(self, candidates, target, objective_name):
        by_id = {item.slide_id: item for item in candidates}
        trajectories = {}
        values = {}
        classes = sorted({item.label for item in candidates})
        for item in candidates:
            grouped = defaultdict(list)
            for option in item.options:
                grouped[option.trajectory].append(option)
                key = (option.slide_id, option.trajectory, option.patch_count)
                if objective_name == "random_options":
                    value = deterministic_random_option_score(
                        self.seed, self.fold, len(self.plan_history), option.slide_id,
                        option.trajectory, option.patch_count,
                    )
                elif objective_name == "decision_boundary":
                    value = float(option.views["boundary"].reshape(-1)[0])
                elif objective_name == "v4_scalar":
                    value = float(option.views["v4_scalar"].reshape(-1)[0])
                else:
                    raise ValueError(objective_name)
                values[key] = value
            trajectories[item.slide_id] = {
                key: sorted(group, key=lambda option: option.patch_count)
                for key, group in grouped.items()
            }
        chosen, marginal_by_slide = {}, {}
        used = 0
        per_class_sum = {label: 0.0 for label in classes}
        per_class_slides = Counter()
        total_min_bags = target // min(self.levels)
        base, remainder = divmod(total_min_bags, max(len(classes), 1))
        slide_quotas = {
            label: base + int(offset < remainder) for offset, label in enumerate(classes)
        }
        while used < target:
            best = None
            for slide_id, item in by_id.items():
                if slide_id not in chosen:
                    if (objective_name == "decision_boundary"
                            and per_class_slides[item.label] >= slide_quotas[item.label]):
                        continue
                    moves = [(None, options[0]) for options in trajectories[slide_id].values()]
                else:
                    current = chosen[slide_id]
                    larger = [option for option in trajectories[slide_id][current.trajectory]
                              if option.patch_count > current.patch_count]
                    moves = [(current, larger[0])] if larger else []
                for previous, option in moves:
                    cost = option.patch_count - (previous.patch_count if previous else 0)
                    if cost < 1 or used + cost > target:
                        continue
                    key = (option.slide_id, option.trajectory, option.patch_count)
                    old = 0.0 if previous is None else values[
                        (previous.slide_id, previous.trajectory, previous.patch_count)
                    ]
                    marginal = values[key] - old
                    score = marginal / cost
                    tie = (-score, -marginal, cost, slide_id, option.trajectory, option.patch_count)
                    if best is None or tie < best[0]:
                        best = (tie, slide_id, previous, option, cost, marginal)
            if best is None:
                break
            _, slide_id, previous, option, cost, marginal = best
            if previous is None:
                per_class_slides[option.label] += 1
            else:
                per_class_sum[option.label] -= values[
                    (previous.slide_id, previous.trajectory, previous.patch_count)
                ]
            per_class_sum[option.label] += values[
                (option.slide_id, option.trajectory, option.patch_count)
            ]
            chosen[slide_id] = option
            marginal_by_slide[slide_id] = marginal_by_slide.get(slide_id, 0.0) + marginal
            used += cost
        per_class = {
            label: per_class_sum[label] / max(slide_quotas[label], 1)
            for label in classes
        }
        return chosen, used, sum(per_class.values()) / max(len(per_class), 1), per_class, marginal_by_slide

    def _greedy_information(self, candidates, target, objective_name):
        references = defaultdict(list)
        trajectories = {}
        for item in candidates:
            references[item.label].append(InformationPoint(
                item.slide_id, item.label, item.reference_views["gradient_information"],
            ))
            grouped = defaultdict(list)
            for option in item.options:
                grouped[option.trajectory].append(option)
            trajectories[item.slide_id] = {
                key: sorted(group, key=lambda option: option.patch_count)
                for key, group in grouped.items()
            }
        selected = {}
        empty = {label: [] for label in references}
        evaluation = evaluate_information_objective(objective_name, references, empty)
        used = 0
        marginal_by_slide = {}
        while used < target:
            best = None
            for item in candidates:
                current = selected.get(item.slide_id)
                if current is None:
                    moves = [(None, options[0]) for options in trajectories[item.slide_id].values()]
                else:
                    larger = [option for option in trajectories[item.slide_id][current.trajectory]
                              if option.patch_count > current.patch_count]
                    moves = [(current, larger[0])] if larger else []
                for previous, option in moves:
                    cost = option.patch_count - (previous.patch_count if previous else 0)
                    if cost < 1 or used + cost > target:
                        continue
                    trial = dict(selected)
                    trial[item.slide_id] = option
                    selected_points = {label: [] for label in references}
                    for chosen in trial.values():
                        selected_points[chosen.label].append(InformationPoint(
                            chosen.slide_id, chosen.label,
                            chosen.views["gradient_information"], cost=chosen.patch_count,
                        ))
                    updated = evaluate_information_objective(
                        objective_name, references, selected_points,
                    )
                    marginal = updated.value - evaluation.value
                    score = marginal / cost
                    key = (-score, -marginal, cost, option.slide_id,
                           option.trajectory, option.patch_count)
                    if best is None or key < best[0]:
                        best = (key, option, cost, marginal, updated)
            if best is None:
                break
            _, option, cost, marginal, evaluation = best
            selected[option.slide_id] = option
            marginal_by_slide[option.slide_id] = (
                marginal_by_slide.get(option.slide_id, 0.0) + marginal
            )
            used += cost
        return (
            selected, used, float(evaluation.value),
            {int(key): float(value) for key, value in evaluation.per_class.items()},
            marginal_by_slide,
        )

    def _greedy_mean_gradient(self, candidates, target):
        classes = sorted({item.label for item in candidates})
        reference_means = {
            label: torch.stack([
                item.reference_views["gradient_information"]
                for item in candidates if item.label == label
            ]).mean(dim=0)
            for label in classes
        }
        trajectories = {}
        for item in candidates:
            grouped = defaultdict(list)
            for option in item.options:
                grouped[option.trajectory].append(option)
            trajectories[item.slide_id] = {
                key: sorted(group, key=lambda option: option.patch_count)
                for key, group in grouped.items()
            }
        dimension = next(iter(reference_means.values())).numel()
        sums = {label: torch.zeros(dimension) for label in classes}
        counts = Counter()
        per_class = {
            label: -float(reference_means[label].square().sum()) for label in classes
        }
        selected, marginal_by_slide, used = {}, {}, 0
        current = sum(per_class.values()) / len(classes)
        while used < target:
            best = None
            for item in candidates:
                previous = selected.get(item.slide_id)
                if previous is None:
                    moves = [(None, options[0]) for options in trajectories[item.slide_id].values()]
                else:
                    larger = [option for option in trajectories[item.slide_id][previous.trajectory]
                              if option.patch_count > previous.patch_count]
                    moves = [(previous, larger[0])] if larger else []
                for old, option in moves:
                    cost = option.patch_count - (old.patch_count if old else 0)
                    if cost < 1 or used + cost > target:
                        continue
                    label = item.label
                    new_sum = sums[label] + option.views["gradient_information"]
                    new_count = counts[label] + 1
                    if old is not None:
                        new_sum = new_sum - old.views["gradient_information"]
                        new_count -= 1
                    mean = new_sum / max(new_count, 1)
                    updated_class = -float((reference_means[label] - mean).square().sum())
                    marginal = (updated_class - per_class[label]) / len(classes)
                    score = marginal / cost
                    key = (-score, -marginal, cost, option.slide_id,
                           option.trajectory, option.patch_count)
                    if best is None or key < best[0]:
                        best = (key, option, old, cost, marginal, new_sum, new_count, updated_class)
            if best is None:
                break
            _, option, old, cost, marginal, new_sum, new_count, updated_class = best
            sums[option.label] = new_sum
            counts[option.label] = new_count
            per_class[option.label] = updated_class
            selected[option.slide_id] = option
            marginal_by_slide[option.slide_id] = (
                marginal_by_slide.get(option.slide_id, 0.0) + marginal
            )
            current += marginal
            used += cost
        return selected, used, current, per_class, marginal_by_slide

    def _greedy(self, candidates, target):
        facility_views = {
            "class_gradient_coverage": "gradient",
            "prediction_fidelity": "probability",
            "embedding_fidelity": "embedding",
            "frozen_feature_coverage": "frozen_feature",
            "frozen_gradient_geomean_coverage": "frozen_gradient_geomean",
        }
        if self.objective in facility_views:
            return self._greedy_facility(candidates, target, facility_views[self.objective])
        if self.objective in {"random_options", "decision_boundary", "v4_scalar"}:
            return self._greedy_additive(candidates, target, self.objective)
        if self.objective == "class_mean_gradient_matching":
            return self._greedy_mean_gradient(candidates, target)
        if self.objective == "fisher_logdet":
            return self._greedy_information(candidates, target, self.objective)
        raise ValueError(f"unsupported unified objective: {self.objective}")

    @staticmethod
    def _close_exact_budget(chosen, candidates, used, target):
        """Deterministically expand new nested prefixes to close a sub-level gap.

        The closure carries the current option view during planning and is
        marked explicitly.  The integration layer recomputes its exact
        multi-view metadata from the materialized prefix before storage.
        Old WSI are never expanded.
        """
        gap = int(target) - int(used)
        if gap <= 0:
            return chosen, used, {}
        by_id = {item.slide_id: item for item in candidates}
        closure = {}
        for slide_id in sorted(chosen):
            if gap <= 0:
                break
            current = chosen[slide_id]
            item = by_id[slide_id]
            if item.is_existing:
                continue
            same_trajectory = sorted(
                [option for option in item.options if option.trajectory == current.trajectory],
                key=lambda option: option.patch_count,
            )
            maximum = same_trajectory[-1]
            capacity = maximum.patch_count - current.patch_count
            if capacity <= 0:
                continue
            addition = min(gap, capacity)
            depth = current.patch_count + addition
            views = {key: value.clone() for key, value in current.views.items()}
            views["budget_closure_approximation"] = torch.ones(1)
            chosen[slide_id] = ReplayOption(
                slide_id=current.slide_id, label=current.label,
                source_task=current.source_task, trajectory=current.trajectory,
                patch_indices=maximum.patch_indices[:depth],
                signature=current.signature, dataset_index=current.dataset_index,
                is_existing=False, views=views,
            )
            closure[slide_id] = addition
            used += addition
            gap -= addition
        return chosen, used, closure

    @staticmethod
    def _assignments(references, selected_by_class, view_name="gradient"):
        counts, masses = Counter(), defaultdict(float)
        total_classes = max(len(references), 1)
        for label, refs in references.items():
            options = selected_by_class.get(label, [])
            if not options:
                continue
            if view_name == "frozen_gradient_geomean":
                frozen_refs, gradient_refs = refs
                columns = []
                for _, views in options:
                    frozen = _view_similarity(
                        frozen_refs, views["frozen_feature"], "frozen_feature",
                    )
                    gradient = _view_similarity(
                        gradient_refs, views["gradient"], "gradient",
                    )
                    columns.append(torch.sqrt(
                        frozen.clamp_min(1e-12) * gradient.clamp_min(1e-12)
                    ))
                matrix = torch.stack(columns, dim=1)
                reference_count = frozen_refs.shape[0]
            else:
                matrix = torch.stack([
                    _view_similarity(refs, value[1], view_name) for value in options
                ], dim=1)
                reference_count = refs.shape[0]
            assignments = matrix.argmax(dim=1)
            per_reference_mass = 1.0 / (total_classes * max(reference_count, 1))
            for index in assignments.tolist():
                slide_id = options[index][0]
                counts[slide_id] += 1
                masses[slide_id] += per_reference_mass
        return counts, masses

    @staticmethod
    def _distortion_summary(references, selected_by_class, view_name):
        distances = []
        for label, refs in references.items():
            options = selected_by_class.get(label, [])
            if not options:
                continue
            if view_name == "frozen_gradient_geomean":
                frozen_refs, gradient_refs = refs
                columns = []
                for _, views in options:
                    frozen = _view_similarity(
                        frozen_refs, views["frozen_feature"], "frozen_feature",
                    )
                    gradient = _view_similarity(
                        gradient_refs, views["gradient"], "gradient",
                    )
                    columns.append(torch.sqrt(
                        frozen.clamp_min(1e-12) * gradient.clamp_min(1e-12)
                    ))
                matrix = torch.stack(columns, dim=1)
            else:
                matrix = torch.stack([
                    _view_similarity(refs, value[1], view_name) for value in options
                ], dim=1)
            distances.append(1 - matrix.max(dim=1).values)
        if not distances:
            return float("nan"), float("nan")
        values = torch.cat(distances).float()
        return float(values.mean()), float(torch.quantile(values, 0.95))

    @staticmethod
    def _effective_rank(vectors):
        if len(vectors) < 1:
            return float("nan")
        matrix = torch.stack([torch.as_tensor(value).float() for value in vectors])
        singular = torch.linalg.svdvals(matrix)
        probability = singular / singular.sum().clamp_min(1e-12)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
        return float(entropy.exp())

    def _shrink_existing(self, chosen):
        retained, evicted, shrunk = set(chosen), [], []
        new_buffers = {}
        for label, samples in self.buffers.items():
            kept = []
            for sample in samples:
                slide_id = self._sample_slide_id(sample)
                if slide_id not in retained:
                    evicted.append(slide_id)
                    continue
                option = chosen[slide_id]
                current_indices = torch.as_tensor(sample["selected_indices"]).long().reshape(-1)
                target_indices = option.patch_indices.long()
                if option.patch_count > current_indices.numel():
                    raise AssertionError("old WSI expansion is forbidden")
                if not torch.equal(current_indices[:option.patch_count].cpu(), target_indices.cpu()):
                    raise AssertionError("old WSI may only retain a stored nested prefix")
                if option.patch_count < current_indices.numel():
                    sample = dict(sample)
                    sample["features"] = sample["features"][:option.patch_count]
                    sample["selected_indices"] = current_indices[:option.patch_count]
                    sample["unified_selected_signature"] = option.signature.clone()
                    for key in ("region_membership", "region_teacher_probs", "region_reliability",
                                "region_stored_classes"):
                        sample.pop(key, None)
                    signatures = sample.get("unified_option_signatures", {})
                    indices = sample.get("unified_option_indices", {})
                    sample["unified_option_signatures"] = {
                        str(depth): value for depth, value in signatures.items()
                        if int(depth) <= option.patch_count
                    }
                    sample["unified_option_indices"] = {
                        str(depth): value for depth, value in indices.items()
                        if int(depth) <= option.patch_count
                    }
                    views = sample.get("unified_option_views", {})
                    sample["unified_option_views"] = {
                        str(depth): value for depth, value in views.items()
                        if int(depth) <= option.patch_count
                    }
                    sample["unified_selected_views"] = {
                        key: value.clone() for key, value in option.views.items()
                    }
                    shrunk.append(slide_id)
                kept.append(sample)
            if kept:
                new_buffers[label] = kept
        self.buffers = new_buffers
        self._refresh()
        return sorted(evicted), sorted(shrunk)

    def plan(self, new_candidates, target_budget=None, task=0):
        target = self.patch_budget if target_budget is None else int(target_budget)
        if target != self.patch_budget:
            raise ValueError("unified allocator uses its configured strict global budget")
        existing = self.existing_slide_candidates()
        candidates = existing + list(new_candidates)
        if not candidates:
            raise ValueError("empty unified admission pool")
        self._validate_candidates(candidates, self.signature_size)
        chosen, used, selection_value, selection_per_class, marginal = self._greedy(
            candidates, target
        )
        chosen, used, closure = self._close_exact_budget(chosen, candidates, used, target)
        if used != target:
            raise RuntimeError(
                f"unified candidate space cannot close exact patch budget: target={target}, used={used}"
            )
        before_ids = {item.slide_id for item in existing}
        evicted, shrunk = self._shrink_existing(chosen)
        gradient_references = {
            label: torch.stack([
                item.reference_views["gradient"] for item in candidates if item.label == label
            ]) for label in sorted({item.label for item in candidates})
        }
        gradient_selected = defaultdict(list)
        for slide_id, option in chosen.items():
            gradient_selected[option.label].append((slide_id, option.views["gradient"]))
        assignment_view = {
            "frozen_feature_coverage": "frozen_feature",
            "class_gradient_coverage": "gradient",
            "frozen_gradient_geomean_coverage": "frozen_gradient_geomean",
        }.get(self.objective, "gradient")
        if self.assignment_view != "auto":
            assignment_view = self.assignment_view
        if assignment_view == "frozen_gradient_geomean":
            assignment_references = {
                label: (
                    torch.stack([item.reference_views["frozen_feature"]
                                 for item in candidates if item.label == label]),
                    torch.stack([item.reference_views["gradient"]
                                 for item in candidates if item.label == label]),
                ) for label in sorted({item.label for item in candidates})
            }
            assignment_selected = defaultdict(list)
            for slide_id, option in chosen.items():
                assignment_selected[option.label].append((slide_id, option.views))
        else:
            assignment_references = {
                label: torch.stack([
                    item.reference_views[assignment_view]
                    for item in candidates if item.label == label
                ]) for label in sorted({item.label for item in candidates})
            }
            assignment_selected = defaultdict(list)
            for slide_id, option in chosen.items():
                assignment_selected[option.label].append(
                    (slide_id, option.views[assignment_view])
                )
        counts, masses = self._assignments(
            assignment_references, assignment_selected, assignment_view,
        )
        self._pending_reference_views = {
            "gradient": gradient_references,
            "frozen_feature": {
                label: torch.stack([
                    item.reference_views.get(
                        "frozen_feature", item.reference_views["gradient"]
                    )
                    for item in candidates if item.label == label
                ]) for label in sorted({item.label for item in candidates})
            },
            "assignment": assignment_references,
        }
        objective, per_class, errors = self._objective(
            gradient_references, gradient_selected, view_name="gradient",
        )
        by_id = {item.slide_id: item for item in candidates}
        admissions = []
        for slide_id, option in chosen.items():
            if option.is_existing:
                continue
            item = by_id[slide_id]
            prefixes = tuple(
                value for value in item.options
                if value.trajectory == option.trajectory and value.patch_count <= option.patch_count
            )
            if "budget_closure_approximation" in option.views:
                prefixes = tuple(
                    value for value in prefixes if value.patch_count < option.patch_count
                ) + (option,)
            admissions.append(SelectedReplayOption(
                option, item.reference_signature, item.reference_views,
                prefixes, item.original_patch_count,
                counts[slide_id], masses[slide_id], marginal.get(slide_id, 0.0),
            ))
        admissions.sort(key=lambda item: item.slide_id)
        retained = before_ids & set(chosen)
        self.last_plan = {
            "fold": self.fold, "task": int(task), "mode": "unified_gradient_coverage",
            "surrogate": ("candidate_restricted_class_conditioned_facility_location"
                          if self.class_conditioned else
                          "candidate_restricted_global_facility_location"),
            "solver": "deterministic_greedy_not_exact",
            "target_budget": target, "target_original": target, "target_effective": target,
            "planned_budget": used, "budget_error": 0.0, "capacity_limited": False,
            "actual_budget_before_add": self.n_patches_total,
            "selected_slide_ids": sorted(chosen), "retained_slide_ids": sorted(retained),
            "evicted_slide_ids": evicted, "shrunk_slide_ids": shrunk,
            "admitted_slide_ids": [item.slide_id for item in admissions],
            "selected_slides": len(chosen), "retained_slides": len(retained),
            "evicted_slides": len(evicted), "shrunk_slides": len(shrunk),
            "admitted_slides": len(admissions),
            "allocation_histogram": dict(Counter(value.patch_count for value in chosen.values())),
            "budget_closure_patches": int(sum(closure.values())),
            "budget_closure_by_slide": closure,
            "selected_trajectories": {key: value.trajectory for key, value in chosen.items()},
            "selected_patch_counts": {key: value.patch_count for key, value in chosen.items()},
            "coverage": objective, "class_coverage": per_class,
            "selection_objective": self.objective,
            "selection_objective_value": selection_value,
            "selection_per_class": selection_per_class,
            "class_quantization_error": errors,
            "quantization_error": sum(errors.values()) / max(len(errors), 1),
            "assignment_count": dict(counts), "assignment_mass": dict(masses),
            "assignment_view": assignment_view,
            "assignment_weighted_replay": self.assignment_weighted_replay,
            "objective": self.objective,
            "worst_class_lexicographic": self.worst_class_lexicographic,
            "trajectory_mode": self.trajectory_mode,
            "class_conditioned": self.class_conditioned,
            "fixed_depth64": self.fixed_depth64,
        }
        self.plan_history.append(self.last_plan)
        return admissions, self.last_plan

    def add(self, sample):
        super().add(sample)

    def finalize_plan(self):
        assignment_view = self.last_plan.get("assignment_view", "gradient")
        if self._pending_reference_views is not None:
            selected_views = defaultdict(list)
            gradient_vectors = []
            for sample in self._all_samples():
                slide_id = self._sample_slide_id(sample)
                label = int(torch.as_tensor(sample["label"]).reshape(-1)[0])
                views = sample["unified_selected_views"]
                value = views if assignment_view == "frozen_gradient_geomean" else views[assignment_view]
                selected_views[label].append((slide_id, value))
                gradient_vectors.append(views["gradient"])
            if assignment_view == "frozen_gradient_geomean":
                assignment_references = {
                    label: (
                        self._pending_reference_views["frozen_feature"][label],
                        self._pending_reference_views["gradient"][label],
                    ) for label in self._pending_reference_views["gradient"]
                }
            else:
                assignment_references = self._pending_reference_views[assignment_view]
            assignments, masses = self._assignments(
                assignment_references, selected_views, assignment_view,
            )
            self.last_plan["assignment_count"] = dict(assignments)
            self.last_plan["assignment_mass"] = dict(masses)

            frozen_selected = defaultdict(list)
            gradient_selected = defaultdict(list)
            for sample in self._all_samples():
                slide_id = self._sample_slide_id(sample)
                label = int(torch.as_tensor(sample["label"]).reshape(-1)[0])
                views = sample["unified_selected_views"]
                frozen_selected[label].append((
                    slide_id, views.get("frozen_feature", views["gradient"]),
                ))
                gradient_selected[label].append((slide_id, views["gradient"]))
            morphology, morphology_p95 = self._distortion_summary(
                self._pending_reference_views["frozen_feature"], frozen_selected,
                "frozen_feature",
            )
            gradient, gradient_p95 = self._distortion_summary(
                self._pending_reference_views["gradient"], gradient_selected, "gradient",
            )
            mass_values = torch.tensor(list(masses.values()), dtype=torch.float64)
            mass_values = mass_values / mass_values.sum().clamp_min(1e-12)
            trajectory_counts = Counter(
                str(sample.get("unified_trajectory", "a")) for sample in self._all_samples()
            )
            total_trajectories = max(sum(trajectory_counts.values()), 1)
            self.last_plan.update({
                "morphology_distortion": morphology,
                "morphology_distortion_p95": morphology_p95,
                "gradient_distortion": gradient,
                "gradient_distortion_p95": gradient_p95,
                "gradient_effective_rank": self._effective_rank(gradient_vectors),
                "assignment_effective_sample_size": float(
                    1 / mass_values.square().sum().clamp_min(1e-12)
                ),
                "trajectory_counts": dict(trajectory_counts),
                "trajectory_ratios": {
                    key: value / total_trajectories
                    for key, value in trajectory_counts.items()
                },
            })
        assignments = self.last_plan.get("assignment_count", {})
        masses = self.last_plan.get("assignment_mass", {})
        for sample in self._all_samples():
            slide_id = self._sample_slide_id(sample)
            sample["unified_assignment_count"] = torch.tensor(int(assignments.get(slide_id, 0)))
            sample["unified_assignment_mass"] = torch.tensor(float(masses.get(slide_id, 0.0)))
        super().finalize_plan()
        if self.n_patches_total != self.patch_budget:
            raise AssertionError("unified final buffer must exactly match the global patch budget")

    def start_epoch(self):
        if not self.assignment_weighted_replay:
            return super().start_epoch()
        self.all_samples = list(self._all_samples())
        weights = torch.tensor([
            float(torch.as_tensor(sample.get("unified_assignment_mass", 0.0)).item())
            for sample in self.all_samples
        ], dtype=torch.float64)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise RuntimeError("assignment-weighted replay requires positive finite assignment masses")
        generator = torch.Generator().manual_seed(
            self.seed + 1000003 * self.fold + len(self.plan_history)
        )
        self.epoch_indices = torch.multinomial(
            weights / weights.sum(), len(self.all_samples), replacement=True,
            generator=generator,
        ).numpy()
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def state_dict(self):
        state = super().state_dict()
        state.update({
            "patch_budget": self.patch_budget, "levels": self.levels,
            "n_classes_total": self.n_classes_total,
            "projection_dim": self.projection_dim,
            "assignment_weighted_replay": self.assignment_weighted_replay,
            "objective": self.objective,
            "worst_class_lexicographic": self.worst_class_lexicographic,
            "trajectory_mode": self.trajectory_mode,
            "assignment_view": self.assignment_view,
            "class_conditioned": self.class_conditioned,
            "fixed_depth64": self.fixed_depth64,
        })
        return state

    @classmethod
    def from_state_dict(cls, state):
        result = cls(
            state["patch_budget"], state["seed"], state["fold"], state["levels"],
            state["tolerance"], state["n_classes_total"], state["projection_dim"],
            assignment_weighted_replay=state.get("assignment_weighted_replay", False),
            objective=state.get("objective", "class_gradient_coverage"),
            worst_class_lexicographic=state.get("worst_class_lexicographic", False),
            trajectory_mode=state.get("trajectory_mode", "ab"),
            assignment_view=state.get("assignment_view", "auto"),
            class_conditioned=state.get("class_conditioned", True),
            fixed_depth64=state.get("fixed_depth64", False),
        )
        result.buffers = state["buffers"]
        result.plan_history = state.get("plan_history", [])
        result.last_plan = state.get("last_plan", {})
        result._refresh()
        return result
