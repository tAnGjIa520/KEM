"""Pure objective functions for candidate-restricted unified replay.

This module intentionally knows nothing about the replay buffer or its greedy
solver.  An integration layer only needs to expose records with
``slide_id``, ``label``, ``patch_count``, and a ``views`` mapping.  Keeping
the objective independent makes it possible to compare what the buffer is
trying to preserve without silently changing its budget or candidate space.

All objectives return larger-is-better scores.  Fidelity objectives use a
class-conditioned facility-location reduction: every full/reference WSI is
represented by its most faithful selected pseudo-bag from the same class.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectiveResult:
    """A JSON-friendly objective value and its audit diagnostics."""

    value: float
    per_class: dict[int, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MarginalObjectiveResult:
    """The exact set-objective change caused by adding one option."""

    current: float
    updated: float
    marginal: float
    diagnostics: dict[str, Any]


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _view(record: Any, name: str) -> Any:
    views = _field(record, "views")
    if views is None:
        raise ValueError(f"objective record has no views mapping: {_record_key(record)}")
    if isinstance(views, Mapping):
        if name not in views:
            raise ValueError(f"objective record is missing view '{name}': {_record_key(record)}")
        return views[name]
    if not hasattr(views, name):
        raise ValueError(f"objective record is missing view '{name}': {_record_key(record)}")
    return getattr(views, name)


def _record_key(record: Any) -> str:
    slide_id = str(_field(record, "slide_id", ""))
    trajectory = str(_field(record, "trajectory", ""))
    patch_count = int(_field(record, "patch_count", 0) or 0)
    return f"{slide_id}|{trajectory}|{patch_count}"


def _labels(
    references_by_class: Mapping[int, Sequence[Any]],
    selected_by_class: Mapping[int, Sequence[Any]],
) -> list[int]:
    return sorted({int(value) for value in references_by_class} | {
        int(value) for value in selected_by_class
    })


def _validate_group(label: int, records: Sequence[Any]) -> None:
    for record in records:
        record_label = _field(record, "label")
        if record_label is not None and int(record_label) != int(label):
            raise ValueError(
                f"record {_record_key(record)} is grouped under class {label} "
                f"but carries label {record_label}"
            )


def _finite_float(value: torch.Tensor | float, name: str) -> float:
    result = float(torch.as_tensor(value).detach().cpu())
    if not math.isfinite(result):
        raise ValueError(f"{name} produced a non-finite value")
    return result


def _as_vector(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.detach().reshape(-1)
    if tensor.numel() < 1:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def align_probability_view(
    view: Any,
    total_classes: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a normalized global-class probability vector.

    ``view`` may be a dense probability tensor or a mapping containing
    ``values`` (or ``probabilities``), optional ``class_indices``, and
    optional ``n_classes``.  The sparse form lets an old target over classes
    ``[0, 2]`` remain comparable after the classifier expands to class 3.
    Missing/new class axes are padded with exact zeros before normalization.
    """

    class_indices = None
    declared_classes = None
    if isinstance(view, Mapping):
        values = view.get("values", view.get("probabilities"))
        if values is None:
            raise ValueError("probability view requires 'values' or 'probabilities'")
        class_indices = view.get("class_indices")
        declared_classes = view.get("n_classes")
    else:
        values = view
    probabilities = _as_vector(values, "probability view")
    if bool((probabilities < -1e-7).any()):
        raise ValueError("probability view cannot contain negative values")
    probabilities = probabilities.clamp_min(0)
    if class_indices is None:
        indices = torch.arange(probabilities.numel(), device=probabilities.device)
    else:
        indices = torch.as_tensor(class_indices, device=probabilities.device, dtype=torch.long).reshape(-1)
        if indices.numel() != probabilities.numel():
            raise ValueError("class_indices and probability values must have equal length")
        if indices.numel() and (int(indices.min()) < 0 or torch.unique(indices).numel() != indices.numel()):
            raise ValueError("class_indices must be unique and non-negative")
    inferred = int(indices.max()) + 1 if indices.numel() else 0
    sizes = [probabilities.numel(), inferred]
    if declared_classes is not None:
        sizes.append(int(declared_classes))
    if total_classes is not None:
        sizes.append(int(total_classes))
    width = max(sizes)
    if width < 1 or inferred > width:
        raise ValueError("invalid global class axis for probability view")
    target_device = probabilities.device if device is None else torch.device(device)
    result = torch.zeros(width, dtype=torch.float32, device=target_device)
    result.index_copy_(0, indices.to(target_device), probabilities.float().to(target_device))
    mass = result.sum()
    if not torch.isfinite(mass) or float(mass) <= 0:
        raise ValueError("probability view must have positive finite mass")
    return result / mass


def js_fidelity(first: Any, second: Any, total_classes: int | None = None) -> torch.Tensor:
    """Numerically stable Jensen--Shannon fidelity in the closed interval [0, 1]."""

    first_dense = align_probability_view(first, total_classes=total_classes)
    second_dense = align_probability_view(
        second, total_classes=max(first_dense.numel(), total_classes or 0),
        device=first_dense.device,
    )
    if first_dense.numel() < second_dense.numel():
        first_dense = F.pad(first_dense, (0, second_dense.numel() - first_dense.numel()))
    elif second_dense.numel() < first_dense.numel():
        second_dense = F.pad(second_dense, (0, first_dense.numel() - second_dense.numel()))
    midpoint = 0.5 * (first_dense + second_dense)
    tiny = torch.finfo(first_dense.dtype).tiny

    def kl(part: torch.Tensor) -> torch.Tensor:
        log_ratio = part.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log()
        return torch.where(part > 0, part * log_ratio, torch.zeros_like(part)).sum()

    divergence = 0.5 * (kl(first_dense) + kl(second_dense)) / math.log(2.0)
    return (1.0 - divergence).clamp(0.0, 1.0)


def normalized_cosine_fidelity(first: Any, second: Any) -> torch.Tensor:
    """Cosine fidelity in [0, 1] with explicit, meaningful zero-vector behavior."""

    first_vector = _as_vector(first, "first signature")
    second_vector = _as_vector(second, "second signature").to(first_vector.device)
    if first_vector.numel() != second_vector.numel():
        raise ValueError("fidelity signatures must have equal dimensions")
    first_norm = torch.linalg.vector_norm(first_vector.float())
    second_norm = torch.linalg.vector_norm(second_vector.float())
    first_zero = bool(first_norm <= 1e-12)
    second_zero = bool(second_norm <= 1e-12)
    if first_zero or second_zero:
        return first_vector.new_tensor(1.0 if first_zero and second_zero else 0.0)
    cosine = torch.dot(first_vector.float(), second_vector.float()) / (first_norm * second_norm)
    return ((cosine.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def frozen_feature_signature(patch_features: torch.Tensor) -> torch.Tensor:
    """Model-independent pooled signature made directly from frozen UNI patches.

    Concatenated mean and population standard deviation retain first- and
    second-order morphology information.  This is intentionally different
    from a learned MIL bag embedding and never calls the MIL model.
    """

    features = torch.as_tensor(patch_features)
    if not features.is_floating_point():
        features = features.float()
    features = features.detach()
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("frozen patch features must have shape [patches, dimensions]")
    if not torch.isfinite(features).all():
        raise ValueError("frozen patch features must be finite")
    as_float = features.float()
    pooled = torch.cat([as_float.mean(dim=0), as_float.std(dim=0, unbiased=False)], dim=0)
    return F.normalize(pooled, dim=0, eps=1e-12)


def normalized_entropy(probability_view: Any, total_classes: int | None = None) -> torch.Tensor:
    """Normalized predictive entropy used as a pure boundary-uncertainty score."""

    probability = align_probability_view(probability_view, total_classes=total_classes)
    active = probability > 0
    active_count = int(active.sum())
    if active_count <= 1:
        return probability.new_zeros(())
    entropy = -(probability[active] * probability[active].log()).sum()
    return (entropy / math.log(active_count)).clamp(0.0, 1.0)


def deterministic_random_option_score(
    seed: int,
    fold: int,
    task: int,
    slide_id: str,
    trajectory: str = "",
    patch_count: int = 0,
) -> float:
    """Stable random-option score independent of Python's salted ``hash``."""

    payload = (
        f"kem-random-options-v1|{int(seed)}|{int(fold)}|{int(task)}|"
        f"{slide_id}|{trajectory}|{int(patch_count)}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


class BaseObjective:
    name = "base"
    required_views: tuple[str, ...] = ()

    def evaluate(
        self,
        references_by_class: Mapping[int, Sequence[Any]],
        selected_by_class: Mapping[int, Sequence[Any]],
    ) -> ObjectiveResult:
        raise NotImplementedError

    def marginal(
        self,
        references_by_class: Mapping[int, Sequence[Any]],
        selected_by_class: Mapping[int, Sequence[Any]],
        candidate: Any,
    ) -> MarginalObjectiveResult:
        label = int(_field(candidate, "label"))
        key = _record_key(candidate)
        selected = {int(group): list(records) for group, records in selected_by_class.items()}
        existing_keys = {_record_key(record) for record in selected.get(label, [])}
        current = self.evaluate(references_by_class, selected)
        if key in existing_keys:
            return MarginalObjectiveResult(
                current.value, current.value, 0.0,
                {"objective": self.name, "candidate": key, "duplicate_option": True},
            )
        selected.setdefault(label, []).append(candidate)
        updated = self.evaluate(references_by_class, selected)
        marginal = updated.value - current.value
        if not math.isfinite(marginal):
            raise ValueError(f"{self.name} marginal is non-finite")
        return MarginalObjectiveResult(
            current.value, updated.value, marginal,
            {
                "objective": self.name,
                "candidate": key,
                "duplicate_option": False,
                "current_per_class": current.per_class,
                "updated_per_class": updated.per_class,
            },
        )


class _FacilityFidelityObjective(BaseObjective):
    view_name = ""

    def _fidelity(self, reference: Any, selected: Any, total_classes: int | None) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, references_by_class, selected_by_class) -> ObjectiveResult:
        labels = _labels(references_by_class, selected_by_class)
        if not labels:
            return ObjectiveResult(0.0, {}, {
                "objective": self.name, "reduction": "class_conditioned_facility_location",
                "empty": True, "per_class": {},
            })
        for label in labels:
            _validate_group(label, references_by_class.get(label, ()))
            _validate_group(label, selected_by_class.get(label, ()))
        total_classes = None
        if self.view_name == "probability":
            widths = []
            for groups in (references_by_class, selected_by_class):
                for records in groups.values():
                    for record in records:
                        view = _view(record, self.view_name)
                        if isinstance(view, Mapping):
                            values = view.get("values", view.get("probabilities"))
                            indices = view.get("class_indices")
                            width = int(view.get("n_classes", 0) or 0)
                            if indices is not None and len(indices):
                                width = max(width, max(int(value) for value in indices) + 1)
                            elif values is not None:
                                width = max(width, int(torch.as_tensor(values).numel()))
                        else:
                            width = int(torch.as_tensor(view).numel())
                        widths.append(width)
            total_classes = max(widths, default=1)

        per_class: dict[int, float] = {}
        class_diagnostics = {}
        for label in labels:
            references = list(references_by_class.get(label, ()))
            selected = list(selected_by_class.get(label, ()))
            if not references:
                class_value = 0.0
                fidelities = []
            elif not selected:
                class_value = 0.0
                fidelities = [0.0] * len(references)
            else:
                fidelities = []
                for reference in references:
                    values = [
                        self._fidelity(
                            _view(reference, self.view_name),
                            _view(option, self.view_name),
                            total_classes,
                        )
                        for option in selected
                    ]
                    fidelities.append(_finite_float(torch.stack(values).max(), self.name))
                class_value = sum(fidelities) / len(fidelities)
            per_class[label] = float(class_value)
            class_diagnostics[label] = {
                "references": len(references),
                "selected": len(selected),
                "mean_fidelity": float(class_value),
                "mean_distortion": float(1.0 - class_value) if references else 0.0,
                "minimum_reference_fidelity": min(fidelities, default=0.0),
            }
        value = sum(per_class.values()) / len(per_class)
        return ObjectiveResult(_finite_float(value, self.name), per_class, {
            "objective": self.name,
            "required_views": list(self.required_views),
            "reduction": "class_conditioned_facility_location",
            "empty": False,
            "per_class": class_diagnostics,
        })


class PredictionFidelityObjective(_FacilityFidelityObjective):
    """Preserve full-bag predictions with bounded Jensen--Shannon fidelity."""

    name = "prediction_fidelity"
    required_views = ("probability",)
    view_name = "probability"

    def _fidelity(self, reference, selected, total_classes):
        return js_fidelity(reference, selected, total_classes=total_classes)


class EmbeddingFidelityObjective(_FacilityFidelityObjective):
    """Preserve learned MIL bag embeddings."""

    name = "embedding_fidelity"
    required_views = ("embedding",)
    view_name = "embedding"

    def _fidelity(self, reference, selected, total_classes):
        del total_classes
        return normalized_cosine_fidelity(reference, selected)


class FrozenFeatureCoverageObjective(_FacilityFidelityObjective):
    """Cover pooled raw UNI morphology signatures without using the MIL model."""

    name = "frozen_feature_coverage"
    required_views = ("frozen_feature",)
    view_name = "frozen_feature"

    def _fidelity(self, reference, selected, total_classes):
        del total_classes
        return normalized_cosine_fidelity(reference, selected)


class DecisionBoundaryObjective(BaseObjective):
    """Select uncertain options under a per-class quota, without diversity terms.

    Entropy can prioritize label noise and outliers.  The diagnostic explicitly
    records that risk; this objective should therefore be treated as an
    ablation rather than automatically composed with a coverage objective.
    """

    name = "decision_boundary"
    required_views = ("boundary",)

    def __init__(self, class_quota: int | Mapping[int, int] | None = None):
        self.class_quota = class_quota

    def _quota(self, label: int, reference_count: int, selected_count: int) -> int:
        if isinstance(self.class_quota, Mapping):
            if label not in self.class_quota:
                raise ValueError(f"missing decision-boundary quota for class {label}")
            quota = int(self.class_quota[label])
        elif self.class_quota is not None:
            quota = int(self.class_quota)
        else:
            quota = max(int(reference_count), int(selected_count), 1)
        if quota < 1:
            raise ValueError("decision-boundary class quotas must be positive")
        return quota

    @staticmethod
    def _score(view: Any) -> float:
        if isinstance(view, Mapping) and "score" in view:
            value = _finite_float(view["score"], "boundary score")
            if value < 0 or value > 1:
                raise ValueError("precomputed boundary score must lie in [0, 1]")
            return value
        if isinstance(view, Mapping) and "probability" in view:
            view = view["probability"]
        return _finite_float(normalized_entropy(view), "boundary entropy")

    def evaluate(self, references_by_class, selected_by_class) -> ObjectiveResult:
        labels = _labels(references_by_class, selected_by_class)
        if not labels:
            return ObjectiveResult(0.0, {}, {
                "objective": self.name, "empty": True,
                "noise_risk": "high entropy may prioritize mislabeled or outlier slides",
            })
        per_class, details = {}, {}
        for label in labels:
            references = list(references_by_class.get(label, ()))
            selected = list(selected_by_class.get(label, ()))
            _validate_group(label, references)
            _validate_group(label, selected)
            quota = self._quota(label, len(references), len(selected))
            if len(selected) > quota:
                raise ValueError(
                    f"decision-boundary class {label} has {len(selected)} options, above quota {quota}"
                )
            scores = [self._score(_view(record, "boundary")) for record in selected]
            per_class[label] = sum(scores) / quota
            details[label] = {
                "selected": len(selected), "quota": quota,
                "mean_selected_entropy": sum(scores) / max(len(scores), 1),
                "quota_normalized_score": per_class[label],
            }
        value = sum(per_class.values()) / len(per_class)
        return ObjectiveResult(_finite_float(value, self.name), per_class, {
            "objective": self.name,
            "required_views": list(self.required_views),
            "reduction": "class_balanced_quota_normalized_entropy_sum",
            "diversity_term": False,
            "noise_risk": "high entropy may prioritize mislabeled or outlier slides",
            "empty": False,
            "per_class": details,
        })


OBJECTIVE_REGISTRY = {
    PredictionFidelityObjective.name: PredictionFidelityObjective,
    EmbeddingFidelityObjective.name: EmbeddingFidelityObjective,
    FrozenFeatureCoverageObjective.name: FrozenFeatureCoverageObjective,
    DecisionBoundaryObjective.name: DecisionBoundaryObjective,
}


def build_objective(name: str, **kwargs) -> BaseObjective:
    try:
        objective_type = OBJECTIVE_REGISTRY[str(name)]
    except KeyError as error:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise ValueError(f"unknown unified objective '{name}'; choose from: {available}") from error
    return objective_type(**kwargs)

