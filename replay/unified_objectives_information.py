"""Information-space objectives for unified KEM replay allocation.

This module is intentionally independent from the buffer and its allocator.  It
contains pure, deterministic objective functions over projected gradient
signatures.  All objectives use the same convention: *larger is better*.

``ObjectivePoint.weight`` is replay probability/loss mass.  It is not patch
cost and it is not an extra objective coefficient.  ``uniform`` replay ignores
this field and gives each selected pseudo-bag equal mass within its class;
``point`` replay uses it directly; ``assignment`` replay recomputes the
facility-location assignment mass used by :mod:`unified_gradient_coverage`.

Mean matching is non-monotone.  Its marginal gain is therefore deliberately
signed and never clipped at zero.  An exact-budget allocator must continue with
the best feasible move, even when every remaining marginal is negative, and
must fail explicitly if the candidate depths cannot close the exact budget.
Objective choice or replay weighting must be selected on training/validation
data only; test-set performance is not an input to any function here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


_VALID_WEIGHT_MODES = frozenset({"uniform", "point", "assignment"})


@dataclass(frozen=True)
class ObjectivePoint:
    """One reference WSI or one selected pseudo-bag signature.

    ``key`` is the stable slide/option identifier used for deterministic
    replacement and tie-breaking.  ``cost`` is metadata for an allocator and
    is never folded into the objective value.
    """

    key: str
    label: int
    value: torch.Tensor
    weight: float = 1.0
    cost: int = 1

    def __post_init__(self):
        value = torch.as_tensor(self.value, dtype=torch.float64).detach().cpu().reshape(-1).clone()
        if value.numel() < 1 or not torch.isfinite(value).all():
            raise ValueError(f"objective point {self.key!r} has an invalid signature")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"objective point {self.key!r} has an invalid weight")
        if int(self.cost) < 0:
            raise ValueError(f"objective point {self.key!r} has a negative cost")
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "cost", int(self.cost))


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Macro class objective and diagnostics; larger ``value`` is better."""

    value: float
    per_class: Mapping[int, float]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ObjectiveMarginal:
    """Signed change caused by adding or replacing an objective point."""

    delta: float
    before: ObjectiveEvaluation
    after: ObjectiveEvaluation


@dataclass(frozen=True)
class WorstClassLexicographicState:
    """Two-level robust policy state, without a weighted-sum coefficient."""

    worst_class_value: float
    mean_class_value: float
    worst_classes: tuple[int, ...]
    per_class: tuple[tuple[int, float], ...]

    @classmethod
    def from_evaluation(cls, evaluation: ObjectiveEvaluation):
        if not evaluation.per_class:
            return cls(0.0, 0.0, (), ())
        values = {int(label): float(value) for label, value in evaluation.per_class.items()}
        worst = min(values.values())
        worst_classes = tuple(sorted(label for label, value in values.items() if value == worst))
        return cls(
            worst_class_value=worst,
            mean_class_value=sum(values.values()) / len(values),
            worst_classes=worst_classes,
            per_class=tuple(sorted(values.items())),
        )


def prefer_worst_class_lexicographic(
    left: ObjectiveEvaluation | WorstClassLexicographicState,
    right: ObjectiveEvaluation | WorstClassLexicographicState,
    *,
    left_cost: int,
    right_cost: int,
    left_stable_id: str,
    right_stable_id: str,
) -> bool:
    """Return whether ``left`` is preferred by the robust deterministic policy.

    The comparison order is exact and parameter-free: largest worst-class
    value, largest macro class value, lowest cost, then lexicographically
    smallest stable identifier.  Callers should not convert it into a weighted
    sum because that changes the policy and introduces a scale hyperparameter.
    """

    if isinstance(left, ObjectiveEvaluation):
        left = WorstClassLexicographicState.from_evaluation(left)
    if isinstance(right, ObjectiveEvaluation):
        right = WorstClassLexicographicState.from_evaluation(right)
    if left.worst_class_value != right.worst_class_value:
        return left.worst_class_value > right.worst_class_value
    if left.mean_class_value != right.mean_class_value:
        return left.mean_class_value > right.mean_class_value
    if int(left_cost) != int(right_cost):
        return int(left_cost) < int(right_cost)
    return str(left_stable_id) < str(right_stable_id)


def _coerce_points_by_class(
    points_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
    *,
    name: str,
) -> dict[int, tuple[ObjectivePoint, ...]]:
    result: dict[int, tuple[ObjectivePoint, ...]] = {}
    dimension = None
    seen_keys = set()
    for raw_label in sorted(points_by_class):
        label = int(raw_label)
        raw_points = points_by_class[raw_label]
        if isinstance(raw_points, torch.Tensor):
            matrix = raw_points.detach().cpu()
            if matrix.ndim == 1:
                matrix = matrix.unsqueeze(0)
            if matrix.ndim != 2:
                raise ValueError(f"{name}[{label}] must be a signature matrix")
            points = tuple(
                ObjectivePoint(f"{name}:{label}:{index}", label, row)
                for index, row in enumerate(matrix)
            )
        else:
            points = tuple(raw_points)
            if any(not isinstance(point, ObjectivePoint) for point in points):
                raise TypeError(f"{name}[{label}] must contain ObjectivePoint values")
            if any(point.label != label for point in points):
                raise ValueError(f"{name}[{label}] contains a different class label")
        for point in points:
            if point.key in seen_keys:
                raise ValueError(f"duplicate objective point key: {point.key!r}")
            seen_keys.add(point.key)
            if dimension is None:
                dimension = point.value.numel()
            elif point.value.numel() != dimension:
                raise ValueError("all objective signatures must have the same dimension")
        result[label] = points
    return result


def _validate_inputs(references, selected):
    if not references or any(not points for points in references.values()):
        raise ValueError("every reference class must contain at least one WSI")
    extra = sorted(set(selected) - set(references))
    if extra:
        raise ValueError(f"selected signatures contain unknown classes: {extra}")
    dimension = next(iter(next(iter(references.values())))).value.numel()
    for points in selected.values():
        for point in points:
            if point.value.numel() != dimension:
                raise ValueError("reference and selected signature dimensions differ")


def assignment_masses(
    references_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
    selected_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
) -> tuple[dict[str, int], dict[str, float]]:
    """Compute the same class-macro nearest-cosine masses as the v1 allocator.

    Each reference in class ``c`` carries mass ``1 / (C * N_c)``.  It is
    assigned to the selected pseudo-bag with largest non-negative cosine.
    Selected points are sorted by stable key so exact ties are deterministic.
    If a class has no selected point, its mass remains unassigned.
    """

    references = _coerce_points_by_class(references_by_class, name="reference")
    selected = _coerce_points_by_class(selected_by_class, name="selected")
    _validate_inputs(references, selected)
    counts: dict[str, int] = {}
    masses: dict[str, float] = {}
    class_count = len(references)
    for label, refs in references.items():
        options = sorted(selected.get(label, ()), key=lambda point: point.key)
        if not options:
            continue
        reference_matrix = F.normalize(torch.stack([point.value for point in refs]), dim=1, eps=1e-12)
        option_matrix = F.normalize(torch.stack([point.value for point in options]), dim=1, eps=1e-12)
        similarities = (reference_matrix @ option_matrix.T).clamp_min(0)
        assigned = similarities.argmax(dim=1)
        mass = 1.0 / (class_count * len(refs))
        for option_index in assigned.tolist():
            key = options[option_index].key
            counts[key] = counts.get(key, 0) + 1
            masses[key] = masses.get(key, 0.0) + mass
    return counts, masses


def _weights_for_selected(references, selected, replay_weighting):
    if replay_weighting not in _VALID_WEIGHT_MODES:
        raise ValueError(f"unknown replay weighting: {replay_weighting!r}")
    if replay_weighting == "assignment":
        _, masses = assignment_masses(references, selected)
        return {
            point.key: float(masses.get(point.key, 0.0))
            for points in selected.values() for point in points
        }, masses
    if replay_weighting == "point":
        weights = {
            point.key: point.weight for points in selected.values() for point in points
        }
        return weights, {}
    weights = {point.key: 1.0 for points in selected.values() for point in points}
    return weights, {}


def _weighted_mean(points: Sequence[ObjectivePoint], weights: Mapping[str, float], dimension: int):
    if not points:
        return torch.zeros(dimension, dtype=torch.float64), 0.0
    values = torch.stack([point.value for point in points])
    mass = torch.tensor([float(weights[point.key]) for point in points], dtype=torch.float64)
    total = float(mass.sum())
    if total <= 0:
        return torch.zeros(dimension, dtype=torch.float64), 0.0
    return (values * mass.unsqueeze(1)).sum(dim=0) / total, total


def _evaluate_mean_gradient_matching(references, selected, replay_weighting):
    selected_weights, masses = _weights_for_selected(references, selected, replay_weighting)
    per_class, squared_errors, selected_mass = {}, {}, {}
    dimension = next(iter(next(iter(references.values())))).value.numel()
    for label, refs in references.items():
        reference_weights = {point.key: point.weight for point in refs}
        reference_mean, _ = _weighted_mean(refs, reference_weights, dimension)
        replay_mean, mass = _weighted_mean(selected.get(label, ()), selected_weights, dimension)
        squared_error = float(torch.sum((reference_mean - replay_mean) ** 2))
        squared_errors[label] = squared_error
        per_class[label] = -squared_error
        selected_mass[label] = mass
    value = sum(per_class.values()) / len(per_class)
    return ObjectiveEvaluation(
        value=value,
        per_class=per_class,
        diagnostics={
            "objective": "class_mean_gradient_matching",
            "direction": "maximize_negative_squared_error",
            "empty_selected_mean": "zero_vector",
            "class_squared_error": squared_errors,
            "selected_weight_mass": selected_mass,
            "replay_weighting": replay_weighting,
            "assignment_mass": masses,
            "monotone": False,
        },
    )


def _stable_logdet(signatures: Sequence[torch.Tensor], weights: Sequence[float]):
    if not signatures:
        return 0.0, "empty"
    matrix = torch.stack(tuple(signatures)).to(dtype=torch.float64)
    mass = torch.tensor(tuple(weights), dtype=torch.float64)
    positive = mass > 0
    if not bool(positive.any()):
        return 0.0, "zero_weight"
    matrix = matrix[positive] * mass[positive].sqrt().unsqueeze(1)
    # Matrix determinant lemma: use the smaller of I + XX^T and I + X^TX.
    if matrix.shape[0] <= matrix.shape[1]:
        gram = torch.eye(matrix.shape[0], dtype=torch.float64) + matrix @ matrix.T
        factorization = "sample_gram_cholesky"
    else:
        gram = torch.eye(matrix.shape[1], dtype=torch.float64) + matrix.T @ matrix
        factorization = "feature_gram_cholesky"
    chol, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    if int(info.max()) == 0:
        value = float(2.0 * torch.log(torch.diagonal(chol)).sum())
    else:  # Defensive fallback; I + X X^T is positive definite in exact arithmetic.
        sign, logabsdet = torch.linalg.slogdet(gram)
        if float(sign) <= 0:
            raise FloatingPointError("Fisher information matrix is not positive definite")
        value = float(logabsdet)
        factorization += "_slogdet_fallback"
    if not math.isfinite(value):
        raise FloatingPointError("non-finite Fisher log determinant")
    return value, factorization


def _evaluate_fisher_logdet(references, selected, replay_weighting):
    selected_weights, masses = _weights_for_selected(references, selected, replay_weighting)
    per_class, factorizations = {}, {}
    for label in references:
        points = selected.get(label, ())
        score, factorization = _stable_logdet(
            [point.value for point in points],
            [selected_weights[point.key] for point in points],
        )
        per_class[label] = score
        factorizations[label] = factorization
    value = sum(per_class.values()) / len(per_class)
    return ObjectiveEvaluation(
        value=value,
        per_class=per_class,
        diagnostics={
            "objective": "fisher_logdet",
            "formula": "log det(I + sum_i w_i psi_i psi_i^T)",
            "factorization": factorizations,
            "replay_weighting": replay_weighting,
            "assignment_mass": masses,
            "monotone": True,
            "signature_norm_note": (
                "The current 109b28e signatures are unit-normalized, so this measures "
                "directional D-optimal diversity rather than gradient magnitude."
            ),
        },
    )


def evaluate_objective(
    name: str,
    references_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
    selected_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
    *,
    replay_weighting: str = "uniform",
) -> ObjectiveEvaluation:
    """Evaluate a class-macro projected-gradient information objective."""

    references = _coerce_points_by_class(references_by_class, name="reference")
    selected = _coerce_points_by_class(selected_by_class, name="selected")
    _validate_inputs(references, selected)
    normalized_name = str(name).strip().lower().replace("-", "_")
    if normalized_name == "class_mean_gradient_matching":
        return _evaluate_mean_gradient_matching(references, selected, replay_weighting)
    if normalized_name in {"fisher_logdet", "d_optimal", "d_optimality"}:
        return _evaluate_fisher_logdet(references, selected, replay_weighting)
    raise ValueError(f"unknown information objective: {name!r}")


class InformationObjectiveState:
    """Immutable-style state API for deterministic add/replace/remove scoring."""

    def __init__(
        self,
        name: str,
        references_by_class: Mapping[int, Sequence[ObjectivePoint] | torch.Tensor],
        selected: Iterable[ObjectivePoint] = (),
        *,
        replay_weighting: str = "uniform",
    ):
        self.name = str(name)
        self.references_by_class = _coerce_points_by_class(references_by_class, name="reference")
        self.replay_weighting = str(replay_weighting)
        selected_by_key = {}
        for point in selected:
            if point.key in selected_by_key:
                raise ValueError(f"duplicate selected point key: {point.key!r}")
            selected_by_key[point.key] = point
        self._selected_by_key = selected_by_key
        self.evaluation = evaluate_objective(
            self.name, self.references_by_class, self.selected_by_class,
            replay_weighting=self.replay_weighting,
        )

    @property
    def selected(self):
        return tuple(self._selected_by_key[key] for key in sorted(self._selected_by_key))

    @property
    def selected_by_class(self):
        result = {label: [] for label in self.references_by_class}
        for point in self.selected:
            result.setdefault(point.label, []).append(point)
        return {label: tuple(points) for label, points in result.items()}

    def with_point(self, point: ObjectivePoint):
        """Return a state in which an existing stable key is replaced atomically."""

        updated = dict(self._selected_by_key)
        updated[point.key] = point
        return InformationObjectiveState(
            self.name, self.references_by_class, updated.values(),
            replay_weighting=self.replay_weighting,
        )

    def without_point(self, key: str):
        updated = dict(self._selected_by_key)
        updated.pop(str(key), None)
        return InformationObjectiveState(
            self.name, self.references_by_class, updated.values(),
            replay_weighting=self.replay_weighting,
        )

    def marginal(self, point: ObjectivePoint):
        after_state = self.with_point(point)
        return ObjectiveMarginal(
            delta=after_state.evaluation.value - self.evaluation.value,
            before=self.evaluation,
            after=after_state.evaluation,
        )


def marginal_objective(state: InformationObjectiveState, point: ObjectivePoint) -> ObjectiveMarginal:
    """Functional alias used by allocators; the delta is signed, not clipped."""

    return state.marginal(point)
