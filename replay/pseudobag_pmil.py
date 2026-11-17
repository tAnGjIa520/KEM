"""PMIL pseudo-bag ranking adapted as a replay patch selector.

The original PMIL implementation [1] sorts patches by attention, optionally
re-ranks a small high-attention candidate set with Shapley-style marginal
contributions, and distributes the resulting order round-robin over pseudo
bags.  This module preserves that candidate/ranking rule while exposing an
exact-budget, CLAM-only selector for continual replay.

This is an adapter, not a reimplementation of PMIL's progressive training
schedule or pseudo-bag augmentation loss.

References
----------
[1] Yan et al., "Shapley Values-enabled Progressive Pseudo Bag Augmentation
    for Whole-Slide Image Classification", IEEE TMI, 2024.
    https://github.com/RenaoYan/PMIL
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn


_SUPPORTED_METHODS = frozenset({"pmil_attention", "pmil_shapley"})


@dataclass(frozen=True)
class PMILConfig:
    """Configuration matching PMIL's public defaults where applicable."""

    m_max: int = 10
    search_rate: int = 10
    subset_count: int = 3
    seed: int = 1
    timeout_seconds: float = 60.0

    @classmethod
    def from_args(cls, args) -> "PMILConfig":
        return cls(
            m_max=int(getattr(args, "pmil_m_max", 10)),
            search_rate=int(getattr(args, "pmil_search_rate", 10)),
            subset_count=int(getattr(args, "pmil_subset_count", 3)),
            seed=int(getattr(args, "seed", 1)),
            timeout_seconds=float(getattr(args, "pmil_timeout_seconds", 60.0)),
        )

    def __post_init__(self) -> None:
        if self.m_max < 1:
            raise ValueError("pmil_m_max must be positive")
        if self.search_rate < 1:
            raise ValueError("pmil_search_rate must be positive")
        if self.subset_count < 1:
            raise ValueError("pmil_subset_count must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("pmil_timeout_seconds must be finite and positive")


@dataclass(frozen=True)
class PMILSelectionResult:
    """Selected patch indices and their PMIL pseudo-bag assignments."""

    indices: torch.Tensor
    partition_ids: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)


def _scalar_label(label) -> int:
    if label is None:
        raise ValueError("PMIL selection requires a label")
    value = torch.as_tensor(label).detach().reshape(-1)
    if value.numel() != 1:
        raise ValueError("PMIL label must contain exactly one class index")
    result = int(value.item())
    if result < 0:
        raise ValueError("PMIL label must be non-negative")
    return result


def _attention_for_label(scores, patch_count: int, label: int) -> torch.Tensor:
    if scores is None:
        raise ValueError("PMIL selection requires attention scores")
    attention = torch.as_tensor(scores).detach().cpu()
    if not attention.is_floating_point():
        attention = attention.float()
    if attention.ndim == 1:
        if attention.numel() != patch_count:
            raise ValueError(
                f"attention has {attention.numel()} entries for {patch_count} patches"
            )
        result = attention
    elif attention.ndim == 2:
        rows, columns = attention.shape
        if rows == 1 and columns == patch_count:
            result = attention[0]
        elif columns == 1 and rows == patch_count:
            result = attention[:, 0]
        elif columns == patch_count and rows != patch_count:
            if label >= rows:
                raise ValueError(
                    f"label {label} is outside {rows} attention branches"
                )
            result = attention[label]
        elif rows == patch_count and columns != patch_count:
            if label >= columns:
                raise ValueError(
                    f"label {label} is outside {columns} attention branches"
                )
            result = attention[:, label]
        else:
            raise ValueError(
                "attention must be [N], [1,N], [N,1], [classes,N], or [N,classes]"
            )
    else:
        raise ValueError(f"attention must be one- or two-dimensional, got {attention.ndim}")
    result = result.to(dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("attention scores must all be finite")
    return result


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Sort descending while retaining the input order for exact ties."""

    return torch.argsort(values, descending=True, stable=True)


def _validate_clam(model: nn.Module, label: int) -> tuple[bool, nn.Module]:
    if model is None:
        raise ValueError("pmil_shapley selection requires a CLAM model")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "attention_net") or not hasattr(model, "classifiers"):
        raise TypeError("pmil_shapley supports CLAM_SB and CLAM_MB only")
    classifiers = model.classifiers
    if isinstance(classifiers, nn.Linear):
        if label >= classifiers.out_features:
            raise ValueError(f"label {label} is outside the CLAM classifier")
        return False, classifiers
    if isinstance(classifiers, nn.ModuleList):
        if label >= len(classifiers) or not isinstance(classifiers[label], nn.Linear):
            raise TypeError("CLAM_MB must have one linear classifier per class")
        return True, classifiers[label]
    raise TypeError("pmil_shapley supports CLAM_SB and CLAM_MB classifiers only")


def _validate_indices(
    indices: torch.Tensor | Iterable[int], patch_count: int, name: str
) -> torch.Tensor:
    result = torch.as_tensor(indices, dtype=torch.long).detach().cpu().reshape(-1)
    if result.numel() and (
        int(result.min()) < 0 or int(result.max()) >= patch_count
    ):
        raise IndexError(f"{name} contains an out-of-range patch index")
    if torch.unique(result).numel() != result.numel():
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


def _module_modes(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    return [(module, bool(module.training)) for module in model.modules()]


def _restore_module_modes(states: Sequence[tuple[nn.Module, bool]]) -> None:
    # Direct assignment restores even deliberately mixed train/eval submodule
    # states, whereas model.train(was_training) would flatten those states.
    for module, training in states:
        module.training = training


def _clam_components(
    model: nn.Module,
    features: torch.Tensor,
    label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_multi_branch, classifier = _validate_clam(model, label)
    raw_attention, hidden = model.attention_net(features)
    if raw_attention.ndim != 2 or hidden.ndim != 2:
        raise TypeError("CLAM attention_net must return [N,K] attention and [N,L] features")
    branch = label if is_multi_branch else 0
    if branch >= raw_attention.shape[1]:
        raise ValueError(
            f"CLAM attention has {raw_attention.shape[1]} branches, cannot use {branch}"
        )
    weight = classifier.weight[0] if is_multi_branch else classifier.weight[label]
    if hidden.shape[1] != weight.numel():
        raise ValueError("CLAM hidden width does not match its classifier")
    return raw_attention[:, branch], hidden, weight


def make_pmil_reference_subsets(
    patch_count: int,
    candidate_indices: torch.Tensor | Iterable[int],
    subset_count: int = 3,
    *,
    seed: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Build PMIL's shuffled, interleaved reference subsets deterministically."""

    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    if subset_count < 1:
        raise ValueError("subset_count must be positive")
    candidates = _validate_indices(candidate_indices, patch_count, "candidate_indices")
    mask = torch.ones(patch_count, dtype=torch.bool)
    mask[candidates] = False
    remaining = torch.nonzero(mask, as_tuple=False).flatten()
    if remaining.numel() == 0:
        raise ValueError("PMIL Shapley estimation requires at least one reference patch")
    generator = torch.Generator().manual_seed(int(seed))
    remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]
    count = min(int(subset_count), int(remaining.numel()))
    return tuple(remaining[offset::count].clone() for offset in range(count))


def clam_analytic_marginal_contributions(
    model: nn.Module,
    features: torch.Tensor,
    label,
    candidate_indices: torch.Tensor | Iterable[int],
    reference_subsets: Sequence[torch.Tensor | Iterable[int]],
    *,
    deadline: float | None = None,
) -> torch.Tensor:
    """Compute exact CLAM candidate-addition margins without repeated forwards.

    CLAM uses a softmax attention weighted sum followed by a linear classifier.
    Consequently the target-logit change after adding any candidate to a
    reference subset has a closed form.  Candidates are batched and only one
    attention-network evaluation is required for the full slide.
    """

    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("features must be non-empty [patches, dimensions]")
    target = _scalar_label(label)
    candidates = _validate_indices(candidate_indices, features.shape[0], "candidate_indices")
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.float64)
    subsets = [
        _validate_indices(indices, features.shape[0], "reference_subset")
        for indices in reference_subsets
    ]
    if not subsets or any(indices.numel() == 0 for indices in subsets):
        raise ValueError("reference_subsets must contain non-empty subsets")
    candidate_set = set(candidates.tolist())
    if any(candidate_set.intersection(indices.tolist()) for indices in subsets):
        raise ValueError("reference subsets must exclude all candidate patches")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("PMIL Shapley selection exceeded its time limit")

    _validate_clam(model, target)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TypeError("CLAM model must have parameters") from error
    model_dtype = next(model.parameters()).dtype
    x = features.detach().to(device=device, dtype=model_dtype)
    states = _module_modes(model)
    try:
        model.eval()
        with torch.no_grad():
            raw, hidden, classifier_weight = _clam_components(model, x, target)
            candidate_device = candidates.to(device)
            candidate_raw = raw.index_select(0, candidate_device)
            candidate_hidden = hidden.index_select(0, candidate_device)
            total = hidden.new_zeros(candidates.numel())
            for subset in subsets:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("PMIL Shapley selection exceeded its time limit")
                subset_device = subset.to(device)
                base_raw = raw.index_select(0, subset_device)
                base_hidden = hidden.index_select(0, subset_device)

                base_max = base_raw.max()
                base_weights = torch.exp(base_raw - base_max)
                base_denominator = base_weights.sum()
                base_numerator = (base_weights[:, None] * base_hidden).sum(dim=0)
                base_embedding = base_numerator / base_denominator

                joint_max = torch.maximum(candidate_raw, base_max)
                base_scale = torch.exp(base_max - joint_max)
                scaled_denominator = base_denominator * base_scale
                scaled_numerator = base_numerator[None, :] * base_scale[:, None]
                candidate_weights = torch.exp(candidate_raw - joint_max)
                joined_embedding = (
                    scaled_numerator + candidate_weights[:, None] * candidate_hidden
                ) / (scaled_denominator + candidate_weights)[:, None]
                embedding_delta = joined_embedding - base_embedding[None, :]
                total.add_(embedding_delta @ classifier_weight)
            total.div_(len(subsets))
        return total.detach().to(device="cpu", dtype=torch.float64)
    finally:
        _restore_module_modes(states)


def clam_bruteforce_marginal_contributions(
    model: nn.Module,
    features: torch.Tensor,
    label,
    candidate_indices: torch.Tensor | Iterable[int],
    reference_subsets: Sequence[torch.Tensor | Iterable[int]],
    *,
    deadline: float | None = None,
) -> torch.Tensor:
    """Reference implementation used to verify the analytic CLAM formula."""

    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("features must be non-empty [patches, dimensions]")
    target = _scalar_label(label)
    candidates = _validate_indices(candidate_indices, features.shape[0], "candidate_indices")
    subsets = [
        _validate_indices(indices, features.shape[0], "reference_subset")
        for indices in reference_subsets
    ]
    if not subsets or any(indices.numel() == 0 for indices in subsets):
        raise ValueError("reference_subsets must contain non-empty subsets")
    candidate_set = set(candidates.tolist())
    if any(candidate_set.intersection(indices.tolist()) for indices in subsets):
        raise ValueError("reference subsets must exclude all candidate patches")
    _validate_clam(model, target)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TypeError("CLAM model must have parameters") from error
    model_dtype = next(model.parameters()).dtype
    x = features.detach().to(device=device, dtype=model_dtype)
    states = _module_modes(model)
    try:
        model.eval()
        with torch.no_grad():
            total = x.new_zeros(candidates.numel())
            for subset in subsets:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("PMIL Shapley selection exceeded its time limit")
                base = x.index_select(0, subset.to(device))
                base_logits = model(base, return_features=True)["logits"]
                for offset, candidate in enumerate(candidates):
                    joined = torch.cat((
                        x[int(candidate)].unsqueeze(0),
                        base,
                    ), dim=0)
                    logits = model(joined, return_features=True)["logits"]
                    total[offset] += logits[0, target] - base_logits[0, target]
            total.div_(len(subsets))
        return total.detach().to(device="cpu", dtype=torch.float64)
    finally:
        _restore_module_modes(states)


def _effective_rank(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return float(features.shape[0])
    values = features.detach().float().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(values)
    total = singular.sum()
    if float(total) <= 0:
        return 0.0
    probability = singular / total
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _selection_seed(seed: int, slide_id, task) -> int:
    material = f"{int(seed)}\0{slide_id!s}\0{task!s}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    # torch.Generator.manual_seed accepts signed 64-bit seeds portably.
    return int.from_bytes(digest, "little") % (2**63 - 1)


class PMILSelector:
    """Exact-budget replay selector using PMIL attention or Shapley ranking."""

    def __init__(self, method: str, config: PMILConfig | None = None):
        if method not in _SUPPORTED_METHODS:
            raise ValueError(f"Unknown PMIL selector: {method!r}")
        self.method = method
        self.config = config or PMILConfig()

    @classmethod
    def from_args(cls, method: str, args) -> "PMILSelector":
        return cls(method, PMILConfig.from_args(args))

    def select(
        self,
        features: torch.Tensor,
        budget: int,
        scores=None,
        model: nn.Module | None = None,
        label=None,
        seen_classes=None,
        coords=None,
        slide_id=None,
        task=None,
    ) -> PMILSelectionResult:
        del seen_classes, coords  # Accepted for the common replay-selector API.
        started = time.monotonic()
        deadline = started + self.config.timeout_seconds
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("PMIL expects non-empty [patches, dimensions] features")
        if not bool(torch.isfinite(features.detach()).all()):
            raise ValueError("PMIL features must all be finite")
        requested_budget = int(budget)
        if requested_budget < 1:
            raise ValueError("PMIL budget must be positive")
        patch_count = int(features.shape[0])
        exact_budget = min(requested_budget, patch_count)
        target = _scalar_label(label)
        attention = _attention_for_label(scores, patch_count, target)
        attention_order = _stable_descending(attention)
        partition_count = min(self.config.m_max, exact_budget)
        search_count = 0
        shapley = torch.empty(0, dtype=torch.float64)

        if self.method == "pmil_attention":
            ranking = attention_order
        else:
            _validate_clam(model, target)
            # This is the official PMIL candidate rule:
            # min(number_of_pseudo_bags * search_rate, floor(N / 2)).
            search_count = min(
                partition_count * self.config.search_rate,
                patch_count // 2,
            )
            if search_count:
                candidates = attention_order[:search_count]
                seed = _selection_seed(self.config.seed, slide_id, task)
                subsets = make_pmil_reference_subsets(
                    patch_count,
                    candidates,
                    self.config.subset_count,
                    seed=seed,
                )
                shapley = clam_analytic_marginal_contributions(
                    model,
                    features,
                    target,
                    candidates,
                    subsets,
                    deadline=deadline,
                )
                # Stable tie handling retains the attention order, just as the
                # official candidate array does before its Shapley re-ranking.
                candidate_order = _stable_descending(shapley)
                ranking = torch.cat((
                    candidates.index_select(0, candidate_order),
                    attention_order[search_count:],
                ))
            else:
                ranking = attention_order

        if time.monotonic() >= deadline:
            raise TimeoutError("PMIL selection exceeded its time limit")
        indices = ranking[:exact_budget].to(dtype=torch.long, device="cpu")
        if indices.numel() != exact_budget or torch.unique(indices).numel() != exact_budget:
            raise RuntimeError("PMIL failed to produce an exact unique patch budget")
        partition_ids = torch.arange(exact_budget, dtype=torch.long) % partition_count
        elapsed = time.monotonic() - started
        diagnostics = {
            "selection_time_sec": float(elapsed),
            "patch_count": float(patch_count),
            "selected_count": float(exact_budget),
            "partition_count": float(partition_count),
            "partition_coverage": float(torch.unique(partition_ids).numel() / partition_count),
            "search_count": float(search_count),
            "search_fraction": float(search_count / patch_count),
            "effective_rank": _effective_rank(features.index_select(0, indices.to(features.device))),
        }
        if shapley.numel():
            diagnostics.update({
                "shapley_mean": float(shapley.mean()),
                "shapley_std": float(shapley.std(unbiased=False)),
                "shapley_min": float(shapley.min()),
                "shapley_max": float(shapley.max()),
            })
        return PMILSelectionResult(indices, partition_ids, diagnostics)


__all__ = [
    "PMILConfig",
    "PMILSelectionResult",
    "PMILSelector",
    "clam_analytic_marginal_contributions",
    "clam_bruteforce_marginal_contributions",
    "make_pmil_reference_subsets",
]
