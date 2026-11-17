"""Shared primitives for deterministic phenotype pseudo-bag selectors.

The adapters in :mod:`pseudobag_prodiv` and :mod:`pseudobag_psemix` only
transfer the papers' within-slide grouping idea to replay selection.  They do
not reproduce the papers' complete training objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PseudoBagSelection:
    """Indices selected from one slide and their pseudo-bag assignments."""

    indices: torch.Tensor
    partition_ids: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)


def validate_inputs(features: torch.Tensor, budget: int) -> tuple[torch.Tensor, int]:
    if features.ndim != 2:
        raise ValueError(f"Expected [patches, features], got {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("Cannot select replay patches from an empty slide")
    if features.shape[1] == 0:
        raise ValueError("Patch features must have a non-empty feature dimension")
    if not torch.isfinite(features).all():
        raise ValueError("Patch features must be finite")
    if int(budget) < 1:
        raise ValueError("Replay patch budget must be positive")
    return features.detach().float(), min(int(budget), int(features.shape[0]))


def cosine_to_prototype(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a mean slide prototype and stable cosine similarities.

    ``F.normalize`` maps zero vectors to zero, so an all-zero slide is a
    supported (and deterministic) degenerate case rather than a NaN source.
    """

    prototype = features.mean(dim=0)
    normalized = F.normalize(features, dim=1, eps=1e-12)
    normalized_prototype = F.normalize(prototype, dim=0, eps=1e-12)
    return prototype, normalized @ normalized_prototype


def stable_argsort(values: torch.Tensor, descending: bool = False) -> torch.Tensor:
    """Stable sort with original patch order as the tie breaker."""

    try:
        return torch.argsort(values, descending=descending, stable=True)
    except TypeError:  # pragma: no cover - compatibility with old torch only
        values_cpu = values.detach().cpu().tolist()
        order = sorted(
            range(len(values_cpu)),
            key=lambda index: ((-values_cpu[index]) if descending else values_cpu[index], index),
        )
        return torch.tensor(order, device=values.device, dtype=torch.long)


def balanced_quotas(partition_ids: torch.Tensor, budget: int, group_count: int) -> torch.Tensor:
    """Water-fill an exact budget across non-empty groups.

    Quotas differ by at most one unless a small group exhausts its capacity.
    Empty groups are skipped.  Lower group IDs receive deterministic residuals.
    """

    capacities = torch.bincount(partition_ids.long(), minlength=int(group_count)).cpu()
    quotas = torch.zeros_like(capacities)
    remaining = int(budget)
    while remaining:
        eligible = torch.nonzero(quotas < capacities, as_tuple=False).flatten()
        if eligible.numel() == 0:
            raise RuntimeError("Pseudo-bag quotas cannot satisfy the requested budget")
        if remaining < eligible.numel():
            quotas[eligible[:remaining]] += 1
            remaining = 0
            continue
        share = max(1, remaining // int(eligible.numel()))
        additions = torch.minimum(
            capacities.index_select(0, eligible) - quotas.index_select(0, eligible),
            torch.full((eligible.numel(),), share, dtype=quotas.dtype),
        )
        quotas[eligible] += additions
        remaining -= int(additions.sum())
    return quotas


def quantile_positions(count: int, selected_count: int, device: torch.device) -> torch.Tensor:
    """Choose deterministic, evenly spaced ranks including group extremes."""

    if selected_count == count:
        return torch.arange(count, device=device)
    if selected_count == 1:
        return torch.tensor([(count - 1) // 2], device=device)
    positions = torch.linspace(0, count - 1, steps=selected_count, device=device)
    return positions.round().long()


def stratified_select(
    partition_ids: torch.Tensor,
    within_group_score: torch.Tensor,
    budget: int,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select exact balanced quotas and return indices, IDs, and quotas."""

    quotas = balanced_quotas(partition_ids, budget, group_count)
    selected: list[torch.Tensor] = []
    selected_groups: list[torch.Tensor] = []
    for group_id, quota in enumerate(quotas.tolist()):
        if quota == 0:
            continue
        members = torch.nonzero(partition_ids == group_id, as_tuple=False).flatten()
        # Cover the whole phenotype rather than selecting only its centroid.
        order = stable_argsort(within_group_score.index_select(0, members))
        ranked = members.index_select(0, order)
        chosen = ranked.index_select(
            0, quantile_positions(int(ranked.numel()), int(quota), ranked.device)
        )
        selected.append(chosen)
        selected_groups.append(
            torch.full((int(quota),), group_id, device=chosen.device, dtype=torch.long)
        )

    indices = torch.cat(selected)
    groups = torch.cat(selected_groups)
    if indices.numel() != budget or torch.unique(indices).numel() != budget:
        raise RuntimeError("Pseudo-bag selector did not return an exact unique budget")
    return indices.cpu(), groups.cpu(), quotas


def grouping_diagnostics(
    prefix: str,
    partition_ids: torch.Tensor,
    selected_groups: torch.Tensor,
    group_count: int,
    budget: int,
    similarities: torch.Tensor,
) -> dict[str, float]:
    nonempty = int(torch.unique(partition_ids).numel())
    covered = int(torch.unique(selected_groups).numel())
    return {
        f"{prefix}/group_count": float(group_count),
        f"{prefix}/nonempty_group_count": float(nonempty),
        f"{prefix}/selected_group_count": float(covered),
        f"{prefix}/group_coverage": float(covered / max(nonempty, 1)),
        f"{prefix}/selected_patch_count": float(budget),
        f"{prefix}/similarity_mean": float(similarities.mean().cpu()),
        f"{prefix}/similarity_std": float(similarities.std(unbiased=False).cpu()),
    }
