"""Selection-only adaptation of ProDiv-style prototype pseudo-bags."""

from __future__ import annotations

import torch

from .pseudobag_prodiv_psemix_common import (
    PseudoBagSelection,
    cosine_to_prototype,
    grouping_diagnostics,
    stable_argsort,
    stratified_select,
    validate_inputs,
)


class ProDivSelector:
    """Create four prototype-similarity strata within a single WSI.

    The sorted slide is split into four equal-count strata.  An exact replay
    budget is water-filled across those strata and sampled at deterministic
    quantiles.  This is a KEM selector adapter, not ProDiv's full objective.
    """

    def __init__(self, group_count: int = 4):
        if int(group_count) < 1:
            raise ValueError("ProDiv group_count must be positive")
        self.group_count = int(group_count)

    def select(self, features: torch.Tensor, budget: int, **_: object) -> PseudoBagSelection:
        features, budget = validate_inputs(features, budget)
        _, similarities = cosine_to_prototype(features)
        order = stable_argsort(similarities)
        # Equal-count strata: floor(rank * K / N).  Empty strata are valid when
        # N < K and are ignored by the quota allocator.
        rank_groups = torch.div(
            torch.arange(features.shape[0], device=features.device) * self.group_count,
            features.shape[0],
            rounding_mode="floor",
        ).clamp_max(self.group_count - 1)
        partition_ids = torch.empty(features.shape[0], device=features.device, dtype=torch.long)
        partition_ids[order] = rank_groups
        indices, selected_groups, _ = stratified_select(
            partition_ids, similarities, budget, self.group_count
        )
        diagnostics = grouping_diagnostics(
            "prodiv", partition_ids, selected_groups, self.group_count, budget, similarities
        )
        return PseudoBagSelection(indices, selected_groups, diagnostics)


def select_prodiv(features: torch.Tensor, budget: int, group_count: int = 4) -> PseudoBagSelection:
    """Functional entry point used by the unified replay registry."""

    return ProDivSelector(group_count=group_count).select(features, budget)
