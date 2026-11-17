"""Selection-only adaptation of PseMix-style phenotype pseudo-bags."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pseudobag_prodiv_psemix_common import (
    PseudoBagSelection,
    cosine_to_prototype,
    grouping_diagnostics,
    stratified_select,
    validate_inputs,
)


class PseMixSelector:
    """Initialize fixed-width phenotype bins, then refine cosine centroids.

    No Mixup or synthetic memory is applied: only the within-slide phenotype
    grouping and balanced selection behavior is adapted for fair KEM replay.
    """

    def __init__(self, group_count: int = 8, refinement_steps: int = 8):
        if int(group_count) < 1:
            raise ValueError("PseMix group_count must be positive")
        if int(refinement_steps) < 0:
            raise ValueError("PseMix refinement_steps must be non-negative")
        self.group_count = int(group_count)
        self.refinement_steps = int(refinement_steps)

    def _initial_partitions(self, similarities: torch.Tensor) -> torch.Tensor:
        # Cosine similarity lies in [-1, 1].  Clamp explicitly to make the
        # upper endpoint land in the final fixed-width bin.
        scaled = ((similarities.clamp(-1, 1) + 1) * (self.group_count / 2)).floor().long()
        return scaled.clamp_(0, self.group_count - 1)

    def _refine(
        self, features: torch.Tensor, partitions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(features, dim=1, eps=1e-12)
        assignments = partitions.clone()
        member_similarity = features.new_zeros(features.shape[0])
        for _ in range(self.refinement_steps):
            active_ids = torch.unique(assignments, sorted=True)
            centroids = []
            for group_id in active_ids.tolist():
                members = normalized[assignments == group_id]
                centroids.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-12))
            centroid_matrix = torch.stack(centroids)
            cosine = normalized @ centroid_matrix.T
            best = cosine.argmax(dim=1)
            assignments = active_ids.index_select(0, best)
            member_similarity = cosine.gather(1, best.unsqueeze(1)).squeeze(1)

        if self.refinement_steps == 0:
            active_ids = torch.unique(assignments, sorted=True)
            for group_id in active_ids.tolist():
                mask = assignments == group_id
                centroid = F.normalize(normalized[mask].mean(dim=0), dim=0, eps=1e-12)
                member_similarity[mask] = normalized[mask] @ centroid
        return assignments, member_similarity

    def select(self, features: torch.Tensor, budget: int, **_: object) -> PseudoBagSelection:
        features, budget = validate_inputs(features, budget)
        _, slide_similarities = cosine_to_prototype(features)
        initial = self._initial_partitions(slide_similarities)
        partitions, member_similarity = self._refine(features, initial)
        indices, selected_groups, _ = stratified_select(
            partitions, member_similarity, budget, self.group_count
        )
        diagnostics = grouping_diagnostics(
            "psemix", partitions, selected_groups, self.group_count, budget, slide_similarities
        )
        diagnostics.update(
            {
                "psemix/initial_nonempty_bin_count": float(torch.unique(initial).numel()),
                "psemix/refinement_steps": float(self.refinement_steps),
                "psemix/refined_centroid_count": float(torch.unique(partitions).numel()),
                "psemix/mixup_enabled": 0.0,
            }
        )
        return PseudoBagSelection(indices, selected_groups, diagnostics)


def select_psemix(
    features: torch.Tensor,
    budget: int,
    group_count: int = 8,
    refinement_steps: int = 8,
) -> PseudoBagSelection:
    """Functional entry point used by the unified replay registry."""

    return PseMixSelector(group_count, refinement_steps).select(features, budget)
