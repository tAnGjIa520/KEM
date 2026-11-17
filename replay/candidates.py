"""Scalable candidate generation for very large WSI bags."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def stratified_attention_indices(
    scores, patch_count, budget, method="maxminrand", device=None, seed=None
):
    """Build deterministic-size attention/random candidate sets without duplicates."""
    budget = min(max(int(budget), 0), int(patch_count))
    device = device or (scores.device if scores is not None else torch.device("cpu"))
    if budget == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    if scores is None:
        return torch.randperm(patch_count, device=device, generator=generator)[:budget]

    scores = scores.detach().reshape(-1).to(device)
    if scores.numel() != patch_count:
        raise ValueError(f"Expected {patch_count} attention scores, got {scores.numel()}")
    if method == "maxrand":
        high_count, low_count = budget // 2, 0
    elif method == "maxminrand":
        high_count, low_count = budget // 4, budget // 4
    else:
        raise ValueError(f"Unsupported stratified candidate method: {method!r}")

    groups = []
    if high_count:
        groups.append(torch.topk(scores, high_count).indices)
    if low_count:
        groups.append(torch.topk(-scores, low_count).indices)
    used = torch.unique(torch.cat(groups)) if groups else torch.empty(0, dtype=torch.long, device=device)
    available = torch.ones(patch_count, dtype=torch.bool, device=device)
    available[used] = False
    pool = torch.nonzero(available, as_tuple=False).flatten()
    need = budget - used.numel()
    if need:
        used = torch.cat([used, pool[torch.randperm(
            pool.numel(), device=device, generator=generator
        )[:need]]])
    return used[:budget]


@dataclass
class CandidatePool:
    features: torch.Tensor
    source_indices: torch.Tensor
    optimization_budget: int
    original_patch_count: int
    mode: str = "patch"
    assignments: torch.Tensor | None = None
    original_candidate_features: torch.Tensor | None = None
    original_candidate_scores: torch.Tensor | None = None

    def finalize(self, weights: torch.Tensor, final_budget: int) -> torch.Tensor:
        weights = weights.detach().float().cpu()
        if self.mode == "patch":
            local = torch.topk(weights, min(final_budget, weights.numel())).indices
            return self.source_indices.cpu().index_select(0, local)
        return self._finalize_groups(weights, final_budget)

    def _finalize_groups(self, weights: torch.Tensor, final_budget: int) -> torch.Tensor:
        assignments = self.assignments.cpu()
        group_count = self.features.shape[0]
        sizes = torch.bincount(assignments, minlength=group_count)
        probabilities = torch.softmax(weights, dim=0)
        raw = probabilities * final_budget
        quotas = torch.minimum(torch.floor(raw).long(), sizes)
        remaining = final_budget - int(quotas.sum())
        priority = torch.argsort((raw - quotas.float()) + probabilities, descending=True)
        while remaining > 0:
            changed = False
            for group in priority.tolist():
                if quotas[group] < sizes[group]:
                    quotas[group] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

        selected = []
        candidate_features = self.original_candidate_features.cpu()
        candidate_scores = self.original_candidate_scores
        if candidate_scores is not None:
            candidate_scores = candidate_scores.detach().reshape(-1).float().cpu()
        for group in range(group_count):
            quota = int(quotas[group])
            if quota == 0:
                continue
            members = torch.nonzero(assignments == group, as_tuple=False).flatten()
            important_count = quota // 2 if candidate_scores is not None else 0
            chosen = []
            if important_count:
                order = torch.topk(candidate_scores[members], min(important_count, members.numel())).indices
                chosen.append(members[order])
            used = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long)
            available_mask = torch.ones(members.numel(), dtype=torch.bool)
            if used.numel():
                positions = torch.isin(members, used)
                available_mask[positions] = False
            available = members[available_mask]
            need = quota - used.numel()
            if need and available.numel():
                centroid = self.features[group].detach().float().cpu()
                similarity = F.cosine_similarity(candidate_features[available], centroid.unsqueeze(0), dim=1)
                representative = available[torch.topk(similarity, min(need, available.numel())).indices]
                used = torch.cat([used, representative])
            selected.append(used[:quota])
        local = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
        if local.numel() < final_budget:
            available = torch.ones(self.source_indices.numel(), dtype=torch.bool)
            available[local] = False
            pool = torch.nonzero(available, as_tuple=False).flatten()
            local = torch.cat([local, pool[torch.randperm(pool.numel())[:final_budget - local.numel()]]])
        return self.source_indices.cpu().index_select(0, local[:final_budget])


def build_candidate_pool(features, scores, final_budget, config) -> CandidatePool:
    patch_count = features.shape[0]
    device = features.device
    prefilter = config.prefilter
    if prefilter == "none":
        coarse_indices = torch.arange(patch_count, device=device)
    else:
        if prefilter == "hierarchical":
            coarse_size = min(patch_count, max(512, final_budget * 4))
        else:
            coarse_size = min(patch_count, max(final_budget, int(round(final_budget * config.coarse_ratio))))
        method = "maxminrand" if prefilter == "maxminrand" else "maxrand"
        coarse_indices = stratified_attention_indices(
            scores, patch_count, coarse_size, method=method, device=device,
            seed=config.seed,
        )

    coarse_features = features.index_select(0, coarse_indices)
    coarse_scores = None if scores is None else scores.detach().reshape(-1).to(device).index_select(0, coarse_indices)
    if coarse_features.shape[0] <= config.candidate_max:
        return CandidatePool(
            features=coarse_features,
            source_indices=coarse_indices,
            optimization_budget=min(final_budget, coarse_features.shape[0]),
            original_patch_count=patch_count,
        )

    group_count = min(config.num_groups, coarse_features.shape[0])
    assignments, prototypes = _cosine_kmeans(
        coarse_features, group_count, config.cluster_dim, config.cluster_steps, config.seed
    )
    group_budget = max(1, min(group_count, round(group_count * final_budget / coarse_features.shape[0])))
    return CandidatePool(
        features=prototypes,
        source_indices=coarse_indices,
        optimization_budget=group_budget,
        original_patch_count=patch_count,
        mode="group",
        assignments=assignments.cpu(),
        original_candidate_features=coarse_features.detach().cpu(),
        original_candidate_scores=None if coarse_scores is None else coarse_scores.cpu(),
    )


def _cosine_kmeans(features, groups, projected_dim, steps, seed):
    device = features.device
    generator = torch.Generator(device=device).manual_seed(seed)
    normalized = F.normalize(features.detach().float(), dim=1)
    if normalized.shape[1] > projected_dim:
        projection = torch.randn(
            normalized.shape[1], projected_dim, device=device, generator=generator
        ) / projected_dim ** 0.5
        projected = F.normalize(normalized @ projection, dim=1)
    else:
        projected = normalized
    initial = torch.randperm(projected.shape[0], device=device, generator=generator)[:groups]
    centroids = projected.index_select(0, initial).clone()
    assignments = torch.zeros(projected.shape[0], device=device, dtype=torch.long)
    for _ in range(steps):
        for start in range(0, projected.shape[0], 8192):
            chunk = projected[start:start + 8192]
            assignments[start:start + chunk.shape[0]] = (chunk @ centroids.T).argmax(dim=1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assignments, projected)
        counts = torch.bincount(assignments, minlength=groups).clamp_min(1).to(sums.dtype)
        centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)
    prototype_sums = torch.zeros(groups, features.shape[1], device=device, dtype=features.dtype)
    prototype_sums.index_add_(0, assignments, features)
    counts = torch.bincount(assignments, minlength=groups).clamp_min(1).to(features.dtype)
    prototypes = prototype_sums / counts.unsqueeze(1)
    return assignments, prototypes
