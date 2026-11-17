"""Deterministic spatial and MMIL-style pseudo-bag patch selectors.

The selectors in this module deliberately do not depend on the continual
learning registry.  They expose a small adapter-friendly result type so they
can be tested and integrated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True)
class PseudoBagSelectionResult:
    """Selected patch indices and pseudo-bag membership.

    ``partition_ids`` is aligned with ``indices`` (rather than with every
    input patch).  Both tensors live on CPU to match replay-buffer indexing.
    """

    indices: torch.Tensor
    partition_ids: torch.Tensor
    diagnostics: dict[str, float]


class MMILPseudoBagSelector:
    """Select a fixed budget of patches using spatial or MMIL grouping."""

    _METHODS = {"spatial", "mmil_random", "mmil_embed"}

    def __init__(
        self,
        method: str,
        num_groups: int = 4,
        seed: int = 0,
        kmeans_steps: int = 8,
        cluster_dim: int = 64,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"Unknown pseudo-bag selector: {method!r}")
        if int(num_groups) < 1:
            raise ValueError("num_groups must be positive")
        if int(kmeans_steps) < 1:
            raise ValueError("kmeans_steps must be positive")
        if int(cluster_dim) < 1:
            raise ValueError("cluster_dim must be positive")
        self.method = method
        self.num_groups = int(num_groups)
        self.seed = int(seed)
        self.kmeans_steps = int(kmeans_steps)
        self.cluster_dim = int(cluster_dim)

    def select(
        self,
        features: torch.Tensor,
        budget: int,
        *,
        coords: torch.Tensor | None = None,
        slide_id: str | None = None,
        task: int | str | None = None,
    ) -> PseudoBagSelectionResult:
        """Return exactly ``min(budget, patch_count)`` unique patch indices."""

        _validate_features(features)
        requested_budget = int(budget)
        if requested_budget < 1:
            raise ValueError("budget must be positive")
        patch_count = int(features.shape[0])
        selected_budget = min(requested_budget, patch_count)
        group_count = min(self.num_groups, patch_count)

        extra_diagnostics: dict[str, float] = {}
        if self.method == "spatial":
            checked_coords = _validate_coords(coords, patch_count)
            partitions = _spatial_partitions(checked_coords, group_count)
            orders = _spatial_orders(checked_coords, partitions, group_count)
        elif self.method == "mmil_random":
            local_seed = _stable_seed(self.seed, slide_id, task, patch_count)
            partitions, orders = _random_partitions(
                patch_count, group_count, local_seed
            )
            extra_diagnostics["pseudobag/local_seed"] = float(local_seed)
        else:
            partitions, orders, kmeans_diagnostics = _embedding_partitions(
                features,
                group_count,
                self.kmeans_steps,
                self.cluster_dim,
            )
            extra_diagnostics.update(kmeans_diagnostics)

        indices, selected_partitions, counts = _balanced_selection(
            partitions, orders, group_count, selected_budget
        )
        if indices.numel() != selected_budget:
            raise RuntimeError(
                f"Selector returned {indices.numel()} patches for budget "
                f"{selected_budget}"
            )
        if torch.unique(indices).numel() != selected_budget:
            raise RuntimeError("Pseudo-bag selector returned duplicate indices")

        nonempty = int((torch.bincount(partitions, minlength=group_count) > 0).sum())
        selected_nonempty = counts[counts > 0]
        diagnostics = {
            "pseudobag/requested_patch_count": float(requested_budget),
            "pseudobag/selected_patch_count": float(selected_budget),
            "pseudobag/partition_count": float(group_count),
            "pseudobag/nonempty_partition_count": float(nonempty),
            "pseudobag/partition_coverage": float(
                (counts > 0).sum().item() / max(nonempty, 1)
            ),
            "pseudobag/min_selected_per_partition": float(
                selected_nonempty.min().item()
            ),
            "pseudobag/max_selected_per_partition": float(
                selected_nonempty.max().item()
            ),
        }
        diagnostics.update(extra_diagnostics)
        return PseudoBagSelectionResult(indices, selected_partitions, diagnostics)


def _validate_features(features: torch.Tensor) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2:
        raise ValueError(f"Expected features [patches, dims], got {tuple(features.shape)}")
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must contain at least one patch and one dimension")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must contain only finite values")


def _validate_coords(coords: torch.Tensor | None, patch_count: int) -> torch.Tensor:
    if coords is None:
        raise ValueError("spatial selection requires patch coordinates")
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be a torch.Tensor")
    if coords.ndim != 2 or coords.shape[0] != patch_count or coords.shape[1] < 2:
        raise ValueError(
            f"Expected coords [{patch_count}, >=2], got {tuple(coords.shape)}"
        )
    xy = coords[:, :2]
    if not torch.isfinite(xy).all().item():
        raise ValueError("coords must contain only finite values")
    return xy.detach().to(device="cpu", dtype=torch.float64)


def _stable_seed(seed: int, slide_id: str | None, task: int | str | None,
                 patch_count: int) -> int:
    payload = f"{seed}|{slide_id or ''}|{task if task is not None else ''}|{patch_count}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _random_partitions(
    patch_count: int, group_count: int, seed: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(patch_count, generator=generator)
    partitions = torch.empty(patch_count, dtype=torch.long)
    partitions[permutation] = torch.arange(patch_count) % group_count
    orders = [permutation[group::group_count] for group in range(group_count)]
    return partitions, orders


def _spatial_partitions(coords: torch.Tensor, group_count: int) -> torch.Tensor:
    """Recursively bisect the widest spatial region into balanced regions."""

    groups = [torch.arange(coords.shape[0], dtype=torch.long)]
    while len(groups) < group_count:
        candidates = []
        for group_id, indices in enumerate(groups):
            if indices.numel() < 2:
                continue
            values = coords[indices]
            spans = values.max(dim=0).values - values.min(dim=0).values
            dimension = int(torch.argmax(spans).item())
            candidates.append((float(spans[dimension]), int(indices.numel()), -group_id,
                               group_id, dimension))
        if not candidates:
            break
        _, _, _, split_group, dimension = max(candidates)
        indices = groups.pop(split_group)
        order = torch.argsort(coords[indices, dimension], stable=True)
        ordered = indices[order]
        midpoint = (ordered.numel() + 1) // 2
        groups.insert(split_group, ordered[:midpoint])
        groups.insert(split_group + 1, ordered[midpoint:])

    partitions = torch.empty(coords.shape[0], dtype=torch.long)
    for group_id, indices in enumerate(groups):
        partitions[indices] = group_id
    return partitions


def _spatial_orders(
    coords: torch.Tensor, partitions: torch.Tensor, group_count: int
) -> list[torch.Tensor]:
    orders = []
    for group_id in range(group_count):
        indices = torch.nonzero(partitions == group_id, as_tuple=False).flatten()
        if indices.numel() < 2:
            orders.append(indices)
            continue
        centered = coords[indices] - coords[indices].mean(dim=0, keepdim=True)
        radius = centered.square().sum(dim=1)
        orders.append(indices[torch.argsort(radius, stable=True)])
    return orders


def _embedding_partitions(
    features: torch.Tensor,
    group_count: int,
    steps: int,
    cluster_dim: int,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    work = features.detach().to(dtype=torch.float32)
    if work.shape[1] > cluster_dim:
        positions = torch.linspace(
            0, work.shape[1] - 1, steps=cluster_dim, device=work.device
        ).round().long()
        work = work[:, positions]
    scale = work.norm(dim=1, keepdim=True).clamp_min(torch.finfo(work.dtype).eps)
    work = work / scale

    centroids = _farthest_point_initialization(work, group_count)
    assignments = torch.full(
        (work.shape[0],), -1, dtype=torch.long, device=work.device
    )
    iterations = 0
    for iterations in range(1, steps + 1):
        distances = torch.cdist(work, centroids).square()
        updated = torch.argmin(distances, dim=1)
        if torch.equal(updated, assignments):
            break
        assignments = updated
        next_centroids = []
        for group_id in range(group_count):
            members = work[assignments == group_id]
            next_centroids.append(
                members.mean(dim=0) if members.numel() else centroids[group_id]
            )
        centroids = torch.stack(next_centroids)

    distances = torch.cdist(work, centroids).square()
    assignments = torch.argmin(distances, dim=1)
    minimum_distances = distances.gather(1, assignments[:, None]).squeeze(1)
    partitions = assignments.detach().cpu()
    orders = []
    for group_id in range(group_count):
        device_indices = torch.nonzero(assignments == group_id, as_tuple=False).flatten()
        member_distances = minimum_distances[device_indices]
        order = torch.argsort(member_distances, stable=True)
        orders.append(device_indices[order].detach().cpu())

    inertia = minimum_distances.mean().item()
    diagnostics = {
        "pseudobag/kmeans_iterations": float(iterations),
        "pseudobag/kmeans_inertia": float(inertia),
        "pseudobag/kmeans_feature_dim": float(work.shape[1]),
    }
    return partitions, orders, diagnostics


def _farthest_point_initialization(
    features: torch.Tensor, group_count: int
) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    mean_distances = (features - mean).square().sum(dim=1)
    first = int(torch.argmin(mean_distances).item())
    selected = [first]
    minimum = (features - features[first]).square().sum(dim=1)
    for _ in range(1, group_count):
        minimum[selected] = -1
        next_index = int(torch.argmax(minimum).item())
        selected.append(next_index)
        distance = (features - features[next_index]).square().sum(dim=1)
        minimum = torch.minimum(minimum, distance)
    return features[torch.tensor(selected, device=features.device)]


def _balanced_selection(
    partitions: torch.Tensor,
    orders: list[torch.Tensor],
    group_count: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capacities = torch.bincount(partitions, minlength=group_count)
    counts = torch.zeros(group_count, dtype=torch.long)
    for _ in range(budget):
        available = torch.nonzero(counts < capacities, as_tuple=False).flatten()
        if available.numel() == 0:
            raise RuntimeError("Pseudo-bag partitions cannot satisfy selection budget")
        minimum = counts[available].min()
        group_id = int(available[counts[available] == minimum][0].item())
        counts[group_id] += 1

    selected_indices = []
    selected_partitions = []
    for group_id, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        if count == 0:
            continue
        order = orders[group_id]
        positions = _even_positions(order.numel(), count)
        chosen = order[positions]
        selected_indices.append(chosen)
        selected_partitions.append(torch.full((count,), group_id, dtype=torch.long))
    indices = torch.cat(selected_indices)
    partition_ids = torch.cat(selected_partitions)
    return indices, partition_ids, counts


def _even_positions(length: int, count: int) -> torch.Tensor:
    if not 0 < count <= length:
        raise ValueError(f"Cannot choose {count} positions from sequence of length {length}")
    if count == length:
        return torch.arange(length, dtype=torch.long)
    positions = torch.floor(
        (torch.arange(count, dtype=torch.float64) + 0.5) * length / count
    ).long()
    if torch.unique(positions).numel() != count:
        raise RuntimeError("Even-position selection produced duplicate positions")
    return positions


def build_mmil_pseudobag_selector(
    method: str,
    *,
    num_groups: int = 4,
    seed: int = 0,
    kmeans_steps: int = 8,
    cluster_dim: int = 64,
) -> MMILPseudoBagSelector:
    """Build an independently testable pseudo-bag selector."""

    return MMILPseudoBagSelector(
        method,
        num_groups=num_groups,
        seed=seed,
        kmeans_steps=kmeans_steps,
        cluster_dim=cluster_dim,
    )
