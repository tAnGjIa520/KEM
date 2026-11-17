"""Breakup--Reorganize rehearsal adapted from ConSlide for CLAM features.

This module implements the BuRo replay algorithm from Huang et al. (ICCV
2023): randomly retain spatial WSI regions and synthesize replay bags by
sampling regions with the same task and label.  It deliberately does *not*
claim to implement ConSlide's HIT backbone or its CSSL asynchronous update;
those require paired patch/region features that are absent from the frozen
UNI MIL benchmark.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import torch

from .region_supervision import RegionConfig, partition_regions


def _priority(seed: int, fold: int, record_id: str) -> int:
    payload = f"{seed}|{fold}|{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class RegionRecord:
    """One stored BuRo region and its provenance."""

    record_id: str
    source_slide_id: str
    source_task: int
    label: int
    region_index: int
    features: torch.Tensor
    selected_indices: torch.Tensor

    @property
    def patch_count(self) -> int:
        return int(self.features.shape[0])

    def truncate(self, count: int) -> "RegionRecord":
        count = int(count)
        if count < 1 or count > self.patch_count:
            raise ValueError("invalid region truncation")
        return replace(
            self,
            features=self.features[:count].clone(),
            selected_indices=self.selected_indices[:count].clone(),
        )


class ConSlideBuRoBuffer:
    """Patch-budgeted region reservoir with ConSlide reorganization.

    The original ConSlide buffer is measured in fixed hierarchical regions.
    This adaptation enforces the benchmark's stricter global patch upper
    bound.  Whole regions are retained whenever possible; at most one region
    is truncated to close a residual patch-budget gap exactly.
    """

    def __init__(
        self,
        patch_budget: int,
        seed: int,
        fold: int,
        *,
        region_size: int = 64,
        store_ratio: float = 0.2,
        replay_regions: int = 8,
        partition: str = "spatial",
    ) -> None:
        if int(patch_budget) < 1:
            raise ValueError("patch_budget must be positive")
        if int(region_size) < 1 or int(replay_regions) < 1:
            raise ValueError("region_size and replay_regions must be positive")
        if not 0 < float(store_ratio) <= 1:
            raise ValueError("store_ratio must be in (0, 1]")
        if partition != "spatial":
            raise ValueError(
                "formal ConSlide-BuRo requires spatial regions; "
                "non-spatial fallback is intentionally disabled"
            )
        self.patch_budget = int(patch_budget)
        self.seed = int(seed)
        self.fold = int(fold)
        self.region_size = int(region_size)
        self.store_ratio = float(store_ratio)
        self.replay_regions = int(replay_regions)
        self.partition = partition

        self.records: list[RegionRecord] = []
        self.buffers: dict[int, list[dict]] = {}
        self.labels: dict[int, int] = {}
        self.n_patches_total = 0
        self.buffer_size = 0
        self.buffer_size_per_class = None
        self.num_classes = 0
        self.n_seen_samples_per_class = defaultdict(int)
        self.ratio = self.store_ratio
        self.floor = 1
        self.tolerance = 0.0
        self.last_plan: dict = {}
        self.plan_history: list[dict] = []

        self._epoch = 0
        self._epoch_replay_updates = 0
        self.total_replay_updates = 0
        self._rng = np.random.default_rng(self.seed)
        self._group_order: list[tuple[int, int]] = []
        self._group_position = 0

    @staticmethod
    def _validate_features(features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features).detach().cpu()
        if features.ndim != 2 or min(features.shape) < 1:
            raise ValueError("ConSlide regions require non-empty [N,D] features")
        if not torch.isfinite(features).all().item():
            raise ValueError("ConSlide features contain NaN or Inf")
        return features

    def breakup(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        *,
        slide_id: str,
        label: int,
        task: int,
    ) -> list[RegionRecord]:
        """Spatially break one WSI and randomly retain 20% of regions."""

        features = self._validate_features(features)
        coords = torch.as_tensor(coords).detach().cpu()
        if coords.shape != (features.shape[0], 2):
            raise ValueError(
                f"coords shape {tuple(coords.shape)} does not match "
                f"features {(features.shape[0], 2)}"
            )
        config = RegionConfig(
            partition="spatial_balanced_100",
            target_patches=self.region_size,
            min_size=1,
            seed=self.seed,
        )
        regions = partition_regions(features, coords, config)
        keep = max(1, math.ceil(self.store_ratio * len(regions)))
        local_seed = _priority(self.seed, self.fold, f"{task}|{slide_id}") % (2**32)
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        chosen = torch.randperm(len(regions), generator=generator)[:keep].tolist()
        result = []
        for region_index in chosen:
            indices = regions[region_index].long().cpu()
            record_id = f"t{int(task)}:{slide_id}:r{int(region_index)}"
            result.append(RegionRecord(
                record_id=record_id,
                source_slide_id=str(slide_id),
                source_task=int(task),
                label=int(label),
                region_index=int(region_index),
                features=features.index_select(0, indices).clone(),
                selected_indices=indices.clone(),
            ))
        return result

    def _retain_under_budget(self, records: Iterable[RegionRecord]) -> list[RegionRecord]:
        unique = {}
        for record in records:
            if record.record_id in unique:
                raise ValueError(f"duplicate BuRo region: {record.record_id}")
            unique[record.record_id] = record
        # Keeping the globally smallest deterministic random priorities is
        # distributionally equivalent to reservoir sampling for a fixed pool,
        # while making task-boundary reconstruction exactly reproducible.
        ordered = sorted(
            unique.values(),
            key=lambda item: (_priority(self.seed, self.fold, item.record_id), item.record_id),
        )
        selected, used = [], 0
        for record in ordered:
            gap = self.patch_budget - used
            if gap <= 0:
                break
            if record.patch_count <= gap:
                selected.append(record)
                used += record.patch_count
            else:
                selected.append(record.truncate(gap))
                used += gap
                break
        return selected

    def add_task_dataset(
        self,
        dataset,
        task: int,
        coords_loader: Callable[[str, torch.Tensor], torch.Tensor],
        max_slides: int | None = None,
    ) -> dict:
        """Break up the current task, update the region reservoir, and audit it."""

        selection_started = time.perf_counter()
        previous_ids = {record.record_id for record in self.records}
        candidates = list(self.records)
        current_region_count = 0
        current_slide_count = 0
        current_patch_count = 0
        limit = len(dataset) if max_slides is None or int(max_slides) < 0 else min(
            len(dataset), int(max_slides)
        )
        if limit < 1:
            raise ValueError("ConSlide buffer update requires at least one WSI")
        for index in range(limit):
            sample = dataset[index]
            features = self._validate_features(sample["features"])
            current_patch_count += int(features.shape[0])
            slide_id = str(sample["slide_id"])
            label = int(torch.as_tensor(sample["label"]).item())
            coords = coords_loader(slide_id, features)
            regions = self.breakup(
                features, coords, slide_id=slide_id, label=label, task=int(task)
            )
            candidates.extend(regions)
            current_region_count += len(regions)
            current_slide_count += 1

        capacity = sum(record.patch_count for record in candidates)
        self.records = self._retain_under_budget(candidates)
        self._refresh()
        retained_ids = {record.record_id for record in self.records}
        source_slides = {record.source_slide_id for record in self.records}
        selection_seconds = time.perf_counter() - selection_started
        plan = {
            "schema_version": "kem-conslide-buro-buffer-v1",
            "method_label": "ConSlide-BuRo (CLAM adaptation)",
            "task": int(task),
            "target_patches": self.patch_budget,
            "available_patches": int(capacity),
            "actual_patches": self.n_patches_total,
            "budget_underfill": self.patch_budget - self.n_patches_total,
            "budget_exceeded": self.n_patches_total > self.patch_budget,
            "stored_regions": len(self.records),
            "stored_wsi_count": len(source_slides),
            "current_task_wsi_count": current_slide_count,
            "current_task_patch_count": current_patch_count,
            "current_task_regions_sampled": current_region_count,
            "retained_old_regions": len(previous_ids & retained_ids),
            "evicted_old_regions": len(previous_ids - retained_ids),
            "admitted_new_regions": len(retained_ids - previous_ids),
            "per_class_regions": dict(sorted(Counter(r.label for r in self.records).items())),
            "per_task_regions": dict(sorted(Counter(r.source_task for r in self.records).items())),
            "replay_updates_before_buffer_update": self.total_replay_updates,
            "selector_metrics": {
                "selector/final_selection_seconds": selection_seconds,
                "selector/analyzed_wsi": float(current_slide_count),
                "selector/analyzed_patch_count": float(current_patch_count),
            },
        }
        if self.n_patches_total > self.patch_budget:
            raise AssertionError("ConSlide BuRo exceeded the global patch budget")
        self.last_plan = plan
        self.plan_history.append(dict(plan))
        return plan

    @staticmethod
    def _record_sample(record: RegionRecord) -> dict:
        return {
            "features": record.features,
            "label": torch.tensor(record.label, dtype=torch.long),
            "slide_id": record.record_id,
            "source_slide_id": record.source_slide_id,
            "source_task": torch.tensor(record.source_task, dtype=torch.long),
            "region_index": torch.tensor(record.region_index, dtype=torch.long),
            "selected_indices": record.selected_indices,
            "original_patch_count": torch.tensor(record.patch_count, dtype=torch.long),
            "selection_method": "conslide_buro_random_region",
        }

    def _refresh(self) -> None:
        self.buffers = {}
        for record in self.records:
            self.buffers.setdefault(record.label, []).append(self._record_sample(record))
        self.labels = {label: len(samples) for label, samples in self.buffers.items()}
        self.n_patches_total = sum(record.patch_count for record in self.records)
        self.buffer_size = len(self.records)
        self.num_classes = len(self.buffers)

    def start_epoch(self) -> None:
        if not self.records:
            raise RuntimeError("cannot replay from an empty ConSlide buffer")
        groups = sorted({(record.source_task, record.label) for record in self.records})
        self._rng = np.random.default_rng(
            self.seed + 1_000_003 * self.fold + 10_007 * self._epoch
        )
        self._group_order = [groups[index] for index in self._rng.permutation(len(groups))]
        self._group_position = 0
        self._epoch_replay_updates = 0
        self._epoch += 1

    def reorganize(self, task: int, label: int) -> dict:
        """Generate one pseudo-WSI from regions sharing task and label."""

        pool = [
            record for record in self.records
            if record.source_task == int(task) and record.label == int(label)
        ]
        if not pool:
            raise KeyError(f"no regions for task={task}, label={label}")
        count = min(self.replay_regions, len(pool))
        chosen_indices = self._rng.choice(len(pool), size=count, replace=False)
        chosen = [pool[int(index)] for index in chosen_indices]
        features = torch.cat([record.features for record in chosen], dim=0)
        source_slides = [record.source_slide_id for record in chosen]
        return {
            "features": features,
            "label": torch.tensor([int(label)], dtype=torch.long),
            "slide_id": ["buro:" + "+".join(record.record_id for record in chosen)],
            "source_task": torch.tensor(int(task), dtype=torch.long),
            "source_slide_ids": source_slides,
            "replay_region_ids": [record.record_id for record in chosen],
            "replay_region_count": torch.tensor(count, dtype=torch.long),
            "replay_source_wsi_count": torch.tensor(len(set(source_slides)), dtype=torch.long),
            "selection_method": "conslide_buro_reorganize",
        }

    def get_next_batch(self, batch_size: int = 1):
        if int(batch_size) != 1:
            raise ValueError("ConSlide-BuRo currently emits one pseudo-WSI per replay update")
        if not self._group_order:
            self.start_epoch()
        key = self._group_order[self._group_position % len(self._group_order)]
        self._group_position += 1
        self._epoch_replay_updates += 1
        self.total_replay_updates += 1
        return [self.reorganize(*key)]

    def get_samples_output_count(self) -> int:
        return self._epoch_replay_updates

    def manifest_rows(self) -> list[dict]:
        return [{
            "record_id": record.record_id,
            "source_slide_id": record.source_slide_id,
            "source_task": record.source_task,
            "label": record.label,
            "region_index": record.region_index,
            "selected_patch_count": record.patch_count,
        } for record in self.records]

    def state_dict(self) -> dict:
        return {
            "schema_version": "kem-conslide-buro-buffer-v1",
            "patch_budget": self.patch_budget,
            "seed": self.seed,
            "fold": self.fold,
            "region_size": self.region_size,
            "store_ratio": self.store_ratio,
            "replay_regions": self.replay_regions,
            "partition": self.partition,
            "records": self.records,
            "last_plan": self.last_plan,
            "plan_history": self.plan_history,
            "total_replay_updates": self.total_replay_updates,
        }

    def __len__(self) -> int:
        return len({record.source_slide_id for record in self.records})
