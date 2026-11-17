"""Online uniform-random full-WSI replay under a strict patch upper bound."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from lightning.fabric.utilities import move_data_to_device


def _priority(seed: int, fold: int, slide_id: str) -> int:
    payload = f"{int(seed)}|{int(fold)}|{slide_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class _Candidate:
    slide_id: str
    label: int
    patch_count: int
    source_task: int
    dataset_index: int | None
    sample: dict[str, Any] | None


class FullWSIPatchBudgetBuffer:
    """Keep complete, randomly prioritized WSIs without exceeding a patch cap.

    Stable random priorities make the online update equivalent to retaining a
    deterministic random prefix of all replay-eligible WSIs seen so far.  Patch
    counts are used only as a feasibility constraint; no patch is filtered and
    no bag is truncated to fill a residual gap.
    """

    def __init__(self, patch_budget: int, seed: int, fold: int, device: str = "cpu"):
        if int(patch_budget) < 1:
            raise ValueError("patch_budget must be positive")
        self.patch_budget = int(patch_budget)
        self.seed = int(seed)
        self.fold = int(fold)
        self.device = device
        self.buffer: list[dict[str, Any]] = []
        self.n_patches_total = 0
        self.labels: dict[int, int] = {}
        self.last_plan: dict[str, Any] = {}
        self.plan_history: list[dict[str, Any]] = []
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    @staticmethod
    def _slide_id(sample: dict[str, Any]) -> str:
        value = sample["slide_id"]
        return str(value[0] if isinstance(value, (list, tuple)) else value)

    @staticmethod
    def _label(sample: dict[str, Any]) -> int:
        return int(torch.as_tensor(sample["label"]).reshape(-1)[0].item())

    def to(self, device):
        if self.device != device:
            self.buffer = [move_data_to_device(sample, device) for sample in self.buffer]
            self.device = device
        return self

    def _refresh(self) -> None:
        self.n_patches_total = sum(int(sample["features"].shape[0]) for sample in self.buffer)
        self.labels = dict(sorted(Counter(self._label(sample) for sample in self.buffer).items()))
        self.epoch_indices = None

    def start_epoch(self) -> None:
        if not self.buffer:
            raise RuntimeError("cannot replay from an empty full-WSI buffer")
        self.epoch_indices = np.random.permutation(len(self.buffer))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.buffer) - self.current_epoch_position
        if remaining <= 0:
            return None
        count = min(int(batch_size), remaining)
        indices = self.epoch_indices[
            self.current_epoch_position:self.current_epoch_position + count
        ]
        self.current_epoch_position += count
        self.samples_output_in_epoch += count
        return [self.buffer[int(index)] for index in indices]

    def get_samples_output_count(self) -> int:
        return self.samples_output_in_epoch

    def __len__(self) -> int:
        return len(self.buffer)

    def update(
        self,
        dataset,
        task: int,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        existing = {
            self._slide_id(sample): _Candidate(
                self._slide_id(sample), self._label(sample),
                int(sample["features"].shape[0]),
                int(torch.as_tensor(sample.get("source_task", -1)).item()),
                None, sample,
            ) for sample in self.buffer
        }
        current: dict[str, _Candidate] = {}
        oversized = []
        for index in range(len(dataset)):
            raw = dataset[index]
            slide_id = self._slide_id(raw)
            if slide_id in existing or slide_id in current:
                raise RuntimeError(f"duplicate or historical WSI in task dataset: {slide_id}")
            patch_count = int(torch.as_tensor(raw.get("n_patch", raw["features"].shape[0])).item())
            if patch_count != int(raw["features"].shape[0]):
                raise ValueError(f"patch-count mismatch for {slide_id}")
            candidate = _Candidate(
                slide_id, self._label(raw), patch_count, int(task), int(index), None,
            )
            current[slide_id] = candidate
            if patch_count > self.patch_budget:
                oversized.append(slide_id)

        candidates = list(existing.values()) + list(current.values())
        ordered = sorted(
            candidates,
            key=lambda item: (_priority(self.seed, self.fold, item.slide_id), item.slide_id),
        )
        chosen, used = [], 0
        for candidate in ordered:
            if used + candidate.patch_count <= self.patch_budget:
                chosen.append(candidate)
                used += candidate.patch_count
        if not chosen:
            raise RuntimeError(
                f"no complete WSI fits strict patch budget {self.patch_budget}"
            )

        materialized = []
        for candidate in chosen:
            if candidate.sample is not None:
                sample = candidate.sample
            else:
                raw = dataset[int(candidate.dataset_index)]
                sample = {
                    "features": raw["features"],
                    "label": torch.tensor([candidate.label], dtype=torch.long),
                    "slide_id": [candidate.slide_id],
                    "n_patch": torch.tensor([candidate.patch_count], dtype=torch.long),
                    "original_patch_count": torch.tensor(candidate.patch_count),
                    "source_task": torch.tensor(candidate.source_task),
                    "selection_method": "uniform_random_full_wsi",
                }
                if transform is not None:
                    sample = transform(sample)
            if int(sample["features"].shape[0]) != candidate.patch_count:
                raise AssertionError("full-WSI replay must not filter or expand patches")
            materialized.append(move_data_to_device(sample, self.device))
        old_ids = set(existing)
        selected_ids = {item.slide_id for item in chosen}
        self.buffer = materialized
        self._refresh()
        if self.n_patches_total > self.patch_budget:
            raise AssertionError("full-WSI replay exceeded strict patch budget")
        plan = {
            "schema_version": "kem-strict-full-wsi-v1",
            "fold": self.fold,
            "task": int(task),
            "target_patches": self.patch_budget,
            "actual_patches": self.n_patches_total,
            "budget_underfill": self.patch_budget - self.n_patches_total,
            "selected_slide_ids": [item.slide_id for item in chosen],
            "stored_wsi_count": len(chosen),
            "retained_wsi_count": len(old_ids & selected_ids),
            "evicted_wsi_count": len(old_ids - selected_ids),
            "admitted_wsi_count": len(selected_ids - old_ids),
            "oversized_current_wsi_count": len(oversized),
            "oversized_current_slide_ids": sorted(oversized),
            "selection": "stable_uniform_random_full_wsi",
            "patch_filtering": False,
            "patch_expansion": False,
        }
        self.last_plan = plan
        self.plan_history.append(dict(plan))
        return plan

    def manifest_rows(self) -> list[dict[str, Any]]:
        return [{
            "slide_id": self._slide_id(sample),
            "label": self._label(sample),
            "source_task": int(torch.as_tensor(sample.get("source_task", -1)).item()),
            "original_patch_count": int(sample["features"].shape[0]),
            "selected_patch_count": int(sample["features"].shape[0]),
            "is_full_wsi": 1,
        } for sample in self.buffer]
