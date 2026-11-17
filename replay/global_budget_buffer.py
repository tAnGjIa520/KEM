"""Deterministic whole-WSI admission under a fixed global patch budget."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


def move_data_to_device(value, device):
    """Small recursive tensor mover without importing the full trainer stack."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_data_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_data_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_data_to_device(item, device) for item in value)
    return value


def ratio_patch_budget(original_patch_count: int, ratio: float, floor: int = 64) -> int:
    """Whole-slide budget: small bags are stored completely; other bags use a floor."""
    count = int(original_patch_count)
    if count <= 0:
        raise ValueError("original_patch_count must be positive")
    if not 0 < float(ratio) <= 1:
        raise ValueError("ratio must be in (0, 1]")
    if count < int(floor):
        return count
    return min(count, max(int(floor), math.floor(float(ratio) * count)))


def load_patch_budget(path, fold: int, task: int) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "kem-global-patch-budget-v1":
        raise ValueError("unsupported global patch-budget schema")
    try:
        value = int(payload["targets"][str(int(fold))][str(int(task))])
    except KeyError as exc:
        raise KeyError(f"missing patch budget for fold={fold}, task={task}") from exc
    if value <= 0:
        raise ValueError("patch budget must be positive")
    return value


def deterministic_priority(seed: int, fold: int, task: int, slide_id: str) -> int:
    text = f"{int(seed)}|{int(fold)}|{int(task)}|{slide_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class AdmissionCandidate:
    slide_id: str
    label: int
    original_patch_count: int
    patch_budget: int
    source_task: int
    dataset_index: int | None = None
    is_existing: bool = False
    allocation_levels: tuple[int, ...] = ()
    allocation_utilities: tuple[float, ...] = ()
    bag_signature: tuple[float, ...] = ()


def candidate_replay_metadata(candidate: AdmissionCandidate | None, device):
    """Materialize optional adaptive-allocation metadata for a selected WSI.

    Ratio-budget baselines intentionally have no allocation curve or bag
    signature. Adaptive candidates carry both. Optional fields therefore stay
    absent instead of receiving a synthetic utility.
    """
    if candidate is None:
        return {}
    metadata = {}
    if candidate.allocation_utilities:
        metadata["allocation_utility"] = torch.tensor(
            candidate.allocation_utilities[-1], device=device,
        )
    if candidate.bag_signature:
        metadata["bag_signature"] = torch.tensor(candidate.bag_signature, device=device)
    return metadata


class GlobalPatchBudgetBuffer:
    """Class-balanced buffer that admits/evicts complete ratio-defined WSI subsets."""

    def __init__(self, ratio: float, seed: int, fold: int, floor: int = 64,
                 tolerance: float = 0.015, device: str = "cpu",
                 allow_capacity_limited: bool = False):
        self.ratio = float(ratio)
        self.seed = int(seed)
        self.fold = int(fold)
        self.floor = int(floor)
        self.tolerance = float(tolerance)
        self.device = device
        self.allow_capacity_limited = bool(allow_capacity_limited)
        self.buffers = {}
        self.labels = {}
        self.n_patches_total = 0
        self.num_classes = 0
        self.buffer_size = 0
        self.buffer_size_per_class = None
        self.n_seen_samples_per_class = defaultdict(int)
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.all_samples = []
        self.last_plan = {}
        self.plan_history = []

    @staticmethod
    def _sample_slide_id(sample):
        value = sample.get("slide_id")
        if isinstance(value, (list, tuple)):
            value = value[0]
        return str(value)

    def _all_samples(self):
        for label in sorted(self.buffers):
            yield from self.buffers[label]

    def to(self, device):
        if self.device != device:
            for label, samples in self.buffers.items():
                self.buffers[label] = [move_data_to_device(sample, device) for sample in samples]
        self.device = device
        return self

    def existing_candidates(self):
        result = []
        for sample in self._all_samples():
            original = int(torch.as_tensor(sample["original_patch_count"]).item())
            result.append(AdmissionCandidate(
                slide_id=self._sample_slide_id(sample),
                label=int(torch.as_tensor(sample["label"]).item()),
                original_patch_count=original,
                patch_budget=int(sample["features"].shape[0]),
                source_task=int(torch.as_tensor(sample.get("source_task", -1)).item()),
                is_existing=True,
            ))
        return result

    def _select_for_class(self, candidates, cap, task):
        ordered = sorted(
            candidates,
            key=lambda item: (deterministic_priority(self.seed, self.fold, task, item.slide_id), item.slide_id),
        )
        chosen, skipped, used = [], [], 0
        # Deterministic priority pass: admit every WSI that still fits the class quota.
        for item in ordered:
            if used + item.patch_budget <= cap:
                chosen.append(item); used += item.patch_budget
            else:
                skipped.append(item)
        # Best-fit completion. This normally leaves less than the 64-patch floor unused.
        while skipped:
            gap = cap - used
            feasible = [item for item in skipped if item.patch_budget <= gap]
            if not feasible:
                break
            item = min(feasible, key=lambda value: (
                gap - value.patch_budget,
                deterministic_priority(self.seed, self.fold, task, value.slide_id),
                value.slide_id,
            ))
            chosen.append(item); used += item.patch_budget; skipped.remove(item)
        return chosen, used

    def plan(self, new_candidates, target_budget: int, task: int):
        target_original = int(target_budget)
        if target_original <= 0:
            raise ValueError("target_budget must be positive")
        candidates = self.existing_candidates() + list(new_candidates)
        slide_ids = [item.slide_id for item in candidates]
        if len(slide_ids) != len(set(slide_ids)):
            duplicates = sorted(key for key, count in Counter(slide_ids).items() if count > 1)
            raise ValueError(f"duplicate slide IDs in admission pool: {duplicates[:5]}")
        label_by_slide = {item.slide_id: item.label for item in candidates}
        if len(label_by_slide) != len(candidates):
            raise ValueError("slide label conflict in admission pool")
        classes = sorted({item.label for item in candidates})
        if not classes:
            raise ValueError("empty admission pool")
        capacity = sum(item.patch_budget for item in candidates)
        capacity_limited = capacity < target_original
        if capacity_limited and not self.allow_capacity_limited:
            raise RuntimeError(
                f"cannot fit whole WSI budgets within tolerance: target={target_original}, "
                f"capacity={capacity}, error={(target_original-capacity)/target_original:.4%}"
            )
        target_effective = min(target_original, capacity)
        base, remainder = divmod(target_effective, len(classes))
        class_caps = {label: base + (1 if offset < remainder else 0)
                      for offset, label in enumerate(classes)}
        selected, class_used = [], {}
        for label in classes:
            pool = [item for item in candidates if item.label == label]
            chosen, used = self._select_for_class(pool, class_caps[label], task)
            selected.extend(chosen); class_used[label] = used

        # Use any globally remaining budget without breaking deterministic ordering.
        selected_ids = {item.slide_id for item in selected}
        remaining = [item for item in candidates if item.slide_id not in selected_ids]
        used = sum(item.patch_budget for item in selected)
        while remaining:
            gap = target_effective - used
            feasible = [item for item in remaining if item.patch_budget <= gap]
            if not feasible:
                break
            item = min(feasible, key=lambda value: (
                gap - value.patch_budget,
                class_used.get(value.label, 0),
                deterministic_priority(self.seed, self.fold, task, value.slide_id),
                value.slide_id,
            ))
            selected.append(item); selected_ids.add(item.slide_id)
            used += item.patch_budget; class_used[item.label] = class_used.get(item.label, 0) + item.patch_budget
            remaining.remove(item)

        # Whole-WSI ratio costs are discrete and can leave a sub-floor gap.
        # Close it exactly by expanding only WSI admitted from the current
        # task. Old replay WSI remain immutable and are never re-read. Assign
        # one patch at a time to the least-used class for deterministic class
        # balance; the gap is normally smaller than the ratio floor.
        gap_fill = defaultdict(int)
        selected_by_id = {item.slide_id: item for item in selected}
        while used < target_effective:
            expandable = [
                item for item in selected_by_id.values()
                if not item.is_existing and item.patch_budget < item.original_patch_count
            ]
            if not expandable:
                raise RuntimeError(
                    "cannot close exact global patch budget without expanding old WSI: "
                    f"target={target_effective}, used={used}"
                )
            item = min(expandable, key=lambda value: (
                class_used.get(value.label, 0),
                deterministic_priority(self.seed, self.fold, task, value.slide_id),
                value.slide_id,
            ))
            expanded = replace(item, patch_budget=item.patch_budget + 1)
            selected_by_id[item.slide_id] = expanded
            gap_fill[item.slide_id] += 1
            class_used[item.label] = class_used.get(item.label, 0) + 1
            used += 1
        selected = [selected_by_id[item.slide_id] for item in selected]

        error = (target_effective - used) / target_effective
        if used > target_effective:
            raise AssertionError("global patch budget exceeded")
        if error > self.tolerance:
            raise RuntimeError(
                f"cannot fit whole WSI budgets within tolerance: target={target_effective}, "
                f"used={used}, error={error:.4%}, tolerance={self.tolerance:.4%}"
            )
        before_ids = {self._sample_slide_id(sample) for sample in self._all_samples()}
        retained_ids = before_ids & selected_ids
        evicted_ids = before_ids - selected_ids
        admitted = [item for item in selected if not item.is_existing]
        self.buffers = {
            label: [sample for sample in self.buffers.get(label, [])
                    if self._sample_slide_id(sample) in retained_ids]
            for label in classes
        }
        self.buffers = {label: samples for label, samples in self.buffers.items() if samples}
        self._refresh()
        plan = {
            "fold": self.fold, "task": int(task), "ratio": self.ratio,
            "target_budget": target_original,
            "target_original": target_original, "target_effective": target_effective,
            "available_unique_wsi_capacity": capacity,
            "capacity_limited": capacity_limited,
            "original_underfill_abs": target_original - used,
            "original_underfill_ratio": (target_original - used) / target_original,
            "actual_budget_before_add": self.n_patches_total,
            "planned_budget": used, "budget_error": error,
            "class_caps": class_caps, "class_planned_patches": class_used,
            "selected_slide_ids": sorted(selected_ids),
            "retained_slide_ids": sorted(retained_ids), "evicted_slide_ids": sorted(evicted_ids),
            "admitted_slide_ids": [item.slide_id for item in admitted],
            "ratio_gap_fill_patches": int(sum(gap_fill.values())),
            "ratio_gap_fill_by_slide": dict(sorted(gap_fill.items())),
            "selected_slides": len(selected), "retained_slides": len(retained_ids),
            "evicted_slides": len(evicted_ids), "admitted_slides": len(admitted),
        }
        self.last_plan = plan
        self.plan_history.append(plan)
        return admitted, plan

    def add(self, sample):
        sample = move_data_to_device(sample, self.device)
        slide_id = self._sample_slide_id(sample)
        if any(self._sample_slide_id(old) == slide_id for old in self._all_samples()):
            raise ValueError(f"duplicate buffer slide: {slide_id}")
        label = int(torch.as_tensor(sample["label"]).item())
        self.buffers.setdefault(label, []).append(sample)
        self._refresh()

    def _refresh(self):
        self.labels = {int(label): len(samples) for label, samples in self.buffers.items()}
        self.n_patches_total = sum(int(sample["features"].shape[0]) for sample in self._all_samples())
        self.num_classes = len(self.buffers)
        self.buffer_size = len(self)
        self.buffer_size_per_class = None
        self.epoch_indices = None

    def finalize_plan(self):
        if not self.last_plan:
            return
        self.last_plan["actual_budget"] = self.n_patches_total
        self.last_plan["actual_budget_error"] = (
            self.last_plan["target_effective"] - self.n_patches_total
        ) / self.last_plan["target_effective"]
        self.last_plan["original_underfill_abs"] = self.last_plan["target_original"] - self.n_patches_total
        self.last_plan["original_underfill_ratio"] = (
            self.last_plan["target_original"] - self.n_patches_total
        ) / self.last_plan["target_original"]
        if self.n_patches_total > self.last_plan["target_original"]:
            raise AssertionError("final buffer exceeds global patch budget")
        if self.last_plan["actual_budget_error"] > self.tolerance:
            raise AssertionError("final buffer misses global patch budget tolerance")
        self.last_plan["per_class_slides"] = dict(self.labels)
        self.last_plan["per_class_patches"] = {
            int(label): sum(int(sample["features"].shape[0]) for sample in samples)
            for label, samples in self.buffers.items()
        }
        self.last_plan["floor_hits"] = sum(
            int(torch.as_tensor(sample.get("ratio_floor_hit", 0)).item()) for sample in self._all_samples()
        )
        self.last_plan["small_bags"] = sum(
            int(torch.as_tensor(sample.get("ratio_small_bag", 0)).item()) for sample in self._all_samples()
        )

    def start_epoch(self):
        self.all_samples = list(self._all_samples())
        rng = np.random.default_rng(self.seed + 1000003 * self.fold + len(self.plan_history))
        self.epoch_indices = rng.permutation(len(self.all_samples))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.all_samples) - self.current_epoch_position
        if remaining <= 0:
            return None
        count = min(int(batch_size), remaining)
        indices = self.epoch_indices[self.current_epoch_position:self.current_epoch_position + count]
        batch = [self.all_samples[int(index)] for index in indices]
        self.current_epoch_position += count
        self.samples_output_in_epoch += count
        return batch

    def get_samples_output_count(self):
        return self.samples_output_in_epoch

    def state_dict(self):
        return {
            "ratio": self.ratio, "seed": self.seed, "fold": self.fold,
            "floor": self.floor, "tolerance": self.tolerance,
            "allow_capacity_limited": self.allow_capacity_limited,
            "buffers": self.buffers, "plan_history": self.plan_history,
            "last_plan": self.last_plan,
        }

    @classmethod
    def from_state_dict(cls, state):
        result = cls(
            state["ratio"], state["seed"], state["fold"], state["floor"],
            state["tolerance"], allow_capacity_limited=state.get("allow_capacity_limited", False),
        )
        result.buffers = state["buffers"]
        result.plan_history = state.get("plan_history", [])
        result.last_plan = state.get("last_plan", {})
        result._refresh()
        return result

    def __len__(self):
        return sum(len(samples) for samples in self.buffers.values())
