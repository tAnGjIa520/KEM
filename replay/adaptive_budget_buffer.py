"""Adaptive WSI admission and nested patch allocation under one global cap."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import torch

from .global_budget_buffer import AdmissionCandidate, GlobalPatchBudgetBuffer


class AdaptiveGlobalPatchBuffer(GlobalPatchBudgetBuffer):
    """Allocate a fixed patch budget across classes, slides and depth levels.

    New-task WSI expose a monotone utility curve over nested patch levels.
    Previously stored WSI are atomic: they can be retained or evicted, but
    never expanded by re-reading old-task features.
    """

    def __init__(self, patch_budget: int, seed: int, fold: int, levels=(64, 128, 256, 512),
                 tolerance: float = 0.001, device: str = "cpu"):
        super().__init__(
            ratio=1.0, seed=seed, fold=fold, floor=min(levels), tolerance=tolerance,
            device=device, allow_capacity_limited=True,
        )
        self.patch_budget = int(patch_budget)
        self.levels = tuple(sorted({int(level) for level in levels}))
        if self.patch_budget < 1 or not self.levels or self.levels[0] < 1:
            raise ValueError("adaptive patch budget and levels must be positive")

    def existing_candidates(self):
        result = []
        for sample in self._all_samples():
            patches = int(sample["features"].shape[0])
            utility = float(torch.as_tensor(sample.get("allocation_utility", 1.0)).item())
            signature = torch.as_tensor(sample.get("bag_signature", torch.empty(0))).float().cpu()
            result.append(AdmissionCandidate(
                slide_id=self._sample_slide_id(sample),
                label=int(torch.as_tensor(sample["label"]).item()),
                original_patch_count=int(torch.as_tensor(sample["original_patch_count"]).item()),
                patch_budget=patches,
                source_task=int(torch.as_tensor(sample.get("source_task", -1)).item()),
                is_existing=True,
                allocation_levels=(patches,), allocation_utilities=(utility,),
                bag_signature=tuple(signature.tolist()),
            ))
        return result

    @staticmethod
    def _options(candidate: AdmissionCandidate):
        if candidate.is_existing:
            return ((int(candidate.patch_budget), float(candidate.allocation_utilities[0] if candidate.allocation_utilities else 1.0)),)
        levels = candidate.allocation_levels or (candidate.patch_budget,)
        utilities = candidate.allocation_utilities or tuple(float(index + 1) for index in range(len(levels)))
        if len(levels) != len(utilities):
            raise ValueError(f"allocation curve length mismatch for {candidate.slide_id}")
        pairs = []
        previous_level, previous_utility = 0, float("-inf")
        for level, utility in zip(levels, utilities):
            level = min(int(level), int(candidate.original_patch_count))
            utility = float(utility)
            if level <= previous_level or utility < previous_utility:
                raise ValueError(f"allocation curve must be strictly nested and monotone for {candidate.slide_id}")
            pairs.append((level, utility))
            previous_level, previous_utility = level, utility
        return tuple(pairs)

    def _allocate(self, candidates, cap):
        options = {item.slide_id: self._options(item) for item in candidates}
        by_id = {item.slide_id: item for item in candidates}
        stage = {item.slide_id: -1 for item in candidates}
        allocation, utility = {}, {}
        used = 0
        while True:
            best = None
            for slide_id, curve in options.items():
                next_stage = stage[slide_id] + 1
                if next_stage >= len(curve):
                    continue
                previous_cost = 0 if stage[slide_id] < 0 else curve[stage[slide_id]][0]
                previous_utility = 0.0 if stage[slide_id] < 0 else curve[stage[slide_id]][1]
                cost, value = curve[next_stage]
                delta_cost, delta_value = cost - previous_cost, value - previous_utility
                if used + delta_cost > cap:
                    continue
                score = delta_value / max(delta_cost, 1)
                key = (score, delta_value, -delta_cost, slide_id)
                if best is None or key > best[0]:
                    best = (key, slide_id, next_stage, cost, value, delta_cost)
            if best is None:
                break
            _, slide_id, next_stage, cost, value, delta_cost = best
            stage[slide_id] = next_stage
            allocation[slide_id], utility[slide_id] = cost, value
            used += delta_cost
        selected = [replace(
            by_id[slide_id], patch_budget=allocation[slide_id],
            allocation_levels=(allocation[slide_id],),
            allocation_utilities=(utility[slide_id],),
        ) for slide_id in allocation]
        return selected, used, utility

    def plan(self, new_candidates, target_budget: int | None = None, task: int = 0):
        target = self.patch_budget if target_budget is None else int(target_budget)
        candidates = self.existing_candidates() + list(new_candidates)
        slide_ids = [item.slide_id for item in candidates]
        duplicates = [key for key, count in Counter(slide_ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate slide IDs in admission pool: {sorted(duplicates)[:5]}")
        classes = sorted({item.label for item in candidates})
        if not classes:
            raise ValueError("empty admission pool")
        base, remainder = divmod(target, len(classes))
        class_caps = {label: base + (offset < remainder) for offset, label in enumerate(classes)}

        selected, class_used, utility = [], {}, {}
        for label in classes:
            chosen, used, values = self._allocate(
                [item for item in candidates if item.label == label], class_caps[label]
            )
            selected.extend(chosen)
            class_used[label] = used
            utility.update(values)

        # Spend class-quota remainder globally using the same nested curves.
        selected_by_id = {item.slide_id: item for item in selected}
        used = sum(item.patch_budget for item in selected)
        by_id = {item.slide_id: item for item in candidates}
        while used < target:
            best = None
            for slide_id, item in by_id.items():
                if item.is_existing and slide_id in selected_by_id:
                    continue
                curve = self._options(item)
                current_cost = selected_by_id[slide_id].patch_budget if slide_id in selected_by_id else 0
                current_utility = utility.get(slide_id, 0.0)
                next_options = [(level, value) for level, value in curve if level > current_cost]
                if not next_options:
                    continue
                next_cost, next_utility = next_options[0]
                delta_cost, delta_utility = next_cost - current_cost, next_utility - current_utility
                if used + delta_cost > target:
                    continue
                key = (delta_utility / max(delta_cost, 1), delta_utility, -delta_cost, slide_id)
                if best is None or key > best[0]:
                    best = (key, slide_id, next_cost, next_utility, delta_cost)
            if best is None:
                break
            _, slide_id, next_cost, next_utility, delta_cost = best
            item = by_id[slide_id]
            selected_by_id[slide_id] = replace(
                item, patch_budget=next_cost,
                allocation_levels=(next_cost,), allocation_utilities=(next_utility,),
            )
            utility[slide_id] = next_utility
            used += delta_cost

        # Discrete levels can leave a sub-level gap.  Fill it on newly admitted
        # WSI only; old WSI remain immutable and no old data are re-read.
        gap = target - used
        expandable = sorted(
            [item for item in selected_by_id.values() if not item.is_existing and item.patch_budget < min(item.original_patch_count, self.levels[-1])],
            key=lambda item: (utility[item.slide_id] / max(item.patch_budget, 1), item.slide_id),
            reverse=True,
        )
        for item in expandable:
            if gap <= 0:
                break
            capacity = min(item.original_patch_count, self.levels[-1]) - item.patch_budget
            addition = min(gap, capacity)
            selected_by_id[item.slide_id] = replace(item, patch_budget=item.patch_budget + addition)
            used += addition
            gap -= addition

        selected = list(selected_by_id.values())
        selected_ids = set(selected_by_id)
        before_ids = {self._sample_slide_id(sample) for sample in self._all_samples()}
        retained_ids = before_ids & selected_ids
        evicted_ids = before_ids - selected_ids
        self.buffers = {
            label: [sample for sample in self.buffers.get(label, []) if self._sample_slide_id(sample) in retained_ids]
            for label in classes
        }
        self.buffers = {label: samples for label, samples in self.buffers.items() if samples}
        self._refresh()
        admitted = [item for item in selected if not item.is_existing]
        error = (target - used) / target
        if error > self.tolerance:
            raise RuntimeError(
                f"adaptive allocator misses patch budget: target={target}, used={used}, error={error:.4%}"
            )
        self.last_plan = {
            "fold": self.fold, "task": int(task), "mode": "adaptive_global_patch",
            "target_budget": target, "target_original": target, "target_effective": target,
            "planned_budget": used, "budget_error": error,
            "class_caps": class_caps, "class_planned_patches": class_used,
            "selected_slide_ids": sorted(selected_ids),
            "retained_slide_ids": sorted(retained_ids), "evicted_slide_ids": sorted(evicted_ids),
            "admitted_slide_ids": sorted(item.slide_id for item in admitted),
            "selected_slides": len(selected), "retained_slides": len(retained_ids),
            "evicted_slides": len(evicted_ids), "admitted_slides": len(admitted),
            "allocation_histogram": dict(Counter(item.patch_budget for item in selected)),
            "allocation_utilities": {key: float(value) for key, value in utility.items()},
            "capacity_limited": False, "original_underfill_ratio": error,
            "actual_budget_before_add": self.n_patches_total,
        }
        self.plan_history.append(self.last_plan)
        return admitted, self.last_plan

    def state_dict(self):
        state = super().state_dict()
        state.update({"patch_budget": self.patch_budget, "levels": self.levels})
        return state

    @classmethod
    def from_state_dict(cls, state):
        result = cls(
            state["patch_budget"], state["seed"], state["fold"], state["levels"],
            state["tolerance"],
        )
        result.buffers = state["buffers"]
        result.plan_history = state.get("plan_history", [])
        result.last_plan = state.get("last_plan", {})
        result._refresh()
        return result
