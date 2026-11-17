"""Deterministic fixed-count replay schedules for protocol-controlled studies.

The natural KEM loop exposes every stored item roughly once per epoch.  That
couples optimization exposure to the number of retained pseudo-bags.  This
module instead materializes exactly ``updates`` replay draws with replacement:

* KEM draws are proportional to the allocator's assignment mass.
* Other replay buffers draw classes uniformly, then an item uniformly within
  the selected class.
* ConSlide-BuRo draws classes uniformly and reorganizes a compatible
  task/class region group.

The schedule is generated once per epoch, so it is auditable and independent
of current-task loader length.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch


def fixed_replay_target(batch_index: int, current_updates: int, replay_updates: int) -> int:
    """Cumulative replay target after a zero-based current update.

    The ceiling form spreads replay throughout the epoch and reaches the exact
    requested count after the final current update.
    """

    current_updates = int(current_updates)
    replay_updates = int(replay_updates)
    batch_index = int(batch_index)
    if current_updates < 1:
        raise ValueError("current_updates must be positive")
    if replay_updates < 0:
        raise ValueError("replay_updates must be non-negative")
    if not 0 <= batch_index < current_updates:
        raise ValueError("batch_index is outside the current-update range")
    return int(np.ceil((batch_index + 1) * replay_updates / current_updates))


def _label(sample: dict[str, Any]) -> int:
    return int(torch.as_tensor(sample["label"]).reshape(-1)[0].item())


def _buffer_samples(buffer) -> list[dict[str, Any]]:
    if hasattr(buffer, "_all_samples"):
        return list(buffer._all_samples())
    if hasattr(buffer, "buffers") and isinstance(buffer.buffers, dict):
        return [sample for label in sorted(buffer.buffers) for sample in buffer.buffers[label]]
    if hasattr(buffer, "buffer"):
        return list(buffer.buffer)
    raise TypeError(f"unsupported replay buffer type: {type(buffer).__name__}")


def _balanced_labels(labels: list[int], updates: int, rng: np.random.Generator) -> list[int]:
    """Return a shuffled, near-exact class-balanced label schedule."""

    if not labels:
        raise RuntimeError("cannot replay from an empty label set")
    labels = sorted(set(map(int, labels)))
    repeats, remainder = divmod(int(updates), len(labels))
    schedule = labels * repeats
    if remainder:
        schedule.extend(np.asarray(labels)[rng.permutation(len(labels))[:remainder]].tolist())
    if schedule:
        schedule = np.asarray(schedule, dtype=np.int64)[rng.permutation(len(schedule))].tolist()
    return list(map(int, schedule))


class FixedReplayEpoch:
    """One exact replacement-sampling schedule for a replay epoch."""

    def __init__(
        self,
        buffer,
        updates: int,
        *,
        seed: int,
        fold: int,
        task: int,
        epoch: int,
        assignment_weighted: bool = False,
    ) -> None:
        self.buffer = buffer
        self.updates = int(updates)
        if self.updates < 0:
            raise ValueError("updates must be non-negative")
        mixed_seed = (
            int(seed) + 1_000_003 * int(fold) + 10_007 * int(task)
            + 101 * int(epoch)
        ) % (2**32)
        self.rng = np.random.default_rng(mixed_seed)
        self.position = 0
        self.mode = "assignment_mass" if assignment_weighted else "class_balanced_uniform"
        self.sampled_labels: list[int] = []
        self.sampled_slide_ids: list[str] = []
        self._is_conslide = hasattr(buffer, "reorganize") and hasattr(buffer, "records")
        self._draws: list[Any] = []
        if self.updates == 0:
            return
        if self._is_conslide:
            self._build_conslide_draws()
        elif assignment_weighted:
            self._build_assignment_draws()
        else:
            self._build_balanced_draws()
        if len(self._draws) != self.updates:
            raise AssertionError("fixed replay schedule has the wrong length")

    def _build_assignment_draws(self) -> None:
        samples = _buffer_samples(self.buffer)
        if not samples:
            raise RuntimeError("cannot replay from an empty buffer")
        weights = np.asarray([
            float(torch.as_tensor(sample.get("unified_assignment_mass", 0.0)).item())
            for sample in samples
        ], dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise RuntimeError(
                "KEM fixed replay requires positive finite assignment masses"
            )
        indices = self.rng.choice(
            len(samples), size=self.updates, replace=True, p=weights / weights.sum()
        )
        self._draws = [samples[int(index)] for index in indices]

    def _build_balanced_draws(self) -> None:
        samples = _buffer_samples(self.buffer)
        if not samples:
            raise RuntimeError("cannot replay from an empty buffer")
        by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            by_label[_label(sample)].append(sample)
        labels = _balanced_labels(list(by_label), self.updates, self.rng)
        self._draws = [
            by_label[label][int(self.rng.integers(len(by_label[label])))]
            for label in labels
        ]

    def _build_conslide_draws(self) -> None:
        groups_by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for record in self.buffer.records:
            key = (int(record.source_task), int(record.label))
            if key not in groups_by_label[key[1]]:
                groups_by_label[key[1]].append(key)
        labels = _balanced_labels(list(groups_by_label), self.updates, self.rng)
        self._draws = [
            groups_by_label[label][int(self.rng.integers(len(groups_by_label[label])))]
            for label in labels
        ]
        # Reorganization uses the buffer RNG for region sampling.
        self.buffer._rng = self.rng

    @staticmethod
    def _slide_id(sample: dict[str, Any]) -> str:
        value = sample.get("slide_id", "")
        if isinstance(value, (list, tuple)):
            value = value[0]
        return str(value)

    def next_batch(self) -> list[dict[str, Any]]:
        if self.position >= self.updates:
            raise StopIteration("fixed replay schedule is exhausted")
        draw = self._draws[self.position]
        self.position += 1
        if self._is_conslide:
            sample = self.buffer.reorganize(*draw)
            self.buffer.total_replay_updates += 1
        else:
            sample = draw
        self.sampled_labels.append(_label(sample))
        self.sampled_slide_ids.append(self._slide_id(sample))
        return [sample]

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_updates": self.updates,
            "completed_updates": self.position,
            "replacement": True,
            "class_histogram": dict(sorted(Counter(self.sampled_labels).items())),
            "slide_histogram": dict(sorted(Counter(self.sampled_slide_ids).items())),
            "unique_replay_items": len(set(self.sampled_slide_ids)),
        }
