"""Registries for model-specific continual-learning hooks.

Model integrations live in small modules and register their hooks here.  This
keeps the replay selector and training loop independent of optional MIL
architectures while retaining explicit failures for unsupported operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReplayAdapterHooks:
    forward: Callable
    configure_proxy_parameters: Callable
    region_forward: Callable | None = None


@dataclass(frozen=True)
class TrainingLossHooks:
    compute: Callable


_REPLAY_ADAPTERS: dict[str, ReplayAdapterHooks] = {}
_TRAINING_LOSSES: dict[str, TrainingLossHooks] = {}


def register_replay_adapter(name: str, hooks: ReplayAdapterHooks, *, replace: bool = False):
    if name in _REPLAY_ADAPTERS and not replace:
        raise ValueError(f"Replay adapter {name!r} is already registered")
    _REPLAY_ADAPTERS[name] = hooks


def get_replay_adapter(name: str) -> ReplayAdapterHooks:
    try:
        return _REPLAY_ADAPTERS[name]
    except KeyError as error:
        supported = ", ".join(sorted(_REPLAY_ADAPTERS)) or "<none>"
        raise ValueError(
            f"Bilevel replay does not support model {name!r}; supported models: {supported}"
        ) from error


def registered_replay_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REPLAY_ADAPTERS))


def register_training_loss(name: str, hooks: TrainingLossHooks, *, replace: bool = False):
    if name in _TRAINING_LOSSES and not replace:
        raise ValueError(f"Training loss {name!r} is already registered")
    _TRAINING_LOSSES[name] = hooks


def get_training_loss(name: str) -> TrainingLossHooks:
    try:
        return _TRAINING_LOSSES[name]
    except KeyError as error:
        supported = ", ".join(sorted(_TRAINING_LOSSES)) or "<none>"
        raise ValueError(
            f"No training loss registered for model {name!r}; supported models: {supported}"
        ) from error


def registered_training_losses() -> tuple[str, ...]:
    return tuple(sorted(_TRAINING_LOSSES))
