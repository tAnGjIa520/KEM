"""Small model adapters for differentiable replay selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.func import functional_call

from .model_registry import (
    ReplayAdapterHooks,
    get_replay_adapter,
    register_replay_adapter,
    registered_replay_adapters,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor


class ReplayModelAdapter:
    """Normalize model calls without teaching the selector about each backbone."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hooks = get_replay_adapter(model_name)

    def forward(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        seen_classes,
        params: Mapping[str, torch.Tensor] | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        return self.hooks.forward(
            model, features, seen_classes, params=params, patch_mask=patch_mask
        )

    def configure_proxy_parameters(self, model: torch.nn.Module, scope: str) -> None:
        self.hooks.configure_proxy_parameters(model, scope)

    @staticmethod
    def _mask_unseen(logits: torch.Tensor, seen_classes) -> torch.Tensor:
        seen = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
        mask = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
        mask[seen] = False
        return logits.masked_fill(mask.unsqueeze(0), -100.0)


def _call(model, params, features, kwargs):
    return functional_call(model, params, (features,), kwargs) if params is not None else model(features, **kwargs)


def _clam_forward(model, features, seen_classes, params=None, patch_mask=None):
    output = _call(model, params, features, {
        "return_features": True,
        "seen_classes": seen_classes,
        "patch_mask": patch_mask,
    })
    return ModelOutput(logits=output["logits"], features=output["features"])


def _transmil_forward(model, features, seen_classes, params=None, patch_mask=None):
    output = _call(model, params, features, {"patch_mask": patch_mask})
    logits = output["logits"]
    if seen_classes is not None:
        logits = ReplayModelAdapter._mask_unseen(logits, seen_classes)
    return ModelOutput(logits=logits, features=output["features"])


def _clam_proxy(model, scope):
    if scope not in {"aggregator_head", "classifier_only"}:
        raise ValueError(f"Unsupported bilevel proxy scope: {scope!r}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("classifiers") if scope == "classifier_only"
            else name.startswith("attention_net") or name.startswith("classifiers")
        )


def _transmil_proxy(model, scope):
    if scope not in {"aggregator_head", "classifier_only"}:
        raise ValueError(f"Unsupported bilevel proxy scope: {scope!r}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("_fc2") if scope == "classifier_only"
            else name.startswith("layer2") or name.startswith("norm") or name.startswith("_fc2")
        )


for _name in ("clam_sb", "clam_mb"):
    register_replay_adapter(_name, ReplayAdapterHooks(_clam_forward, _clam_proxy))
register_replay_adapter("transmil", ReplayAdapterHooks(_transmil_forward, _transmil_proxy))

# Backwards-compatible, dynamically evaluated view used by older callers.
ReplayModelAdapter.supported_models = set(registered_replay_adapters())
