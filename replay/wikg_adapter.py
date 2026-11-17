"""WiKG hooks for the shared continual-learning pipeline.

WiKG does not expose a bag-attention distribution.  The baseline adapter
therefore supports supervised training, random-patch ER and stored-logit
DER++, while selector/region methods that require attention fail explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .adapters import ModelOutput, ReplayModelAdapter
from .model_registry import (
    ReplayAdapterHooks,
    TrainingLossHooks,
    register_replay_adapter,
    register_training_loss,
)


SUPPORTED_CL_METHODS = frozenset({"none", "finetune", "ER", "derpp", "joint"})


def mask_unseen_logits(logits: torch.Tensor, seen_classes) -> torch.Tensor:
    if seen_classes is None:
        return logits
    return ReplayModelAdapter._mask_unseen(logits, seen_classes)


def continual_forward(model, features, seen_classes=None, *, params=None):
    output = (
        functional_call(model, params, (features,))
        if params is not None else model(features)
    )
    result = dict(output)
    result["logits"] = mask_unseen_logits(output["logits"], seen_classes)
    result["probabilities"] = F.softmax(result["logits"], dim=-1)
    result["predictions"] = result["logits"].argmax(dim=-1)
    return result


def supervised_loss(output, label, **_):
    loss = F.cross_entropy(output["logits"], label.long().reshape(-1))
    return loss, {"bag_loss": loss}


def _forward(model, features, seen_classes, params=None, patch_mask=None):
    if patch_mask is not None:
        raise ValueError("WiKG replay adapter does not support differentiable patch masks")
    output = continual_forward(model, features, seen_classes, params=params)
    return ModelOutput(logits=output["logits"], features=output["features"])


def _configure_proxy(model, scope):
    if scope not in {"aggregator_head", "classifier_only"}:
        raise ValueError(f"Unsupported bilevel proxy scope: {scope!r}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("fc") if scope == "classifier_only"
            else name.startswith(("linear", "norm", "fc"))
        )


def validate_continual_config(args):
    method = getattr(args, "cl_method", "none")
    if method not in SUPPORTED_CL_METHODS:
        raise ValueError(
            f"WiKG continual training does not support cl_method={method!r}; "
            f"supported methods: {sorted(SUPPORTED_CL_METHODS)}"
        )
    if method in {"ER", "derpp"}:
        if int(getattr(args, "buffer_size", 0)) <= 0:
            raise ValueError(f"WiKG {method} requires buffer_size > 0")
        if getattr(args, "distill_method", "random") != "random":
            raise ValueError(
                "WiKG ER/DER++ currently supports only distill_method=random; "
                "AMICIL, ConSlide and KEM selectors are not implemented"
            )
        if getattr(args, "buffer_mode", "legacy") not in {
            "legacy", "ratio_with_global_cap"
        }:
            raise ValueError(
                "WiKG supports legacy or ratio_with_global_cap buffers; "
                "adaptive/KEM buffer modes are not implemented"
            )
    elif int(getattr(args, "buffer_size", 0)) > 0 and method in {"none", "finetune"}:
        raise ValueError(f"WiKG {method} requires buffer_size=0")


register_replay_adapter("wikg", ReplayAdapterHooks(_forward, _configure_proxy))
register_training_loss("wikg", TrainingLossHooks(supervised_loss))
