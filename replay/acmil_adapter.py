"""ACMIL hooks for continual training and differentiable replay selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .adapters import ModelOutput
from .model_registry import (
    ReplayAdapterHooks,
    TrainingLossHooks,
    register_replay_adapter,
    register_training_loss,
)


def acmil_official_loss(output, label, **_):
    label = label.long().reshape(-1)
    token_logits = output["token_logits"]
    token_targets = label.repeat_interleave(token_logits.shape[0])
    token_loss = F.cross_entropy(token_logits, token_targets)
    slide_loss = F.cross_entropy(output["logits"], label)
    attention = output["A_tokens_normalized"]
    tokens = attention.shape[1]
    diversity = attention.new_zeros(())
    pairs = tokens * (tokens - 1) // 2
    if pairs:
        for left in range(tokens):
            for right in range(left + 1, tokens):
                diversity = diversity + F.cosine_similarity(
                    attention[:, left], attention[:, right], dim=-1
                ).mean() / pairs
    total = token_loss + slide_loss + diversity
    return total, {
        "token_loss": token_loss,
        "slide_loss": slide_loss,
        "diversity_loss": diversity,
    }


def _forward(model, features, seen_classes, params=None, patch_mask=None):
    kwargs = {"seen_classes": seen_classes, "patch_mask": patch_mask}
    output = (
        functional_call(model, params, (features,), kwargs)
        if params is not None else model(features, **kwargs)
    )
    return ModelOutput(logits=output["logits"], features=output["features"])


def _configure_proxy(model, scope):
    if scope not in {"aggregator_head", "classifier_only"}:
        raise ValueError(f"Unsupported bilevel proxy scope: {scope!r}")
    prefixes = ("classifier", "Slide_classifier")
    if scope == "aggregator_head":
        prefixes = ("attention",) + prefixes
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))


def region_forward(model, padded, mask, seen_classes, params=None):
    if padded.ndim != 3 or mask.shape != padded.shape[:2]:
        raise ValueError("expected padded [R,M,D] and mask [R,M]")
    outputs = []
    for row in range(padded.shape[0]):
        length = int(mask[row].sum())
        if length < 1 or not bool(mask[row, :length].all()) or bool(mask[row, length:].any()):
            raise ValueError("region mask must be a non-empty contiguous prefix")
        features = padded[row, :length]
        kwargs = {"seen_classes": seen_classes}
        output = (
            functional_call(model, params, (features,), kwargs)
            if params is not None else model(features, **kwargs)
        )
        outputs.append(output["logits"])
    return torch.cat(outputs, dim=0)


register_replay_adapter(
    "acmil", ReplayAdapterHooks(_forward, _configure_proxy, region_forward)
)
register_training_loss("acmil", TrainingLossHooks(acmil_official_loss))
