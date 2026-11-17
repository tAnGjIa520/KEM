"""MambaMIL-specific replay and region adapters.

Mamba state propagation makes soft patch masking semantically invalid.  The
adapter therefore exposes ordinary model forwards and exact, unpadded region
batches; KEM uses input-gradient matching for selection.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import torch
from torch.func import functional_call

from .adapters import ModelOutput, ReplayModelAdapter
from .model_registry import ReplayAdapterHooks, register_replay_adapter


def _call(model, params, features):
    return functional_call(model, params, (features,)) if params is not None else model(features)


def _forward(model, features, seen_classes, params=None, patch_mask=None):
    if patch_mask is not None:
        raise ValueError(
            "MambaMIL does not support soft patch masks; use the gradient-matching KEM backend"
        )
    output = _call(model, params, features)
    logits = output["logits"]
    if seen_classes is not None:
        logits = ReplayModelAdapter._mask_unseen(logits, seen_classes)
    return ModelOutput(logits=logits, features=output["features"])


def _configure_proxy(model, scope):
    if scope not in {"aggregator_head", "classifier_only"}:
        raise ValueError(f"Unsupported bilevel proxy scope: {scope!r}")
    for name, parameter in model.named_parameters():
        if scope == "classifier_only":
            trainable = name.startswith("classifier")
        else:
            trainable = (
                name.startswith("norm")
                or name.startswith("attention")
                or name.startswith("classifier")
            )
        parameter.requires_grad_(trainable)


def canonicalize_region_order(regions):
    """Preserve the original patch sequence within every Mamba region."""
    return [torch.as_tensor(region, dtype=torch.long).sort().values for region in regions]


def region_ordering_hash(regions) -> str:
    digest = hashlib.sha256(b"mambamil:original_patch_index:v1")
    for region in regions:
        values = torch.as_tensor(region, dtype=torch.int64).cpu().contiguous()
        digest.update(int(values.numel()).to_bytes(8, "little", signed=False))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()[:20]


class RegionBatchMambaMILAdapter:
    """Run exact-length Mamba region buckets without padded state pollution."""

    ordering_policy = "original_patch_index"

    @staticmethod
    def forward(model, padded, mask, seen_classes, params=None):
        if padded.ndim != 3 or mask.shape != padded.shape[:2]:
            raise ValueError("expected padded [R,M,D] and mask [R,M]")
        if model.__class__.__name__ != "MambaMIL":
            raise ValueError("MambaMIL region adapter requires a MambaMIL model")

        lengths = []
        for row in range(padded.shape[0]):
            length = int(mask[row].sum())
            if length < 1 or not bool(mask[row, :length].all()) or bool(mask[row, length:].any()):
                raise ValueError("region mask must be a non-empty contiguous prefix")
            lengths.append(length)

        buckets = defaultdict(list)
        for row, length in enumerate(lengths):
            buckets[length].append(row)

        outputs = [None] * padded.shape[0]
        for length, rows in sorted(buckets.items()):
            indices = torch.as_tensor(rows, device=padded.device, dtype=torch.long)
            features = padded.index_select(0, indices)[:, :length]
            output = _call(model, params, features)
            logits = output["logits"]
            for offset, row in enumerate(rows):
                outputs[row] = logits[offset:offset + 1]

        logits = torch.cat(outputs, dim=0)
        if seen_classes is not None:
            logits = ReplayModelAdapter._mask_unseen(logits, seen_classes)
        return logits


register_replay_adapter(
    "mambamil",
    ReplayAdapterHooks(
        forward=_forward,
        configure_proxy_parameters=_configure_proxy,
        region_forward=RegionBatchMambaMILAdapter.forward,
    ),
)
ReplayModelAdapter.supported_models.add("mambamil")

