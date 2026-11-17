"""License-safe adapters for externally provided foundation slide encoders."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


FEATURE_PROFILES = {
    "chief_ctranspath": {"input_dim": 768, "requires_coords": False, "requires_patch_size": False},
    "titan_conch_v1_5": {"input_dim": 768, "requires_coords": True, "requires_patch_size": True},
    "prism_virchow_1": {"input_dim": 2560, "requires_coords": False, "requires_patch_size": False},
}


def validate_feature_profile(
    profile: str,
    features: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    patch_size_lv0: Optional[int] = None,
) -> None:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {profile}")
    spec = FEATURE_PROFILES[profile]
    if features.ndim not in (2, 3) or features.shape[-1] != spec["input_dim"]:
        raise ValueError(
            f"{profile} requires [..., {spec['input_dim']}] features, got {tuple(features.shape)}"
        )
    if spec["requires_coords"] and coords is None:
        raise ValueError(f"{profile} requires patch coordinates; projection-based compatibility is forbidden")
    if spec["requires_patch_size"] and patch_size_lv0 is None:
        raise ValueError(f"{profile} requires patch_size_lv0")
    if profile != "chief_ctranspath":
        raise NotImplementedError(
            f"{profile} validation is available, but its gated provider is not integrated in this release"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_chief(provider_root: str | Path):
    """Load CHIEF from an external provider without vendoring its AGPL source."""
    backbone = Path(provider_root) / "slide_encoder_models" / "chief" / "backbone.py"
    if not backbone.is_file():
        raise FileNotFoundError(f"External CHIEF provider not found: {backbone}")
    spec = importlib.util.spec_from_file_location("kem_external_chief_backbone", backbone)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import external CHIEF provider: {backbone}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHIEF


class ChiefFoundationEncoder(nn.Module):
    """Frozen CHIEF encoder backed by externally configured code and checkpoints."""

    output_dim = 768

    def __init__(
        self,
        provider_root: str,
        checkpoint_path: str,
        word_embedding_path: str,
        anatomic_id: int = 13,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= int(anatomic_id) < 19:
            raise ValueError("CHIEF anatomic_id must be in [0, 18]")
        chief_cls = _load_external_chief(provider_root)
        self.backbone = chief_cls(
            size_arg="small",
            word_embedding_path=word_embedding_path,
            dropout=True,
            n_classes=2,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(state, strict=True)
        self.anatomic_id = int(anatomic_id)
        self.checkpoint_path = str(checkpoint_path)
        self.word_embedding_path = str(word_embedding_path)
        self.provider_root = str(provider_root)
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, input_dict: Dict[str, Any]) -> Dict[str, torch.Tensor | None]:
        features = input_dict.get("features", input_dict.get("feats"))
        if features is None:
            raise KeyError("CHIEF input requires 'features'")
        validate_feature_profile(
            "chief_ctranspath", features, input_dict.get("coords"), input_dict.get("patch_size_lv0")
        )
        if features.ndim == 3 and features.size(0) != 1:
            raise ValueError("CHIEF currently supports one slide per forward pass")
        results = self.backbone(features, x_anatomic=self.anatomic_id)
        embedding = results["WSI_feature"]
        if embedding.shape != (1, self.output_dim):
            raise RuntimeError(f"Unexpected CHIEF embedding shape: {tuple(embedding.shape)}")
        if not torch.isfinite(embedding).all():
            raise RuntimeError("CHIEF produced NaN/Inf embedding")
        attention = results.get("attention_raw")
        return {"features": embedding, "A": attention, "tokens": None}


class FoundationSlideClassifier(nn.Module):
    """Classifier for cached foundation-slide embeddings."""

    def __init__(self, n_classes: int, input_dim: int = 768, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.input_dim, int(n_classes))

    def forward(self, features: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor | None]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim == 3 and features.size(1) == 1:
            features = features[:, 0]
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(f"Expected cached slide embeddings [B,{self.input_dim}], got {tuple(features.shape)}")
        logits = self.classifier(self.dropout(features))
        return {
            "logits": logits,
            "features": features,
            "A": None,
            "tokens": None,
            "probabilities": torch.softmax(logits, dim=-1),
            "predictions": logits.argmax(dim=-1),
        }
