"""KEM adapter for the official 2DMamba whole-slide MIL model.

The model equations follow AtlasAnalyticsLab/2DMamba, commit 9fe12aaa.
Only the return interface and explicit runtime validation are KEM-specific.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .twodmamba import Mamba, MambaConfig


def initialize_weights(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        elif isinstance(child, nn.LayerNorm):
            nn.init.constant_(child.bias, 0)
            nn.init.constant_(child.weight, 1.0)


class MambaMIL2D(nn.Module):
    """Coordinate-aware 2DMamba MIL with a KEM dictionary output."""

    def __init__(
        self,
        in_dim=1024,
        n_classes=2,
        mamba_dim=128,
        state_dim=16,
        layers=1,
        dropout=0.25,
        max_w=197796,
        max_h=110976,
        patch_size=512,
        pad_token="trainable",
    ):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("2DMamba requires CUDA; CPU fallback is disabled")

        self._fc1 = nn.Sequential(
            nn.Linear(in_dim, mamba_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        config = MambaConfig(
            d_model=mamba_dim,
            n_layers=layers,
            d_state=state_dim,
            inner_layernorms=False,
            pscan=True,
            use_cuda=True,
            mamba_2d=True,
            mamba_2d_max_w=max_w,
            mamba_2d_max_h=max_h,
            mamba_2d_patch_size=patch_size,
            mamba_2d_pad_token=pad_token,
        )
        self.layers = Mamba(config)
        self.config = config
        self.norm = nn.LayerNorm(mamba_dim)
        self.attention = nn.Sequential(
            nn.Linear(mamba_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Linear(mamba_dim, n_classes)
        self.n_classes = n_classes
        self.apply(initialize_weights)

    def forward(self, features, coords):
        if not features.is_cuda or not coords.is_cuda:
            raise RuntimeError("2DMamba features and coordinates must be CUDA tensors")
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if coords.ndim == 3:
            if coords.size(0) != 1:
                raise ValueError("2DMamba supports batch_size=1 only")
            coords = coords.squeeze(0)
        if features.ndim != 3 or features.size(0) != 1:
            raise ValueError(f"Expected features [1,N,D], got {tuple(features.shape)}")
        if coords.ndim != 2 or coords.size(1) != 2 or coords.size(0) != features.size(1):
            raise ValueError(
                f"Expected coords [N,2] aligned with features, got {tuple(coords.shape)}"
            )

        tokens = self._fc1(features.float())
        grid_tokens = self.norm(self.layers(tokens, coords, None))
        if grid_tokens.ndim != 3:
            raise RuntimeError(f"CUDA 2DMamba must return [B,H*W,D], got {tuple(grid_tokens.shape)}")

        attention = self.attention(grid_tokens).transpose(1, 2)
        attention = F.softmax(attention, dim=-1)
        slide_features = torch.bmm(attention, grid_tokens).squeeze(1)
        logits = self.classifier(slide_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1, keepdim=True)

        normalized = torch.div(coords, self.config.mamba_2d_patch_size, rounding_mode="floor")
        flat_indices = normalized[:, 1].long() * self.config.WW + normalized[:, 0].long()
        patch_attention = attention.index_select(2, flat_indices)
        patch_tokens = grid_tokens.index_select(1, flat_indices)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "features": slide_features,
            "A": patch_attention,
            "tokens": patch_tokens,
            "grid_attention": attention,
        }

    def relocate(self):
        if not torch.cuda.is_available():
            raise RuntimeError("2DMamba requires CUDA; CPU fallback is disabled")
        return self.cuda()


__all__ = ["MambaMIL2D"]
