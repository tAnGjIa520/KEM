"""Self-contained ACMIL-GA implementation used by the single-task runner."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ACMILConfig:
    n_class: int
    n_token: int = 5
    D_feat: int = 1024
    D_inner: int = 512


class DimReduction(nn.Module):
    def __init__(self, n_channels: int, m_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu1(self.fc1(x))


class Classifier1FC(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, droprate: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if droprate:
            self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        if self.droprate:
            x = self.dropout(x)
        return self.fc(x)


class GatedAttention(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128, tokens: int = 5):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, tokens)

    def forward(self, x):
        return self.attention_weights(self.attention_V(x) * self.attention_U(x)).transpose(1, 0)


class ACMIL_GA(nn.Module):
    """Official ACMIL-GA math with a stable KEM-local import path."""

    def __init__(self, conf: ACMILConfig, attention_dim=128, droprate=0.0,
                 n_token=5, n_masked_patch=10, mask_drop=0.6):
        super().__init__()
        if n_token != conf.n_token:
            raise ValueError("n_token must match conf.n_token")
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = GatedAttention(conf.D_inner, attention_dim, n_token)
        self.classifier = nn.ModuleList(
            [Classifier1FC(conf.D_inner, conf.n_class, droprate) for _ in range(n_token)]
        )
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.Slide_classifier = Classifier1FC(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = mask_drop

    def forward(self, x):
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"ACMIL expects [1, patches, features], got {tuple(x.shape)}")
        x = self.dimreduction(x[0])
        attention = self.attention(x)
        if self.n_masked_patch > 0 and self.training:
            tokens, patches = attention.shape
            topk = min(self.n_masked_patch, patches)
            indices = torch.topk(attention, topk, dim=-1).indices
            selected = torch.argsort(torch.rand(*indices.shape, device=attention.device), dim=-1)
            selected = selected[:, :int(topk * self.mask_drop)]
            masked = indices[torch.arange(tokens, device=attention.device).unsqueeze(-1), selected]
            mask = torch.ones(tokens, patches, device=attention.device)
            mask.scatter_(-1, masked, 0)
            attention = attention.masked_fill(mask == 0, -1e9)
        raw_attention = attention
        normalized = F.softmax(attention, dim=1)
        token_features = torch.mm(normalized, x)
        token_logits = torch.stack(
            [head(token_features[i]) for i, head in enumerate(self.classifier)], dim=0
        )
        bag_attention = F.softmax(raw_attention, dim=1).mean(0, keepdim=True)
        bag_feature = torch.mm(bag_attention, x)
        return token_logits, self.Slide_classifier(bag_feature), raw_attention.unsqueeze(0)


class ACMILContinual(ACMIL_GA):
    """ACMIL-GA with the dictionary interface used by the KEM framework.

    Parameter names intentionally remain identical to :class:`ACMIL_GA`, so
    official and existing single-task checkpoints strict-load without a state
    conversion step.
    """

    replay_model_name = "acmil"

    def forward(self, x, seen_classes=None, patch_mask=None, return_attn=True):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"ACMIL expects [N,D] or [1,N,D], got {tuple(x.shape)}")
        hidden = self.dimreduction(x[0])
        raw_attention = self.attention(hidden)
        if self.n_masked_patch > 0 and self.training:
            tokens, patches = raw_attention.shape
            topk = min(self.n_masked_patch, patches)
            indices = torch.topk(raw_attention, topk, dim=-1).indices
            selected = torch.argsort(
                torch.rand(*indices.shape, device=raw_attention.device), dim=-1
            )[:, :int(topk * self.mask_drop)]
            masked = indices[
                torch.arange(tokens, device=raw_attention.device).unsqueeze(-1), selected
            ]
            random_mask = torch.ones(tokens, patches, device=raw_attention.device)
            random_mask.scatter_(-1, masked, 0)
            raw_attention = raw_attention.masked_fill(random_mask == 0, -1e9)

        token_attention = F.softmax(raw_attention, dim=1)
        if patch_mask is not None:
            shared_mask = torch.as_tensor(
                patch_mask, device=token_attention.device, dtype=token_attention.dtype
            ).reshape(1, -1)
            if shared_mask.shape[1] != token_attention.shape[1]:
                raise ValueError(
                    f"patch_mask has {shared_mask.shape[1]} entries for "
                    f"{token_attention.shape[1]} patches"
                )
            token_attention = token_attention * shared_mask
            token_attention = token_attention / token_attention.sum(
                dim=1, keepdim=True
            ).clamp_min(torch.finfo(token_attention.dtype).eps)

        token_features = torch.mm(token_attention, hidden)
        token_logits = torch.stack(
            [head(token_features[i]) for i, head in enumerate(self.classifier)], dim=0
        )
        bag_attention = token_attention.mean(0, keepdim=True)
        bag_feature = torch.mm(bag_attention, hidden)
        slide_logits = self.Slide_classifier(bag_feature)

        if seen_classes is not None and len(seen_classes) < slide_logits.shape[-1]:
            seen = torch.as_tensor(seen_classes, device=slide_logits.device, dtype=torch.long)
            unseen = torch.ones(slide_logits.shape[-1], device=slide_logits.device, dtype=torch.bool)
            unseen[seen] = False
            slide_logits = slide_logits.masked_fill(unseen.unsqueeze(0), -100.0)
            token_logits = token_logits.masked_fill(unseen.unsqueeze(0), -100.0)

        output = {
            "logits": slide_logits,
            "features": bag_feature,
            "A": bag_attention,
            "A_tokens": raw_attention.unsqueeze(0),
            "A_tokens_normalized": token_attention.unsqueeze(0),
            "token_logits": token_logits,
            "token_features": token_features,
        }
        return output
