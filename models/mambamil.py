"""H200-compatible MambaMIL adapter used for migration parity tests."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SequenceReorderingMamba(nn.Module):
    """Sequence-reordering block backed by current mamba-ssm CUDA kernels."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.reordered_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    @staticmethod
    def reorder(x, rate):
        batch, length, dim = x.shape
        padding = (-length) % rate
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
        groups = x.shape[1] // rate
        x = x.reshape(batch, groups, rate, dim).transpose(1, 2).reshape(batch, -1, dim)
        return x, length, groups

    @staticmethod
    def restore(x, rate, length, groups):
        batch, _, dim = x.shape
        x = x.reshape(batch, rate, groups, dim).transpose(1, 2).reshape(batch, -1, dim)
        return x[:, :length]

    def forward(self, x, rate=10):
        reordered, length, groups = self.reorder(x, rate)
        reordered = self.reordered_mamba(reordered)
        reordered = self.restore(reordered, rate, length, groups)
        return self.forward_mamba(x) + reordered


def initialize_weights(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        if isinstance(child, nn.LayerNorm):
            nn.init.constant_(child.bias, 0)
            nn.init.constant_(child.weight, 1.0)


class MambaMIL(nn.Module):
    """MambaMIL with KEM dictionary output and legacy tuple compatibility."""

    def __init__(
        self,
        in_dim,
        n_classes,
        dropout=0.25,
        act="gelu",
        layer=2,
        rate=10,
        mamba_type="SRMamba",
    ):
        super().__init__()
        fc = [nn.Linear(in_dim, 512)]
        if act.lower() == "relu":
            fc.append(nn.ReLU())
        elif act.lower() == "gelu":
            fc.append(nn.GELU())
        if dropout:
            fc.append(nn.Dropout(dropout))
        self._fc1 = nn.Sequential(*fc)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        if mamba_type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SequenceReorderingMamba(512, d_state=16, d_conv=4, expand=2),
                    )
                )
        elif mamba_type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(d_model=512, d_state=16, d_conv=4, expand=2),
                    )
                )
        else:
            raise NotImplementedError(f"Mamba type {mamba_type!r} is not implemented")

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = mamba_type
        self.apply(initialize_weights)

    def forward(self, x, legacy=False):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        tokens = self._fc1(x.float())
        for layer in self.layers:
            residual = tokens
            tokens = layer[0](tokens)
            if self.type == "SRMamba":
                tokens = layer[1](tokens, rate=self.rate)
            else:
                tokens = layer[1](tokens)
            tokens = tokens + residual

        tokens = self.norm(tokens)
        attention = self.attention(tokens).transpose(1, 2)
        attention = F.softmax(attention, dim=-1)
        slide_features = torch.bmm(attention, tokens).squeeze(1)
        logits = self.classifier(slide_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.topk(logits, 1, dim=1)[1]
        if legacy:
            return logits, probabilities, predictions, None, None
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "features": slide_features,
            "A": attention,
            "tokens": tokens,
        }

    def relocate(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; CPU execution is disabled")
        return self.cuda()
