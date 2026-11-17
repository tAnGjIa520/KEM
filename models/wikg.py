"""Self-contained WiKG backbone for single-bag WSI classification.

The parameter names and computations intentionally match the official WiKG
implementation so official checkpoints can be loaded with ``strict=True``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WiKG(nn.Module):
    def __init__(
        self,
        dim_in=384,
        dim_hidden=512,
        topk=6,
        n_classes=2,
        agg_type="bi-interaction",
        dropout=0.3,
        pool="mean",
    ):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden**-0.5
        self.topk = topk
        self.agg_type = agg_type

        # Retained for strict compatibility with official checkpoints.
        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if agg_type == "gcn":
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif agg_type == "sage":
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif agg_type == "bi-interaction":
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError(f"Unsupported aggregation: {agg_type}")

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)
        self.pool = pool
        if pool not in {"mean", "max"}:
            raise ValueError("The self-contained KEM WiKG adapter supports mean/max pooling")

    def forward(self, x):
        if isinstance(x, dict):
            x = x["feature"]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.size(0) != 1:
            raise ValueError(f"WiKG expects one bag shaped [1, N, D], got {tuple(x.shape)}")
        if x.size(1) < self.topk:
            raise ValueError(
                f"WiKG topk={self.topk} requires at least {self.topk} patches, got {x.size(1)}"
            )

        x = self._fc1(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        e_h = self.W_head(x)
        e_t = self.W_tail(x)
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0), device=topk_index.device).view(-1, 1, 1)
        neighbours = e_t[batch_indices, topk_index_expanded, :]

        topk_prob = F.softmax(topk_weight, dim=2)
        relation = torch.mul(topk_prob.unsqueeze(-1), neighbours) + torch.matmul(
            (1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2)
        )
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + relation)
        knowledge_weight = torch.einsum("ijkl,ijkm->ijk", neighbours, gate)
        knowledge_prob = F.softmax(knowledge_weight, dim=2).unsqueeze(dim=2)
        neighbour_embedding = torch.matmul(knowledge_prob, neighbours).squeeze(dim=2)

        if self.agg_type == "gcn":
            embedding = self.activation(self.linear(e_h + neighbour_embedding))
        elif self.agg_type == "sage":
            embedding = self.activation(self.linear(torch.cat([e_h, neighbour_embedding], dim=2)))
        else:
            sum_embedding = self.activation(self.linear1(e_h + neighbour_embedding))
            bi_embedding = self.activation(self.linear2(e_h * neighbour_embedding))
            embedding = sum_embedding + bi_embedding

        tokens = self.message_dropout(embedding)
        if self.pool == "mean":
            features = tokens.squeeze(0).mean(dim=0, keepdim=True)
        else:
            features = tokens.squeeze(0).max(dim=0, keepdim=True).values
        features = self.norm(features)
        logits = self.fc(features)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "predictions": logits.argmax(dim=1),
            "features": features,
            "tokens": tokens,
            "A": None,
        }
