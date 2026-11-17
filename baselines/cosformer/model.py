"""Independent COSFormer architecture / explicit MorphCL Class-IL adaptation.

Retrieval is separate. UNI is the visual training input; pretrained PubMedBERT
word vectors are required in production. No ground-truth task is used at test.
See PROTOCOL.md for departures from the published task-aware inference path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ExpertProjection(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.input_dim, self.dim = input_dim, dim
        self.base = nn.Linear(input_dim, dim)
        self.router_hidden = nn.Linear(input_dim, input_dim)
        nn.init.normal_(self.router_hidden.weight, std=.01)
        nn.init.zeros_(self.router_hidden.bias)
        self.experts = nn.ModuleList()
        self.router_rows = nn.ModuleList()

    def add_task(self):
        # Freeze old expert maps and router output rows, not the shared router.
        for block in list(self.experts) + list(self.router_rows):
            block.requires_grad_(False)
        device = self.base.weight.device
        self.experts.append(nn.Linear(self.input_dim, self.dim).to(device))
        row = nn.Linear(self.input_dim, 1).to(device)
        nn.init.normal_(row.weight, std=.01)
        nn.init.constant_(row.bias, .3)
        self.router_rows.append(row)

    def forward(self, x, task=None, scale=5., shift=1.):
        if not self.experts:
            raise ValueError('No task has been added')
        gate = torch.cat([row(F.gelu(self.router_hidden(x))) for row in self.router_rows], -1)
        boost = torch.ones(len(self.experts), device=x.device)
        residual = torch.zeros_like(boost)
        if task is not None:
            if not 0 <= task < len(self.experts):
                raise ValueError('Unknown task')
            boost[task] = scale
            residual[task] = shift
        # For Class-IL inference task=None: learned routing, no oracle boost.
        weights = (gate * boost).softmax(-1).mean(0) + residual
        projected = self.base(x)
        for weight, expert in zip(weights, self.experts):
            projected = projected + weight * expert(x)
        return F.relu(projected)


class EncoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        from nystrom_attention import NystromAttention
        self.norm = nn.LayerNorm(dim)
        self.attention = NystromAttention(dim=dim, dim_head=dim//8, heads=8,
                                         num_landmarks=dim//2, pinv_iterations=6,
                                         residual=True, dropout=.1)

    def forward(self, x):
        # Explicit right padding/mask avoids 0.0.9's left-pad/right-crop mismatch
        # and its in-place mutation of chunk views on current PyTorch.
        n = x.shape[1]
        pad = (-n) % self.attention.num_landmarks
        norm = F.pad(self.norm(x), (0, 0, 0, pad))
        mask = torch.arange(n + pad, device=x.device)[None] < n
        mask = mask.expand(x.shape[0], -1)
        return x + self.attention(norm, mask=mask)[:, :n]


class DecoderAttention(nn.Module):
    """Published decoder uses 8x64 heads even when d_model is768."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, 512)
        self.key = nn.Linear(dim, 512)
        self.value = nn.Linear(dim, 512)
        self.output = nn.Linear(512, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(.1)
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, memory, blocked=None):
        b, length, _ = query.shape
        q = self.query(query).reshape(b, length, 8, 64).transpose(1, 2)
        k = self.key(memory).reshape(b, -1, 8, 64).transpose(1, 2)
        v = self.value(memory).reshape(b, -1, 8, 64).transpose(1, 2)
        allowed = None if blocked is None else ~blocked[:, None]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                dropout_p=.1 if self.training else 0.)
        result = self.output(result.transpose(1, 2).reshape(b, length, 512))
        return self.norm(query + self.dropout(result))


class DecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = DecoderAttention(dim)
        self.cross_attn = DecoderAttention(dim)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.dropout = nn.Dropout(.1)
        self.ff = nn.Sequential(nn.Linear(dim, 2048), nn.ReLU(), nn.Dropout(.1), nn.Linear(2048, dim))

    def forward(self, x, memory, pad_mask):
        causal = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).triu(1)
        y = self.self_attn(x, x, blocked=causal[None] | pad_mask[:, None, :])
        valid = (~pad_mask).unsqueeze(-1)
        x = self.norms[0](x + self.dropout(y)) * valid
        y = self.cross_attn(x, memory)
        x = self.norms[1](x + self.dropout(y)) * valid
        return self.norms[2](x + self.dropout(self.ff(x))) * valid


class COSFormer(nn.Module):
    def __init__(self, word_vectors, input_dim=1024, max_length=32):
        super().__init__()
        if word_vectors.ndim != 2 or word_vectors.shape[1] % 8:
            raise ValueError('Expected [vocabulary, decoder dimension] pretrained vectors')
        dim = word_vectors.shape[1]
        self.projection = ExpertProjection(input_dim, dim)
        self.encoder = nn.ModuleList(EncoderLayer(dim) for _ in range(2))
        self.decoder = nn.ModuleList(DecoderLayer(dim) for _ in range(2))
        self.words = nn.Embedding.from_pretrained(word_vectors.float(), freeze=True, padding_idx=3)
        pos = torch.arange(max_length).float()[:, None]
        scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.)/dim))
        table = torch.zeros(max_length, dim)
        table[:, 0::2], table[:, 1::2] = torch.sin(pos*scale), torch.cos(pos*scale)
        table[0] = 0
        self.positions = nn.Embedding.from_pretrained(table, freeze=False)
        self.head = nn.Linear(dim, len(word_vectors), bias=False)
        self.register_buffer('seen_tokens', torch.zeros(len(word_vectors), dtype=torch.bool))

    def add_task(self, token_ids):
        self.projection.add_task()
        self.seen_tokens[list(token_ids) + [0, 1, 2, 3]] = True

    def encode(self, x, task=None):
        memory = self.projection(x, task)[None]
        for layer in self.encoder:
            memory = layer(memory)
        return memory

    def decode(self, memory, tokens):
        if tokens.ndim != 2 or tokens.shape[1] >= self.positions.num_embeddings:
            raise ValueError('Invalid sequence length')
        padding = tokens.eq(3)
        pos = torch.arange(1, tokens.shape[1]+1, device=tokens.device)[None].expand_as(tokens)
        x = self.words(tokens) + self.positions(pos.masked_fill(padding, 0))
        if memory.shape[0] == 1 and tokens.shape[0] > 1:
            memory = memory.expand(tokens.shape[0], -1, -1)
        for layer in self.decoder:
            x = layer(x, memory, padding)
        return self.head(x).masked_fill(~self.seen_tokens, -torch.inf)

    def forward(self, x, tokens, task=None):
        return self.decode(self.encode(x, task), tokens)

    @torch.no_grad()
    def class_scores(self, x, sequences):
        """Length-normalized diagnosis log likelihood over all seen classes.

        Explicit classification adaptation, NOT original greedy text generation.
        sequences maps global class ID to [BOS, words..., EOS].
        """
        memory = self.encode(x, task=None)
        scores = {}
        for label, seq in sequences.items():
            ids = torch.tensor(seq, device=x.device)
            logits = self.decode(memory, ids[:-1][None])[0]
            scores[label] = logits.log_softmax(-1).gather(1, ids[1:, None]).mean()
        return scores
