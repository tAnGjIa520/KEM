"""Whole-slide replay with immutable entry-time predictions; no disk reload of old bags."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass
class Entry:
    slide_id: str
    task: int
    label: int
    features: torch.Tensor
    sequence: torch.Tensor
    logit_columns: torch.Tensor
    logits: torch.Tensor

    def validate(self):
        if self.features.device.type != 'cpu' or self.features.dtype != torch.float32:
            raise ValueError('Stored features must be CPU float32')
        if self.features.ndim != 2 or self.features.shape[1] != 1024 or not len(self.features):
            raise ValueError('Expected complete UNI bag')
        if self.sequence.ndim != 1 or len(self.sequence) < 3:
            raise ValueError('Invalid diagnosis sequence')
        if self.sequence.dtype != torch.long or self.logit_columns.dtype != torch.long:
            raise ValueError('Token IDs must be int64')
        if (self.sequence < 0).any() or (self.logit_columns < 0).any() or self.logits.dtype != torch.float32:
            raise ValueError('Invalid token IDs or logits dtype')
        if self.sequence[0] != 0 or self.sequence[-1] != 2:
            raise ValueError('Missing BOS/EOS')
        if self.logit_columns.ndim != 1 or len(torch.unique(self.logit_columns)) != len(self.logit_columns):
            raise ValueError('Duplicate prediction columns')
        if self.logits.shape != (len(self.sequence)-1, len(self.logit_columns)):
            raise ValueError('Stored predictions/diagnosis mismatch')
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.logits).all():
            raise ValueError('Nonfinite stored data')
        if any(t.device.type != 'cpu' for t in (self.sequence, self.logit_columns, self.logits)):
            raise ValueError('Metadata tensors must be on CPU')


class ReplayMemory:
    def __init__(self, capacity=26, seed=1):
        if capacity < 1:
            raise ValueError('Invalid slide capacity')
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.entries = []
        self.seen_ids = set()
        self.last_task = -1

    def update(self, task, entries):
        if task != self.last_task + 1:
            raise ValueError('Task updates must be sequential')
        entries = list(entries)
        ids = [e.slide_id for e in entries]
        if len(ids) != len(set(ids)) or self.seen_ids.intersection(ids):
            raise ValueError('Repeated or resurrected slide')
        for entry in entries:
            entry.validate()
            if entry.task != task:
                raise ValueError('Historical/future data cannot enter current candidates')
        retained = [Entry(e.slide_id, e.task, e.label,
                          *(t.detach().clone().contiguous() for t in
                            (e.features, e.sequence, e.logit_columns, e.logits))) for e in entries]
        pool = self.entries + retained
        if len(pool) > self.capacity:
            keep = sorted(self.rng.choice(len(pool), self.capacity, replace=False).tolist())
            pool = [pool[i] for i in keep]
        self.entries = pool
        self.seen_ids.update(ids)
        self.last_task = task

    def sample(self):
        if not self.entries:
            raise ValueError('Empty replay memory')
        return self.entries[int(self.rng.integers(len(self.entries)))]

    @property
    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for e in self.entries
                   for t in (e.features, e.sequence, e.logit_columns, e.logits))

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save([vars(e) for e in self.entries], path/'entries.pt')
        (path/'metadata.json').write_text(json.dumps({
            'capacity_slides': self.capacity, 'last_task': self.last_task,
            'seen_ids': sorted(self.seen_ids), 'rng': self.rng.bit_generator.state,
            'slides': len(self.entries), 'patches': sum(len(e.features) for e in self.entries),
            'tensor_bytes': self.tensor_bytes,
            'policy': 'whole-slide random-overflow eviction; entry-time logits never refreshed',
        }, indent=2) + '\n')
        return sum(f.stat().st_size for f in path.iterdir())


def replay_loss(model, entry, device):
    """One replay update: diagnosis CE + old-column logit MSE (coefficient1)."""
    seq = entry.sequence.to(device)
    prediction = model(entry.features.to(device), seq[:-1][None], task=entry.task)[0]
    ce = torch.nn.functional.cross_entropy(prediction, seq[1:])
    mse = torch.nn.functional.mse_loss(prediction[:, entry.logit_columns.to(device)], entry.logits.to(device))
    return ce + mse
