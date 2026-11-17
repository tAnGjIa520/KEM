"""Independent AGLR-CL MorphCL adapter. Smoke runs are NOT paper results.

Frozen UNI -> gated AB-MIL -> current-training top-attention -> full GMM.
No imports from KEM. Dataset paths are supplied via CLI, never inferred globally.
"""
import argparse
import csv
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

from latent_memory import FitConfig, LatentMemory


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ABMIL(nn.Module):
    def __init__(self, dim=1024, classes=11):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.25))
        self.attn_v = nn.Sequential(nn.Linear(512, 256), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
        self.attn_w = nn.Linear(256, 1)
        self.head = nn.Linear(512, classes)

    def forward(self, x, seen):
        h = self.project(x)
        attention = self.attn_w(self.attn_v(h) * self.attn_u(h)).flatten().softmax(0)
        logits = self.head((attention[:, None] * h).sum(0))
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[list(seen)] = False
        return logits.masked_fill(mask, -torch.inf), attention


class Corpus:
    def __init__(self, manifest, root, splits, fold):
        self.root = Path(root)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        self.rows = {r['slide_id']: r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate slide IDs')
        self.parts, self.hashes = [], {'manifest': digest(Path(manifest))}
        patients = {p: set() for p in ('train', 'val', 'test')}
        slides = {p: set() for p in patients}
        for task in range(5):
            path = Path(splits) / f'fold_{fold}' / f'splits_{task}.csv'
            self.hashes[f'task{task}'] = digest(path)
            with path.open() as f:
                table = list(csv.DictReader(f))
            parts = {}
            for part in patients:
                ids = [r[part] for r in table if r.get(part)]
                if len(ids) != len(set(ids)) or not ids:
                    raise ValueError('Duplicate or empty partition')
                for sid in ids:
                    row = self.rows[sid]
                    if int(row['task_id']) != task:
                        raise ValueError('Task/slide mismatch')
                    if int(row['feature_dim']) != 1024:
                        raise ValueError('Expected frozen UNI 1024')
                    patients[part].add(row['case_id'])
                    slides[part].add(sid)
                parts[part] = ids
            self.parts.append(parts)
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if patients[a] & patients[b] or slides[a] & slides[b]:
                raise ValueError(f'Patient/slide leakage: {a}/{b}')

    def label(self, sid):
        return int(self.rows[sid]['label_id'])

    def load(self, sid, smoke_patches=0):
        path = self.root / self.rows[sid]['relative_path']
        x = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(x, dict):
            x = x['embedding']
        if x.ndim != 2 or x.shape[1] != 1024 or not len(x) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid features: {sid}')
        if smoke_patches:
            x = x[:smoke_patches]
        return x.float()

    def subset(self, ids, per_class):
        if not per_class:
            return ids
        counts, out = {}, []
        for sid in ids:
            label = self.label(sid)
            if counts.get(label, 0) < per_class:
                out.append(sid)
                counts[label] = counts.get(label, 0) + 1
        return out


def evaluate(model, corpus, ids, seen, cap):
    model.eval()
    truth, probabilities = [], []
    with torch.no_grad():
        for sid in ids:
            logits, _ = model(corpus.load(sid, cap).cuda(), seen)
            truth.append(corpus.label(sid))
            probabilities.append(logits.softmax(0).cpu().numpy())
    probabilities = np.asarray(probabilities)
    pred = probabilities.argmax(1)
    classes = sorted(set(truth))
    auc = [roc_auc_score(np.asarray(truth) == c, probabilities[:, c])
           for c in classes if len(set(np.asarray(truth) == c)) == 2]
    return {'accuracy': float(np.mean(pred == truth)),
            'macro_f1': float(f1_score(truth, pred, labels=classes, average='macro', zero_division=0)),
            'macro_auroc': float(np.mean(auc)) if auc else None, 'slides': len(ids)}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    if args.fit_probe and (args.tasks != 1 or args.epochs != 1):
        raise ValueError('Full-data fit probe must be one task and one epoch')
    if not (args.smoke or args.fit_probe) and (args.tasks != 5 or args.epochs != 50 or args.replay_updates != 78):
        raise ValueError('Non-smoke protocol must be 5 tasks, 50 epochs, 78 replay updates')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    corpus = Corpus(args.manifest, args.features, args.splits, args.fold)
    model = ABMIL().cuda()
    memory = LatentMemory(FitConfig(seed=args.seed, fit_backend=args.fit_backend,
                                   max_iter=args.gmm_max_iter))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', vars(args))
    write_json(out / 'sources.json', {**corpus.hashes, 'trainer': digest(Path(__file__)),
                                     'memory': digest(Path(__file__).with_name('latent_memory.py')),
                                     'gpu_gmm': digest(Path(__file__).with_name('gpu_gmm.py'))})
    started = time.time()
    seen, metrics = set(), []
    cap, per_class = (256, 6) if args.smoke else (0, 0)
    try:
        for task in range(args.tasks):
            taskdir = out / f'task_{task}'
            taskdir.mkdir()
            train_ids = corpus.subset(corpus.parts[task]['train'], per_class)
            seen.update(corpus.label(sid) for sid in train_ids)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
            labels = [corpus.label(sid) for sid in train_ids]
            counts = {y: labels.count(y) for y in set(labels)}
            weights = torch.tensor([1 / counts[y] for y in labels], dtype=torch.double)
            replay_count = 0
            with (taskdir / 'training.jsonl').open('w', buffering=1) as log:
                for epoch in range(args.epochs):
                    model.train()
                    draws = torch.multinomial(weights, len(train_ids), replacement=True).tolist()
                    # Spread a fixed number of replay-only updates over the current epoch.
                    schedule = [(False, train_ids[i]) for i in draws]
                    if task:
                        schedule += [(True, None)] * args.replay_updates
                        random.shuffle(schedule)
                    losses = []
                    for replay, sid in schedule:
                        if replay:
                            x, y, old_task = memory.sample()
                            if old_task >= task:
                                raise RuntimeError('Replay from current/future task')
                            x = torch.from_numpy(x).cuda()
                            replay_count += 1
                        else:
                            x, y = corpus.load(sid, cap).cuda(), corpus.label(sid)
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(x, seen)
                        loss = nn.functional.cross_entropy(logits[None], torch.tensor([y], device='cuda'))
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    row = {'task': task, 'epoch': epoch, 'loss': float(np.mean(losses)),
                           'replay_updates_cumulative': replay_count, 'time': time.time()}
                    log.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
            model.eval()
            evaluation = []
            # Smoke uses validation only; formal fixed-epoch evaluation uses test.
            for past in range(task + 1):
                part = 'val' if (args.smoke or args.fit_probe) else 'test'
                ids = corpus.subset(corpus.parts[past][part], per_class)
                evaluation.append(evaluate(model, corpus, ids, seen, cap))
            metrics.append(evaluation)
            write_json(taskdir / 'evaluation.json', evaluation)
            def bags():
                with torch.no_grad():
                    for sid in train_ids:
                        x = corpus.load(sid, cap)
                        _, attention = model(x.cuda(), seen)
                        yield corpus.label(sid), x.numpy(), attention.cpu().numpy()
            print(json.dumps({'phase': 'gmm_fit_start', 'task': task, 'time': time.time()}), flush=True)
            fit_start = time.time()
            memory.add_episode(task, bags())
            fit_seconds = time.time() - fit_start
            stored_bytes = memory.save(taskdir / 'memory')
            expected = args.epochs * args.replay_updates if task else 0
            if replay_count != expected:
                raise RuntimeError('Replay count mismatch')
            write_json(taskdir / 'audit.json', {
                'task': task, 'smoke': args.smoke, 'fit_probe': args.fit_probe, 'epochs': args.epochs,
                'replay_updates': replay_count, 'expected_replay_updates': expected,
                'train_slides': len(train_ids), 'train_ids': train_ids,
                'feature_dimension': 1024, 'smoke_patch_cap': cap,
                'memory_parameter_bytes': memory.parameter_bytes, 'memory_file_bytes': stored_bytes,
                'fit_seconds': fit_seconds, 'seen_classes': sorted(seen),
                'class_incremental_no_task_identity': True, 'patient_split_audited': True})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'task': task, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                        'numpy_rng': np.random.get_state()}, taskdir / 'checkpoint.pt')
            print(json.dumps({'phase': 'task_complete', 'task': task, 'fit_seconds': fit_seconds}), flush=True)
        write_json(out / 'complete.json', {'smoke': args.smoke, 'fit_probe': args.fit_probe, 'tasks': args.tasks,
                                           'seconds': time.time() - started, 'metrics': metrics})
    except Exception:
        write_json(out / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('manifest', 'features', 'splits', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tasks', type=int, default=5, choices=range(1, 6))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--replay-updates', type=int, default=78)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--fit-backend', choices=('sklearn', 'cuda'), default='sklearn')
    p.add_argument('--gmm-max-iter', type=int, default=500)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--fit-probe', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
