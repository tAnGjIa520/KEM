"""Independent COSFormer retrieval-adapted MorphCL training.

Smoke/pilot are validation-only. Legacy CONCH caches require explicit acceptance
and are always reported as provenance-unverified. No imports from KEM.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from data import Corpus
from memory import Entry, ReplayMemory, replay_loss
from model import COSFormer
from prepare_words import DIAGNOSES
from retrieval import select_representatives


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selection(corpus, ids, cache, checkpoint, seed, cap):
    from conch.open_clip_custom import create_model_from_pretrained
    from conch.open_clip_custom.custom_tokenizer import get_tokenizer, tokenize
    encoder, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint, device='cuda')
    encoder.eval()
    prompts = [f'a histopathology image of {name}.' for name in DIAGNOSES]
    with torch.no_grad():
        text = encoder.encode_text(tokenize(texts=prompts, tokenizer=get_tokenizer()).cuda()).cpu().numpy()
    def bags():
        with torch.no_grad():
            for sid in ids:
                path = Path(cache)/corpus.conch_relative(sid)
                with h5py.File(path) as handle, h5py.File(corpus.uni_h5(sid)) as uni:
                    if not np.array_equal(handle['coords'][:], uni['coords'][:]):
                        raise ValueError(f'CONCH/UNI coordinate mismatch: {sid}')
                    raw = handle['features'][:cap] if cap else handle['features'][:]
                    x = torch.from_numpy(raw).float().cuda()
                projected = F.normalize(encoder.visual.forward_project(x), dim=-1)
                yield sid, corpus.label(sid), projected.cpu().numpy()
    selected, audit = select_representatives(bags(), text, ids, seed=seed)
    del encoder
    torch.cuda.empty_cache()
    return selected, audit, prompts


def evaluate(model, corpus, ids, sequences, cap):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sid in ids:
            scores = model.class_scores(corpus.load(sid, cap).cuda(), sequences)
            pred = max(scores, key=lambda key: float(scores[key]))
            predictions.append({'slide_id': sid, 'label': corpus.label(sid), 'prediction': pred,
                                'scores': {str(k): float(v) for k, v in scores.items()}})
    return {'accuracy': sum(r['label'] == r['prediction'] for r in predictions)/len(predictions),
            'slides': len(predictions), 'predictions': predictions}


def run(a):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly the authorized CUDA GPU')
    if not a.smoke and not a.accept_legacy_conch:
        raise RuntimeError('Explicit acceptance of unverified legacy CONCH caches required')
    if a.epochs < 1 or a.replay_updates < 0:
        raise ValueError('Invalid training schedule')
    cap = 256 if a.smoke else 0
    per_class = 6 if a.smoke or a.pilot else 0
    evaluation_split = 'val' if a.smoke or a.pilot else 'test'
    if not a.smoke and not a.pilot and a.tasks != 5:
        raise ValueError('Formal run requires all five tasks')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(a.threads)
    corpus = Corpus(a.manifest, a.features, a.splits, a.fold)
    word_root = Path(a.words)
    word_meta = json.loads((word_root/'metadata.json').read_text())
    sequences = {int(k): v for k, v in word_meta['class_sequences'].items()}
    vectors = torch.load(word_root/'word_vectors.pt', weights_only=True)
    model = COSFormer(vectors).cuda()
    memory = ReplayMemory(a.capacity, a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', vars(a))
    save(out/'sources.json', {**corpus.hashes, 'word_vectors': sha(word_root/'word_vectors.pt'),
         'conch_checkpoint': sha(a.conch_checkpoint),
         'source_code': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
         'historical_conch_provenance_verified': False,
         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    started = time.time()
    seen, metrics = set(), []
    try:
        for task in range(a.tasks):
            directory = out/f'task_{task}'
            directory.mkdir()
            ids = corpus.subset(corpus.parts[task]['train'], per_class)
            labels = {corpus.label(sid) for sid in ids}
            seen.update(labels)
            model.add_task([token for label in labels for token in sequences[label]])
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=2e-4, weight_decay=1e-5)
            counts = {label: sum(corpus.label(sid) == label for sid in ids) for label in labels}
            weights = torch.tensor([1/counts[corpus.label(sid)] for sid in ids], dtype=torch.double)
            replay_count = 0
            for epoch in range(a.epochs):
                model.train()
                schedule = [(False, ids[i]) for i in torch.multinomial(weights, len(ids), replacement=True)]
                if task:
                    # Formal/pilot budget is TOTAL updates per task, not per epoch.
                    updates = (a.replay_updates if a.smoke else
                               a.replay_updates//a.epochs + int(epoch < a.replay_updates % a.epochs))
                    schedule += [(True, None)]*updates
                    random.shuffle(schedule)
                losses = []
                for replay, sid in schedule:
                    optimizer.zero_grad(set_to_none=True)
                    if replay:
                        entry = memory.sample()
                        if entry.task >= task:
                            raise ValueError('Future/current replay')
                        loss = replay_loss(model, entry, 'cuda')
                        replay_count += 1
                    else:
                        seq = torch.tensor(sequences[corpus.label(sid)], device='cuda')
                        logits = model(corpus.load(sid, cap).cuda(), seq[:-1][None], task=task)[0]
                        loss = F.cross_entropy(logits, seq[1:])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise RuntimeError('Nonfinite gradients')
                    optimizer.step()
                    losses.append(float(loss.detach()))
                print(json.dumps({'phase': 'epoch_done', 'task': task, 'epoch': epoch,
                                  'loss': float(np.mean(losses)), 'replay_updates': replay_count,
                                  'time': time.time()}), flush=True)
            active = {label: sequences[label] for label in sorted(seen)}
            evaluation = [evaluate(model, corpus, corpus.subset(corpus.parts[t][evaluation_split], per_class), active, cap)
                          for t in range(task+1)]
            metrics.append(evaluation)
            save(directory/'evaluation.json', evaluation)
            selected, audit, prompts = selection(corpus, ids, a.conch_cache, a.conch_checkpoint, a.seed, cap)
            save(directory/'selection.json', {'selected': selected, 'audit': audit, 'prompts': prompts,
                                              'train_ids': ids, 'class_ids': sorted(labels)})
            entries = []
            model.eval()
            with torch.no_grad():
                columns = model.seen_tokens.nonzero().flatten().cpu()
                for sid in selected:
                    x = corpus.load(sid, cap)
                    seq = torch.tensor(sequences[corpus.label(sid)])
                    logits = model(x.cuda(), seq[:-1][None].cuda(), task=task)[0][:, columns.cuda()].cpu()
                    entries.append(Entry(sid, task, corpus.label(sid), x, seq, columns, logits))
            memory.update(task, entries)
            stored_bytes = memory.save(directory/'memory')
            expected = (a.epochs*a.replay_updates if a.smoke else a.replay_updates) if task else 0
            if replay_count != expected:
                raise ValueError('Replay updates mismatch')
            save(directory/'audit.json', {'smoke': a.smoke, 'pilot': a.pilot,
                 'evaluation_split': evaluation_split, 'task': task, 'epochs': a.epochs,
                 'replay_updates': replay_count, 'expected_replay_updates': expected,
                 'train_ids': ids, 'memory_slide_ids': [e.slide_id for e in memory.entries],
                 'memory_patches': sum(len(e.features) for e in memory.entries),
                 'memory_file_bytes': stored_bytes, 'memory_tensor_bytes': memory.tensor_bytes,
                 'patch_cap': cap, 'patient_split_audited': True,
                 'no_oracle_task_at_evaluation': True, 'historical_features_reloaded_for_replay': False,
                 'historical_conch_provenance_verified': False})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'task': task,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}, directory/'checkpoint.pt')
            print(json.dumps({'phase': 'task_complete', 'task': task, 'time': time.time()}), flush=True)
        save(out/'complete.json', {'smoke': a.smoke, 'pilot': a.pilot,
                                  'evaluation_split': evaluation_split,
                                  'historical_conch_provenance_verified': False,
                                  'tasks': a.tasks, 'seconds': time.time()-started,
                                  'metrics': metrics})
    except Exception:
        save(out/'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('manifest', 'features', 'splits', 'words', 'conch-cache', 'conch-checkpoint', 'output'):
        parser.add_argument('--'+key, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--pilot', action='store_true', help='Six slides/class, full bags, validation only')
    parser.add_argument('--accept-legacy-conch', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tasks', type=int, default=2, choices=range(1, 6))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--replay-updates', type=int, default=2)
    parser.add_argument('--capacity', type=int, default=26)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
