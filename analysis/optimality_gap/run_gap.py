"""Recompute the 60 restricted instances of the optimality-gap audit (Appendix, optimality gaps).

Eight slides per (fold, subset) pair (four per class, fixed in per_instance.csv), three prefix options per slide (64/128/256),
budgets 256/384/512/768, exact enumeration versus KEM's allocator and two reference greedy solvers.

usage: python analysis/optimality_gap/run_gap.py --cache-root CACHE_DIR --run-root OUT_DIR
CACHE_DIR holds the task-0 candidate archives fold{0..4}_task0_*.pt that the training run writes when
`unified_candidate_cache: true` (clam_sb__kem, seed 1, one archive per fold). Output: OUT_DIR/reports/FACILITY_GAP/.
"""
import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import benchmark_facility_gap as gap  # noqa: E402


def choose_cache(cache_root, fold):
    expected = dict(fold=fold, seed=1, net='clam_sb', feature_extractor='UNI', patch_budget=5000, selection_rule='logdet',
                    projection_dim=64, signature_mode='mean_std', signature_seed=1701, trajectory_mode='a',
                    class_conditioned=True, assignment_weighted=True)
    for path in sorted(Path(cache_root).glob(f'fold{fold}_task0_*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        fields = payload.get('cache_fields', {})
        if all(fields.get(k) == v for k, v in expected.items()):
            return path, payload
    raise RuntimeError(f'no matching log-det candidate archive for fold {fold}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--instances', type=Path, default=HERE / 'per_instance.csv')
    a = p.parse_args()
    rows = list(csv.DictReader(a.instances.open()))
    gap.fixed_subsets = {(int(r['fold']), int(r['subset_seed'])): json.loads(r['slides']) for r in rows}
    assert len(gap.fixed_subsets) == 15
    gap.choose_cache = choose_cache
    anchor = """            for label in labels:
                group = sorted([c for c in pool if c.label == label], key=lambda c: c.slide_id)
                selected.extend(group[i] for i in rng.choice(len(group), 4, replace=False))
"""
    replacement = """            by_id = {c.slide_id: c for c in pool}
            selected = [by_id[s] for s in fixed_subsets[fold, subset_seed]]
            assert len(selected) == 8 and all(sum(c.label == label for c in selected) == 4 for label in labels)
"""
    source = inspect.getsource(gap.main)
    assert source.count(anchor) == 1
    exec(compile(source.replace(anchor, replacement), 'gaps_main_paired.py', 'exec'), gap.__dict__)
    sys.argv = ['benchmark_facility_gap.py', '--run-root', str(a.run_root), '--cache-root', str(a.cache_root)]
    gap.main()


if __name__ == '__main__':
    main()
