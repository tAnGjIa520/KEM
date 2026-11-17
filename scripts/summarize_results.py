"""Average the per-fold result files of a method into AACC and F (final accuracy / forgetting).

usage: python scripts/summarize_results.py LOG_ROOT clam_sb__kem
Reads LOG_ROOT/<preset>_fold<f>_seed<s>/fold_<f>_results.csv.
AACC = mean over tasks of the final accuracy; F = mean over the first four tasks of
(best accuracy after learning the task - final accuracy).
"""
import csv
import glob
import os
import sys

import numpy as np


def metrics(path, n_tasks=5):
    rows = list(csv.DictReader(open(path)))
    acc = np.array([[float(r[f'{j}_acc'] or 0) for j in range(n_tasks)] for r in rows])
    aacc = acc[-1].mean()
    forgetting = np.mean([acc[j:, j].max() - acc[-1, j] for j in range(n_tasks - 1)])
    return aacc, forgetting


def main():
    root, preset = sys.argv[1], sys.argv[2]
    per_seed = {}
    for path in sorted(glob.glob(os.path.join(root, f'{preset}_fold*_seed*', 'fold_*_results.csv'))):
        seed = int(path.split('_seed')[1].split(os.sep)[0])
        per_seed.setdefault(seed, []).append(metrics(path))
    seed_means = []
    for seed, vals in sorted(per_seed.items()):
        m = np.mean(vals, axis=0)
        seed_means.append(m)
        print(f'seed {seed}: {len(vals)} folds  AACC {m[0]:.4f}  F {m[1]:.4f}')
    if len(seed_means) > 1:
        arr = np.array(seed_means)
        print(f'mean over seeds: AACC {arr[:, 0].mean():.4f} +- {arr[:, 0].std(ddof=1):.4f}  '
              f'F {arr[:, 1].mean():.4f} +- {arr[:, 1].std(ddof=1):.4f}')


if __name__ == '__main__':
    main()
