"""Fill a preset template with paths, seed and fold, and print the training command.

usage:
  python scripts/render_preset.py --preset clam_sb__kem --seed 1 --fold 0 \
      --feature-root $FEATURE_ROOT --log-root $LOG_ROOT [--set buffer_patch_budget=10000] [--run]
"""
import argparse
import os
import subprocess
import sys

import yaml

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', required=True, help='template name in configs/presets, e.g. clam_sb__kem')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--feature-root', required=True, help='directory containing CAMELYON/UNI/..., TCGA/<cohort>/UNI/...')
    p.add_argument('--log-root', required=True, help='output directory for runs, buffer snapshots and caches')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a preset key (YAML value)')
    p.add_argument('--out', default=None, help='where to write the rendered preset (default: LOG_ROOT/presets)')
    p.add_argument('--run', action='store_true', help='launch main_cl.py after rendering')
    a = p.parse_args()

    name = f'{a.preset}_fold{a.fold}_seed{a.seed}'
    text = open(os.path.join(CODE_ROOT, 'configs', 'presets', f'{a.preset}.yaml')).read()
    for key, val in {
        'CODE_ROOT': CODE_ROOT, 'FEATURE_ROOT': os.path.abspath(a.feature_root), 'LOG_ROOT': os.path.abspath(a.log_root),
        'EXP_NAME': name, 'SEED': str(a.seed), 'FOLD': str(a.fold), 'FOLD_END': str(a.fold + 1),
    }.items():
        text = text.replace('{{' + key + '}}', val)
    config = yaml.safe_load(text)
    for key in ('seed', 'folds_start', 'folds_end'):
        config[key] = int(config[key])
    for item in a.set:
        key, value = item.split('=', 1)
        config[key] = yaml.safe_load(value)
    assert '{{' not in yaml.safe_dump(config), 'unresolved placeholder'

    out_dir = a.out or os.path.join(a.log_root, 'presets')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=True)
    cmd = [sys.executable, 'main_cl.py', '--preset', path]
    print(' '.join(cmd))
    if a.run:
        subprocess.run(cmd, cwd=CODE_ROOT, check=True)


if __name__ == '__main__':
    main()
