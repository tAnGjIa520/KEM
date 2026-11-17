#!/usr/bin/env bash
# Run one method over all folds and seeds, sequentially on the visible GPU.
# usage: FEATURE_ROOT=... LOG_ROOT=... bash scripts/run_all_folds.sh clam_sb__kem "1 2 3" "0 1 2 3 4"
set -euo pipefail
PRESET=${1:?preset name, e.g. clam_sb__kem}
SEEDS=${2:-"1 2 3"}
FOLDS=${3:-"0 1 2 3 4"}
: "${FEATURE_ROOT:?set FEATURE_ROOT}" "${LOG_ROOT:?set LOG_ROOT}"
cd "$(dirname "$0")/.."
for seed in $SEEDS; do
  for fold in $FOLDS; do
    python scripts/render_preset.py --preset "$PRESET" --seed "$seed" --fold "$fold" \
      --feature-root "$FEATURE_ROOT" --log-root "$LOG_ROOT" --run
  done
done
