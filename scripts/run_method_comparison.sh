#!/usr/bin/env bash
# Run the three paper arms into separate directories. This script intentionally
# does not start the policy server or bypass the reproduction gate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/../.conda/envs/ipu/bin/python}"
PERCEPTION_ENDPOINT="${PERCEPTION_ENDPOINT:-}"
TASKS="${TASKS:-T01_multi_drawer_search T04_visible_direct T05_exhaustive_not_found T06_severe_clutter_occlusion}"
SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-implicit}"

if [ -z "$PERCEPTION_ENDPOINT" ]; then
  echo "PERCEPTION_ENDPOINT is required for the uncertainty-router arm" >&2
  exit 2
fi

for ARM in monolithic fixed-rule uncertainty-router; do
  ARGS=(
    "$PYTHON" "$REPO/scripts/run_challenge_rollout.py"
    --arm "$ARM"
    --task-ids $TASKS
    --seeds $SEEDS
    --variants $VARIANTS
    --output "$REPO/outputs/method_comparison/$ARM"
  )
  if [ "$ARM" = uncertainty-router ]; then
    ARGS+=(--perception-endpoint "$PERCEPTION_ENDPOINT")
  fi
  env -u PYTHONPATH "${ARGS[@]}"
done

env -u PYTHONPATH "$PYTHON" "$REPO/scripts/compare_method_arms.py" \
  "$REPO/outputs/method_comparison/monolithic/rollout_summary.json" \
  "$REPO/outputs/method_comparison/fixed-rule/rollout_summary.json" \
  "$REPO/outputs/method_comparison/uncertainty-router/rollout_summary.json" \
  --output "$REPO/outputs/method_comparison/comparison.json"
