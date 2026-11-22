#!/usr/bin/env bash
# One-command Stage-2 ActiveArena closed-loop evaluation wrapper.
# Examples:
#   bash scripts/eval_stage2_activearena.sh --mode oracle_subtask_oft --tasks check_block_color --episodes 5 --dry-run
#   bash scripts/eval_stage2_activearena.sh --mode ipvla_planner --setting ood --tasks check_block_color check_cola_date --episodes 10
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
exec python3 -m eval.activearena.run_eval "$@"
