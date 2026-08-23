#!/usr/bin/env bash
set -euo pipefail

REPRO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${REPRO_SCRIPT_DIR}/.." && pwd)"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-${REPRO_ROOT}/.venv/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-${REPRO_ROOT}/../.conda/envs/ipu/bin/python}"

cd "${REPRO_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${SIM_PYTHON}" scripts/evaluation/relabel_original_drawer_open_target.py \
  --condition open_closed_drawer \
  --target-object cream_cheese_1 \
  --target-destination-region basket_1_contain_region \
  --output results/method/original_drawer_open_cream_relabel_v2.json \
  --force

"${SIM_PYTHON}" scripts/evaluation/relabel_original_drawer_open_target.py \
  --condition direct_visible_cream_cheese \
  --target-object cream_cheese_1 \
  --target-destination-region basket_1_contain_region \
  --output results/method/original_drawer_direct_cream_final_relabel_v2.json \
  --force

"${ANALYSIS_PYTHON}" scripts/evaluation/summarize_original_drawer_paper_cycle.py \
  --output results/method/original_drawer_paper_cycle_v2.json \
  --force

"${ANALYSIS_PYTHON}" scripts/data/build_executed_counterfactual_dataset.py \
  --task-specific-relabel \
    results/method/original_drawer_open_cream_relabel_v2.json \
  --direct-cream-relabel \
    results/method/original_drawer_direct_cream_final_relabel_v2.json \
  --output \
    data/calibrated_interaction/original_drawer_executed_v2/development.jsonl \
  --manifest \
    data/calibrated_interaction/original_drawer_executed_v2/development.manifest.json \
  --force

"${ANALYSIS_PYTHON}" scripts/training/evaluate_executed_effect_cv.py \
  --dataset \
    data/calibrated_interaction/original_drawer_executed_v2/development.jsonl \
  --manifest \
    data/calibrated_interaction/original_drawer_executed_v2/development.manifest.json \
  --candidates configs/experiments/original_drawer_candidate_set.yaml \
  --output results/method/original_drawer_executed_effect_cv_v1.json \
  --model checkpoints/calibrated_interaction/original_drawer_executed_effect_cv_v1.pt \
  --epochs 300 \
  --force

"${ANALYSIS_PYTHON}" scripts/visualization/render_paper_trace.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${ANALYSIS_PYTHON}" -m pytest -q
