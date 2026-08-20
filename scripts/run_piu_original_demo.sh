#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?usage: scripts/run_piu_original_demo.sh RUN_ID}"
GPU="${EXPERIMENT_GPU_INDEX:-0}"
ALLOW_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-0}"
PY_QWEN="${PIU_QWEN_PYTHON:-$ROOT/../openpi/.venv/bin/python}"
PY_SIM="${PIU_SIM_PYTHON:-$ROOT/../.conda/envs/ipu/bin/python}"
PORT="${PIU_POLICY_PORT:-8014}"

INITIAL_REPORT="$ROOT/results/diagnostics/${RUN_ID}_initial.json"
INITIAL_ASSETS="$ROOT/results/assets/${RUN_ID}_initial"
OPTION_REPORT="$ROOT/results/diagnostics/${RUN_ID}_fresh_option.json"
OPTION_ASSETS="$ROOT/results/assets/${RUN_ID}_fresh_option"
OPTION_WORK="$ROOT/outputs/${RUN_ID}_fresh_option"
OUTCOME_REPORT="$ROOT/results/diagnostics/${RUN_ID}_outcome_v13.json"
POST_REPORT="$ROOT/results/diagnostics/${RUN_ID}_post.json"
POST_ASSETS="$ROOT/results/assets/${RUN_ID}_post"
DEMO_DIR="$ROOT/results/demos/${RUN_ID}"

for path in \
  "$INITIAL_REPORT" "$INITIAL_ASSETS" "$OPTION_REPORT" "$OPTION_ASSETS" \
  "$OPTION_WORK" "$OUTCOME_REPORT" "$POST_REPORT" "$POST_ASSETS" "$DEMO_DIR"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite immutable run path: $path" >&2
    exit 1
  fi
done

cd "$ROOT"
EXPERIMENT_GPU_INDEX="$GPU" EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$ALLOW_RUSTDESK" \
  bash scripts/check_gpu_preflight.sh
CUDA_VISIBLE_DEVICES="$GPU" "$PY_QWEN" scripts/run_piu_qwen_observation_pipeline.py \
  --agentview results/assets/t01_easy_handoff_states_seed1399_v1/closed_easy_agentview.png \
  --wrist results/assets/t01_easy_handoff_states_seed1399_v1/closed_easy_wrist.png \
  --prompt "Place the butter in the basket" \
  --device cuda \
  --asset-dir "$INITIAL_ASSETS" \
  --output "$INITIAL_REPORT"

CUDA_VISIBLE_DEVICES="$GPU" "$PY_SIM" scripts/run_piu_selected_open_fresh.py \
  --initial-report "$INITIAL_REPORT" \
  --bddl scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl \
  --initial-state-npz data/development/t01_easy_handoff_seed1399_v1.npz \
  --initial-state-key closed_easy \
  --outcome-composite results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json \
  --seed 1399 --open-steps 300 --replan-steps 5 --gpu "$GPU" --port "$PORT" \
  --asset-dir "$OPTION_ASSETS" --work-dir "$OPTION_WORK" --output "$OPTION_REPORT"

"$PY_SIM" scripts/score_public_rgb_option_outcome.py \
  --option-report "$OPTION_REPORT" \
  --composite results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json \
  --prompt "Place the butter in the basket" \
  --fusion v13-complementary \
  --output "$OUTCOME_REPORT"

EXPERIMENT_GPU_INDEX="$GPU" EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$ALLOW_RUSTDESK" \
  bash scripts/check_gpu_preflight.sh
CUDA_VISIBLE_DEVICES="$GPU" "$PY_QWEN" scripts/run_piu_qwen_observation_pipeline.py \
  --agentview "$OPTION_ASSETS/public_keyframes/05_returned_agentview.png" \
  --wrist "$OPTION_ASSETS/public_keyframes/05_returned_wrist.png" \
  --prompt "Place the butter in the basket" \
  --previous-report "$INITIAL_REPORT" \
  --executed-action OPEN_CONTAINER \
  --observed-outcome EVIDENCE_ACQUIRED \
  --device cuda \
  --asset-dir "$POST_ASSETS" \
  --output "$POST_REPORT"

"$PY_SIM" scripts/render_piu_information_demo.py \
  --initial-report "$INITIAL_REPORT" \
  --option-report "$OPTION_REPORT" \
  --outcome-report "$OUTCOME_REPORT" \
  --post-report "$POST_REPORT" \
  --output-dir "$DEMO_DIR"

echo "completed PIU original-scene development run: $RUN_ID"
