#!/usr/bin/env bash
# End-to-End Formal AAWR (`Asymmetric Advantage-Weighted Regression`) ActiveArena Baseline Pipeline:
#   [1/3] Collect zero-leakage offline expert + suboptimal episodes with training-only privileged state `s^+_t`
#   [2/3] Train AAWR (privileged twin-Q + expectile-V critics, unprivileged visual-proprio actor; fits on 8 GB RTX 4060!)
#   [3/3] Evaluate unprivileged actor `best/` on official held-out ActiveArena test seeds (0 privileged info at eval!)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BS_ROOT="${BASELINE_STUDY_ROOT:-/home/ubuntu/ygwang/IP-VLA/baseline-study}"
SIM_PY="${ACTIVEARENA_PYTHON:-${BS_ROOT}/.venvs/activearena-sim/bin/python}"
TRAIN_PY="${IPVLA_PLANNER_PYTHON:-/home/ubuntu/miniforge3/envs/ygbench/bin/python}"

DATA_DIR="${AAWR_DATA_DIR:-${REPO_ROOT}/runs/aawr_astribot/dataset_v1}"
CKPT_DIR="${AAWR_CKPT_DIR:-${REPO_ROOT}/runs/aawr_astribot/ckpt_v1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="disabled"
export HF_HUB_OFFLINE="1"
if [[ -d "${BS_ROOT}/env/cuda" ]]; then
  export CC="${BS_ROOT}/.venvs/build-tools/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${BS_ROOT}/.venvs/build-tools/bin/x86_64-conda-linux-gnu-g++"
  export CUDA_HOME="${BS_ROOT}/env/cuda"
  export ACTIVEARENA_CUDA_HOME="${CUDA_HOME}"
  export ROBOTWIN_CUDA_HOME="${CUDA_HOME}"
  export LD_LIBRARY_PATH="${BS_ROOT}/env/cxx-runtime:${LD_LIBRARY_PATH:-}"
fi

echo "=================================================================="
echo "[1/3] Collecting offline expert + suboptimal privileged data -> ${DATA_DIR}"
echo "=================================================================="
"${SIM_PY}" -u "${REPO_ROOT}/scripts/collect_aawr_astribot_data.py" \
  --output-dir "${DATA_DIR}" \
  --ipvla-repo "${REPO_ROOT}" \
  --tasks ${TASKS:-check_block_color check_cola_date check_cola_color} \
  --episodes-per-task "${EPISODES_PER_TASK:-30}" \
  --start-seed 600000 \
  --save-freq 15

echo "=================================================================="
echo "[2/3] Training Asymmetric AWR (privileged Q/V -> unprivileged actor) -> ${CKPT_DIR}"
echo "=================================================================="
"${TRAIN_PY}" -u "${REPO_ROOT}/scripts/train_aawr_astribot.py" \
  --data-root "${DATA_DIR}" \
  --output-dir "${CKPT_DIR}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --max-steps "${MAX_STEPS:-4000}" \
  --expectile "${EXPECTILE:-0.9}" \
  --a-scaling "${A_SCALING:-3.0}"

echo "=================================================================="
echo "[3/3] Evaluating unprivileged AAWR actor on official ActiveArena ID test seeds"
echo "=================================================================="
cd "${REPO_ROOT}"
"${TRAIN_PY}" -u -m eval.activearena.run_eval \
  --mode aawr_baseline \
  --aawr-ckpt "${CKPT_DIR}/best" \
  --setting "${SETTING:-id}" \
  --tasks ${TASKS:-check_block_color check_cola_date check_cola_color} \
  --episodes "${EVAL_EPISODES:-10}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "ALL_AAWR_BASELINE_PIPELINE_COMPLETE!"
