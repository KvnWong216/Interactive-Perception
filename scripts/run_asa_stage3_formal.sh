#!/usr/bin/env bash
# End-to-end Formal ASA Stage-3 Embodiment Adaptation & ActiveArena Baseline Pipeline:
#   [1/3] Collect 120 zero-leakage Astribot S1 expert episodes (40 x 3 active-perception tasks)
#   [2/3] Train ASA Stage-3 (`EgoActiveFlowModelv2` initialized from `asa-human-stage2`, 1500 steps, eff BS=32)
#   [3/3] Evaluate trained `best/` checkpoint on official held-out ActiveArena test seeds (`instruction` + `oracle_subtask`)
set -euo pipefail

REPO_ROOT="/home/ubuntu/ygwang/IP-VLA/ip-vla"
BS_ROOT="/home/ubuntu/ygwang/IP-VLA/baseline-study"
SIM_PY="${BS_ROOT}/.venvs/activearena-sim/bin/python"
ASA_PY="/home/ubuntu/ygwang/IP-VLA/asa-iteration/.venv/bin/python"
YGB_PY="/home/ubuntu/miniforge3/envs/ygbench/bin/python"

DATA_DIR="${REPO_ROOT}/runs/asa_stage3_astribot/dataset_v1"
CKPT_DIR="${REPO_ROOT}/runs/asa_stage3_astribot/ckpt_v1"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="disabled"
export HF_HUB_OFFLINE="1"
export ACTIVEARENA_DEBUG_TIMING="0"
export CC="${BS_ROOT}/.venvs/build-tools/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${BS_ROOT}/.venvs/build-tools/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="${BS_ROOT}/env/cuda"
export ACTIVEARENA_CUDA_HOME="${CUDA_HOME}"
export ROBOTWIN_CUDA_HOME="${CUDA_HOME}"
export LD_LIBRARY_PATH="${BS_ROOT}/env/cxx-runtime:${LD_LIBRARY_PATH:-}"
export PATH="/home/ubuntu/miniforge3/envs/ygbench/bin:${PATH}"

echo "=================================================================="
echo "[1/3] Collecting zero-leakage Astribot S1 expert episodes -> ${DATA_DIR}"
echo "=================================================================="
"${SIM_PY}" -u "${REPO_ROOT}/scripts/collect_asa_astribot_data.py" \
  --output-dir "${DATA_DIR}" \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes-per-task "${EPISODES_PER_TASK:-30}" \
  --start-seed 500000 \
  --save-freq 15

echo "=================================================================="
echo "[2/3] Running formal ASA Stage-3 Robot Finetuning -> ${CKPT_DIR}"
echo "=================================================================="
"${ASA_PY}" -u "${REPO_ROOT}/scripts/train_asa_astribot_stage3.py" \
  --data-root "${DATA_DIR}" \
  --output-dir "${CKPT_DIR}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --lr "${LR:-2e-5}" \
  --max-steps "${MAX_STEPS:-1500}" \
  --eval-every "${EVAL_EVERY:-250}" \
  --log-every 10 \
  --tune-mm-mlp

echo "=================================================================="
echo "[3/3] Evaluating trained ASA Stage-3 baseline on official ActiveArena ID test seeds"
echo "=================================================================="
cd "${REPO_ROOT}"
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode asa_baseline \
  --asa-ckpt "${CKPT_DIR}/best" \
  --asa-subtask-mode instruction \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "ALL_ASA_STAGE3_PIPELINE_COMPLETE!"
