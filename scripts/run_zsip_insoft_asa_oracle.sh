#!/usr/bin/env bash
# Sequential GPU-0 Formal Pipeline (Zero Contention, Pinocchio-Calibrated IK):
#   [1/5] Calibrated ZS-IP v2 Formal Eval (3 tasks x 10 official ID seeds = 30 episodes, 0 training)
#   [2/5] Official End-to-End Baseline `instruction_oft` (`oft_instruction_action_12_ws`, 30 episodes)
#   [3/5] Autonomous ASA Stage-3 Formal Eval with Pinocchio IK (`ckpt_v1/best`, `instruction`, 30 episodes)
#   [4/5] ASA + Ground-Truth Subtask Recipe Stage-3 Training (`--recipe-mode subtask_only`, 1,200 steps)
#   [5/5] ASA + Ground-Truth Subtask Oracle Upper-Bound Eval (`ckpt_oracle_subtask_v1/best`, `oracle_subtask`, 30 episodes)
set -euo pipefail

export XDG_RUNTIME_DIR="/run/user/1000"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"

REPO_ROOT="/home/ubuntu/ygwang/IP-VLA/ip-vla"
BS_ROOT="/home/ubuntu/ygwang/IP-VLA/baseline-study"
ASA_PY="/home/ubuntu/ygwang/IP-VLA/asa-iteration/.venv/bin/python"
YGB_PY="/home/ubuntu/miniforge3/envs/ygbench/bin/python"

DATA_DIR="${REPO_ROOT}/runs/asa_stage3_astribot/dataset_v1"
AUTO_CKPT_DIR="${REPO_ROOT}/runs/asa_stage3_astribot/ckpt_v1"
ORACLE_CKPT_DIR="${REPO_ROOT}/runs/asa_stage3_astribot/ckpt_oracle_subtask_v1"

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

cd "${REPO_ROOT}"

echo "=================================================================="
echo "[1/5] Calibrated ZS-IP Formal Evaluation (3 Tasks x 10 ID Seeds = 30 Episodes)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode zsip_baseline \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --zsip-port 19300 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "=================================================================="
echo "[2/5] Official End-to-End Baseline instruction_oft (3 Tasks x 10 ID Seeds = 30 Episodes)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode instruction_oft \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --action-port 19005 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "=================================================================="
echo "[3/5] Autonomous ASA Stage-3 Re-Eval with Pinocchio IK (3 Tasks x 10 ID Seeds = 30 Episodes)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode asa_baseline \
  --asa-ckpt "${AUTO_CKPT_DIR}/best" \
  --asa-subtask-mode instruction \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --asa-port 19204 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "=================================================================="
echo "[4/5] ASA + Ground-Truth Subtask Recipe Stage-3 Training -> ${ORACLE_CKPT_DIR}"
echo "=================================================================="
"${ASA_PY}" -u "${REPO_ROOT}/scripts/train_asa_astribot_stage3.py" \
  --data-root "${DATA_DIR}" \
  --output-dir "${ORACLE_CKPT_DIR}" \
  --recipe-mode subtask_only \
  --batch-size "${BATCH_SIZE:-8}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --lr "${LR:-2e-5}" \
  --max-steps "${MAX_STEPS:-1200}" \
  --eval-every "${EVAL_EVERY:-200}" \
  --log-every 10 \
  --tune-mm-mlp

echo "=================================================================="
echo "[5/5] ASA + Ground-Truth Subtask Oracle Upper-Bound Evaluation (30 Episodes)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode asa_baseline \
  --asa-ckpt "${ORACLE_CKPT_DIR}/best" \
  --asa-subtask-mode oracle_subtask \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --asa-port 19205 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "ALL_FIVE_EXPERIMENTS_COMPLETE!"
