#!/usr/bin/env bash
# Zero-Shot Interactive Perception (ZS-IP, arXiv:2602.18374) Matched-Budget Formal Pipeline:
#   - Waits until `asa-stage3-formal.service` finishes on GPU 0 (zero GPU contention!)
#   - [1/2] Runs 1-episode live smoke verification (`check_block_color`, `--episodes 1`)
#   - [2/2] Runs full matched-budget formal evaluation (`3 tasks x 10 official ID seeds = 30 episodes`,
#           100% frozen `Qwen3-VL-2B-Instruct`, 0 training steps, 0 simulator oracle)
set -euo pipefail

export XDG_RUNTIME_DIR="/run/user/1000"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"

REPO_ROOT="/home/ubuntu/ygwang/IP-VLA/ip-vla"
BS_ROOT="/home/ubuntu/ygwang/IP-VLA/baseline-study"
YGB_PY="/home/ubuntu/miniforge3/envs/ygbench/bin/python"

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

if systemctl --user is-active --quiet asa-stage3-formal.service; then
  echo "[zsip-formal] Waiting for asa-stage3-formal.service to finish on GPU ${CUDA_VISIBLE_DEVICES}..."
  while systemctl --user is-active --quiet asa-stage3-formal.service; do
    sleep 15
  done
  echo "[zsip-formal] asa-stage3-formal.service finished! Cooling down GPU for 5s..."
  sleep 5
fi

cd "${REPO_ROOT}"

echo "=================================================================="
echo "[1/2] ZS-IP 1-Episode Live Smoke Verification (0 Training, Frozen Qwen3-VL-2B)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode zsip_baseline \
  --setting id \
  --tasks check_block_color \
  --episodes 1 \
  --zsip-port 19300 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "=================================================================="
echo "[2/2] ZS-IP Matched-Budget Formal Evaluation (3 Tasks x 10 Official ID Seeds = 30 Episodes)"
echo "=================================================================="
"${YGB_PY}" -u -m eval.activearena.run_eval \
  --mode zsip_baseline \
  --setting id \
  --tasks check_block_color check_cola_date check_cola_color \
  --episodes "${EVAL_EPISODES:-10}" \
  --zsip-port 19300 \
  --gpu "${CUDA_VISIBLE_DEVICES}"

echo "ALL_ZSIP_FORMAL_EVAL_COMPLETE!"
