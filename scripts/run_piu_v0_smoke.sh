#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
EXPERIMENT_GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
EXPERIMENT_ALLOW_LOCAL_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-1}"
PORT="${PORT:-8000}"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$REPO"
env -u PYTHONPATH "$IPU_PYTHON" -m pytest -q
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" \
  EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$EXPERIMENT_ALLOW_LOCAL_RUSTDESK" \
  bash scripts/check_gpu_preflight.sh
mkdir -p outputs/logs
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_INDEX" \
  PORT="$PORT" bash scripts/serve_pi05_with_prefix.sh \
  >outputs/logs/piu_v0_server.log 2>&1 &
SERVER_PID=$!
"$IPU_PYTHON" -c "import socket,time
deadline=time.time()+180
while time.time()<deadline:
    try:
        with socket.create_connection(('127.0.0.1', int('$PORT')), timeout=1): break
    except OSError: time.sleep(1)
else: raise SystemExit('pi0.5 prefix server did not become ready')"
env -u PYTHONPATH NUMBA_DISABLE_JIT=1 "$IPU_PYTHON" \
  scripts/run_piu_v0_smoke.py --port "$PORT" "$@"
