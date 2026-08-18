#!/usr/bin/env bash
# Start the frozen action+prefix server, run the T01 observability loop, stop it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
PORT="${PORT:-8000}"
EXPERIMENT_GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
EXPERIMENT_ALLOW_LOCAL_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-1}"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for artifact in \
  results/calibration/prompt_state_belief_t01_v1.json \
  results/calibration/t01_action_outcome_critic_v1.json \
  results/calibration/t01_action_outcome_critic_audit_v1.json \
  results/calibration/t01_action_effect_v2.json \
  results/calibration/t01_observability_loss_contract_v1.json; do
  [ -f "$REPO/$artifact" ] || { echo "missing frozen artifact: $artifact" >&2; exit 1; }
done

OUTPUT="$REPO/results/capability/t01_outcome_closed_loop_100seed_v1.json"
[ ! -e "$OUTPUT" ] || { echo "immutable result already exists: $OUTPUT" >&2; exit 1; }
mkdir -p "$REPO/outputs/logs"
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" \
  EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$EXPERIMENT_ALLOW_LOCAL_RUSTDESK" \
  bash "$REPO/scripts/check_gpu_preflight.sh"
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_INDEX" \
  PORT="$PORT" bash "$REPO/scripts/serve_pi05_with_prefix.sh" \
  >"$REPO/outputs/logs/t01_outcome_closed_loop_server.log" 2>&1 &
SERVER_PID=$!
"$IPU_PYTHON" -c "import socket,time
deadline=time.time()+180
while time.time()<deadline:
    try:
        with socket.create_connection(('127.0.0.1', int('$PORT')), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('extended pi0.5 server did not become ready')"

cd "$REPO"
env -u PYTHONPATH NUMBA_DISABLE_JIT=1 "$IPU_PYTHON" \
  scripts/run_t01_outcome_closed_loop.py \
  --port "$PORT" --output "$OUTPUT"
