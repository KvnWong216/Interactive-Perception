#!/usr/bin/env bash
# Collect the versioned OPEN_AND_OBSERVE physical-effect data and always stop
# the local pi0.5 server on exit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
PORT="${PORT:-8000}"
MODE="${1:-smoke}"
EXPERIMENT_GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
EXPERIMENT_ALLOW_LOCAL_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-1}"
SERVER_PID=""

case "$MODE" in
  smoke|extension|v10-clean|v11-clean|v12b-clean|v12b-audit) ;;
  *) echo "usage: $0 [smoke|extension|v10-clean|v11-clean|v12b-clean|v12b-audit]" >&2; exit 2 ;;
esac

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping pi0.5 server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

extra=()
dataset="$REPO/data/calibration/t01_open_and_observe_effect_v4_extension.jsonl"
if [ "$MODE" = "smoke" ]; then
  extra=(--smoke)
  dataset="$REPO/data/calibration/t01_open_and_observe_effect_v4_smoke.jsonl"
elif [ "$MODE" = "extension" ]; then
  artifact="${OPEN_AND_OBSERVE_ARTIFACT:-$REPO/results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_visual.json}"
  [ -f "$artifact" ] || {
    echo "missing frozen v9 candidate for clean extension: $artifact" >&2
    exit 1
  }
  extra=(--extension --artifact "$artifact")
elif [ "$MODE" = "v10-clean" ]; then
  artifact="${OPEN_AND_OBSERVE_ARTIFACT:-$REPO/results/calibration/t01_open_and_observe_outcome_v10_composite_candidate.json}"
  [ -f "$artifact" ] || {
    echo "missing frozen v10 candidate for clean development: $artifact" >&2
    exit 1
  }
  extra=(--v10-clean --artifact "$artifact")
  dataset="$REPO/data/calibration/t01_open_and_observe_effect_v10_clean.jsonl"
elif [ "$MODE" = "v11-clean" ]; then
  artifact="${OPEN_AND_OBSERVE_ARTIFACT:-$REPO/results/calibration/t01_open_and_observe_outcome_v11_composite_candidate.json}"
  [ -f "$artifact" ] || {
    echo "missing frozen v11 candidate for clean development: $artifact" >&2
    exit 1
  }
  extra=(--v11-clean --artifact "$artifact")
  dataset="$REPO/data/calibration/t01_open_and_observe_effect_v11_clean.jsonl"
elif [ "$MODE" = "v12b-clean" ]; then
  artifact="${OPEN_AND_OBSERVE_ARTIFACT:-$REPO/results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json}"
  [ -f "$artifact" ] || {
    echo "missing frozen v12b candidate for clean development: $artifact" >&2
    exit 1
  }
  extra=(--v12b-clean --artifact "$artifact")
  dataset="$REPO/data/calibration/t01_open_and_observe_effect_v12b_clean.jsonl"
else
  artifact="${OPEN_AND_OBSERVE_ARTIFACT:-$REPO/results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json}"
  authorization="$REPO/results/calibration/t01_open_and_observe_v12b_sealed_authorization.json"
  [ -f "$artifact" ] || {
    echo "missing frozen v12b candidate for sealed audit: $artifact" >&2
    exit 1
  }
  [ -f "$authorization" ] || {
    echo "missing frozen v12b sealed authorization: $authorization" >&2
    exit 1
  }
  "$IPU_PYTHON" "$REPO/scripts/verify_t01_v12b_sealed_authorization.py"
  extra=(--v12b-audit --artifact "$artifact")
  dataset="$REPO/data/calibration/t01_open_and_observe_effect_v12b_sealed_audit.jsonl"
fi

manifest="${dataset%.jsonl}.manifest.json"
if [ -f "$manifest" ]; then
  "$IPU_PYTHON" "$REPO/scripts/verify_dataset_manifest.py" "$manifest"
  exit 0
fi

mkdir -p "$REPO/outputs/logs"
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" \
  EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$EXPERIMENT_ALLOW_LOCAL_RUSTDESK" \
  bash "$REPO/scripts/check_gpu_preflight.sh"
EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_INDEX" \
  PORT="$PORT" bash "$REPO/scripts/serve_pi05.sh" \
  >"$REPO/outputs/logs/t01_open_and_observe_server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 180))
while ! "$IPU_PYTHON" -c \
  "import socket; socket.create_connection(('127.0.0.1', int('$PORT')), timeout=1).close()" \
  >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    set +e
    wait "$SERVER_PID"
    server_status=$?
    set -e
    echo "pi0.5 server exited during startup (status $server_status)" >&2
    tail -n 60 "$REPO/outputs/logs/t01_open_and_observe_server.log" >&2
    exit "$server_status"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "pi0.5 server did not become ready" >&2
    exit 1
  fi
  sleep 1
done

resume=()
[ -f "$dataset" ] && resume=(--resume)
if [ "$MODE" = "smoke" ]; then
  env -u PYTHONPATH NUMBA_DISABLE_JIT=1 "$IPU_PYTHON" \
    "$REPO/scripts/collect_t01_open_and_observe_effect.py" \
    "${extra[@]}" "${resume[@]}" --port "$PORT"
else
  env -u PYTHONPATH NUMBA_DISABLE_JIT=1 "$IPU_PYTHON" \
    "$REPO/scripts/collect_t01_open_and_observe_effect.py" \
    "${extra[@]}" "${resume[@]}" --fresh-policy-server --port "$PORT"
fi

"$IPU_PYTHON" "$REPO/scripts/verify_dataset_manifest.py" "$manifest"
