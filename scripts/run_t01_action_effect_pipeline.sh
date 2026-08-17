#!/usr/bin/env bash
# Reproduce the paired-RGB T01 action-effect calibration and frozen audit.
#
# Usage:
#   bash scripts/run_t01_action_effect_pipeline.sh development
#   bash scripts/run_t01_action_effect_pipeline.sh development-v2
#   bash scripts/run_t01_action_effect_pipeline.sh development-v3
#   bash scripts/run_t01_action_effect_pipeline.sh development-v4
#   bash scripts/run_t01_action_effect_pipeline.sh audit
#
# Existing datasets and frozen artifacts are verified and never overwritten.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
PORT="${PORT:-8000}"
MODE="${1:-development}"
SERVER_PID=""

case "$MODE" in
  development|development-v2|development-v3|development-v4|audit) ;;
  *) echo "usage: $0 [development|development-v2|development-v3|development-v4|audit]" >&2; exit 2 ;;
esac

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping pi0.5 server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  mkdir -p "$REPO/outputs/logs"
  PORT="$PORT" bash "$REPO/scripts/serve_pi05.sh" \
    >"$REPO/outputs/logs/t01_action_effect_server.log" 2>&1 &
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
    raise SystemExit('pi0.5 server did not become ready')"
}

stop_server() {
  cleanup
  SERVER_PID=""
}

collect() {
  local phase="$1"
  local dataset="$REPO/data/calibration/t01_action_effect_v1.jsonl"
  local manifest="$REPO/data/calibration/t01_action_effect_v1.manifest.json"
  local audit_arg=()
  if [ "$phase" = "audit" ]; then
    dataset="$REPO/data/calibration/t01_action_effect_v1_audit.jsonl"
    manifest="$REPO/data/calibration/t01_action_effect_v1_audit.manifest.json"
    audit_arg=(--audit)
  fi
  if [ -f "$manifest" ]; then
    "$IPU_PYTHON" "$REPO/scripts/verify_dataset_manifest.py" "$manifest"
    return
  fi
  start_server
  local resume_arg=()
  [ -f "$dataset" ] && resume_arg=(--resume)
  env -u PYTHONPATH NUMBA_DISABLE_JIT=1 "$IPU_PYTHON" \
    "$REPO/scripts/collect_t01_action_effect_transitions.py" \
    "${audit_arg[@]}" "${resume_arg[@]}" --fresh-policy-server --port "$PORT"
  stop_server
  "$IPU_PYTHON" "$REPO/scripts/verify_dataset_manifest.py" "$manifest"
}

extract() {
  local phase="$1"
  local dataset="$REPO/data/calibration/t01_action_effect_v1.jsonl"
  local output="$REPO/outputs/t01_action_effect_v1/pi05_transition_embeddings.npz"
  if [ "$phase" = "audit" ]; then
    dataset="$REPO/data/calibration/t01_action_effect_v1_audit.jsonl"
    output="$REPO/outputs/t01_action_effect_v1_audit/pi05_transition_embeddings.npz"
  fi
  if [ -f "$output" ] && [ -f "${output%.npz}.json" ]; then
    echo "Using immutable embeddings: $output"
    return
  fi
  (
    cd "$OPENPI_DIR"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$dataset" --output "$output"
  )
}

extract_spatial_v2() {
  local output="$REPO/outputs/t01_action_effect_v2/pi05_transition_spatial_embeddings.npz"
  if [ -f "$output" ] && [ -f "${output%.npz}.json" ]; then
    echo "Using immutable spatial embeddings: $output"
    return
  fi
  (
    cd "$OPENPI_DIR"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$REPO/data/calibration/t01_action_effect_v1.jsonl" \
      --output "$output" --spatial-v2
  )
}

extract_target_query_v3() {
  local output="$REPO/outputs/t01_action_effect_v3/pi05_transition_target_query_spatial_embeddings.npz"
  if [ -f "$output" ] && [ -f "${output%.npz}.json" ]; then
    echo "Using immutable target-query spatial embeddings: $output"
    return
  fi
  (
    cd "$OPENPI_DIR"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$REPO/data/calibration/t01_action_effect_v1.jsonl" \
      --output "$output" --spatial-v2 --query-prompt "Find the butter"
  )
}

extract_cognitive_query_v4() {
  local output="$REPO/outputs/t01_action_effect_v4/pi05_transition_cognitive_query_embeddings.npz"
  if [ -f "$output" ] && [ -f "${output%.npz}.json" ]; then
    echo "Using immutable cognitive-query embeddings: $output"
    return
  fi
  (
    cd "$OPENPI_DIR"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$REPO/data/calibration/t01_action_effect_v1.jsonl" \
      --output "$output" --spatial-v2 --query-prompt "Find the butter" \
      --cognitive-query-v4
  )
}

cd "$REPO"
if [ "$MODE" = "development" ]; then
  collect development
  extract development
  ARTIFACT="$REPO/results/calibration/t01_action_outcome_critic_v1.json"
  if [ -f "$ARTIFACT" ]; then
    echo "Using frozen critic artifact: $ARTIFACT"
  else
    "$IPU_PYTHON" scripts/fit_t01_action_outcome_critic.py
  fi
elif [ "$MODE" = "development-v2" ]; then
  collect development
  extract_spatial_v2
  ARTIFACT="$REPO/results/calibration/t01_action_outcome_critic_v2.json"
  if [ -f "$ARTIFACT" ]; then
    echo "Using frozen spatial critic artifact: $ARTIFACT"
  else
    "$IPU_PYTHON" scripts/fit_t01_action_outcome_critic.py --spatial-v2
  fi
elif [ "$MODE" = "development-v3" ]; then
  collect development
  extract_target_query_v3
  ARTIFACT="$REPO/results/calibration/t01_action_outcome_critic_v3.json"
  if [ -f "$ARTIFACT" ]; then
    echo "Using frozen target-query critic artifact: $ARTIFACT"
  else
    "$IPU_PYTHON" scripts/fit_t01_action_outcome_critic.py --standardized-v3
  fi
elif [ "$MODE" = "development-v4" ]; then
  collect development
  extract_cognitive_query_v4
  ARTIFACT="$REPO/results/calibration/t01_action_outcome_critic_v4.json"
  if [ -f "$ARTIFACT" ]; then
    echo "Using frozen cognitive-query critic artifact: $ARTIFACT"
  else
    "$IPU_PYTHON" scripts/fit_t01_action_outcome_critic.py --cognitive-v4
  fi
else
  [ -f results/calibration/t01_action_outcome_critic_v1.json ] || {
    echo "Run the development stage and freeze its artifact first." >&2
    exit 1
  }
  collect audit
  extract audit
  RESULT="$REPO/results/calibration/t01_action_outcome_critic_audit_v1.json"
  if [ -f "$RESULT" ]; then
    echo "Using immutable audit result: $RESULT"
  else
    "$IPU_PYTHON" scripts/audit_t01_action_outcome_critic.py
  fi
  EFFECT_V2="$REPO/results/calibration/t01_action_effect_v2.json"
  if [ -f "$EFFECT_V2" ]; then
    echo "Using immutable five-pixel effect artifact: $EFFECT_V2"
  else
    "$IPU_PYTHON" scripts/freeze_t01_action_effect_v2.py
  fi
fi
