#!/usr/bin/env bash
# One-command, dependency-gated RSS experiment ladder.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
MODE="${1:-preflight}"

case "$MODE" in
  preflight|smoke|development|audit) ;;
  *) echo "usage: $0 [preflight|smoke|development|audit]" >&2; exit 2 ;;
esac

cd "$REPO"
env -u PYTHONPATH "$IPU_PYTHON" -m pytest -q
env -u PYTHONPATH "$IPU_PYTHON" -m py_compile \
  src/interactive_perception/pipeline.py \
  scripts/collect_t01_open_and_observe_effect.py \
  scripts/extract_pi05_transition_embeddings.py \
  scripts/serve_pi05_with_prefix.py \
  scripts/freeze_t01_open_and_observe_effect.py \
  scripts/freeze_t01_observed_effect_v2.py \
  scripts/freeze_t01_temporal_effect_v3.py \
  scripts/fit_t01_temporal_outcome_critic.py \
  scripts/fit_t01_hierarchical_outcome_critic.py \
  scripts/audit_t01_temporal_outcome_critic.py \
  scripts/audit_t01_hierarchical_outcome_critic.py \
  scripts/check_pi05_prefix_parity.py \
  scripts/run_t01_closed_loop_v5.py

if [ "$MODE" = "preflight" ]; then
  echo "Preflight passed. No model server was started."
  exit 0
fi

if [ "$MODE" = "smoke" ]; then
  bash scripts/run_t01_open_and_observe_pipeline.sh smoke
  echo "Smoke collection passed. No calibration or audit claim was made."
  exit 0
fi

if [ "$MODE" = "development" ]; then
  bash scripts/run_t01_open_and_observe_pipeline.sh development
  EFFECT="results/calibration/t01_open_and_observe_effect_v3.json"
  if [ ! -f "$EFFECT" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/freeze_t01_temporal_effect_v3.py
  fi
  EMBEDDINGS="outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz"
  if [ ! -f "$EMBEDDINGS" ]; then
    (
      cd "$OPENPI_DIR"
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
        "$REPO/scripts/extract_pi05_transition_embeddings.py" \
        --dataset "$REPO/data/calibration/t01_open_and_observe_effect_v3.jsonl" \
        --output "$REPO/$EMBEDDINGS" \
        --spatial-v2 --query-prompt "Find the butter" \
        --cognitive-query-v4 --temporal-v5
    )
  fi
  CRITIC="results/calibration/t01_open_and_observe_outcome_critic_v8.json"
  if [ ! -f "$CRITIC" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/fit_t01_hierarchical_outcome_critic.py
  fi
  env -u PYTHONPATH "$IPU_PYTHON" -c \
    "import json; d=json.load(open('$CRITIC')); assert d['development']['passed']; print('Development GO; sealed audit remains untouched.')"
  exit 0
fi

CRITIC="results/calibration/t01_open_and_observe_outcome_critic_v8.json"
[ -f "$CRITIC" ] || { echo "missing frozen v8 critic; run development first" >&2; exit 1; }
env -u PYTHONPATH "$IPU_PYTHON" -c \
  "import json; d=json.load(open('$CRITIC')); assert d['development']['passed'], 'development NOT-GO'"
[ "${ALLOW_SEALED_AUDIT:-0}" = "1" ] || {
  echo "Refusing one-time seeds 900-999. Re-run with ALLOW_SEALED_AUDIT=1 after reviewing development." >&2
  exit 1
}
AUDIT_DATA="data/calibration/t01_open_and_observe_effect_v3_audit.jsonl"
AUDIT_MANIFEST="data/calibration/t01_open_and_observe_effect_v3_audit.manifest.json"
AUDIT_RESULT="results/calibration/t01_open_and_observe_outcome_audit_v8.json"
if [ -f "$AUDIT_RESULT" ]; then
  env -u PYTHONPATH "$IPU_PYTHON" -c \
    "import json; d=json.load(open('$AUDIT_RESULT')); print('Immutable audit already finished: FP3', 'GO' if d['fp3_passed'] else 'NOT-GO'); raise SystemExit(0 if d['fp3_passed'] else 1)"
  exit $?
fi
if [ -f "$AUDIT_MANIFEST" ]; then
  [ -f "$AUDIT_DATA" ] || {
    echo "audit manifest exists without its dataset; refusing recovery" >&2
    exit 1
  }
  env -u PYTHONPATH "$IPU_PYTHON" -c \
    "import hashlib,json; p='$AUDIT_DATA'; m=json.load(open('$AUDIT_MANIFEST')); a='$CRITIC'; assert m['phase']=='heldout_audit' and m['seeds']==list(range(900,1000)); assert m['dataset_sha256']==hashlib.sha256(open(p,'rb').read()).hexdigest(); assert m['audit_artifact_sha256']==hashlib.sha256(open(a,'rb').read()).hexdigest(); print('Verified immutable audit collection; resuming post-processing only.')"
else
  OPEN_AND_OBSERVE_ARTIFACT="$REPO/$CRITIC" \
    bash scripts/run_t01_open_and_observe_pipeline.sh audit
fi

AUDIT_EMBEDDINGS="outputs/t01_open_and_observe_effect_v3_audit/pi05_temporal_embeddings_v5.npz"
if [ ! -f "$AUDIT_EMBEDDINGS" ]; then
  (
    cd "$OPENPI_DIR"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$REPO/data/calibration/t01_open_and_observe_effect_v3_audit.jsonl" \
      --output "$REPO/$AUDIT_EMBEDDINGS" \
      --spatial-v2 --query-prompt "Find the butter" \
      --cognitive-query-v4 --temporal-v5
  )
fi
env -u PYTHONPATH "$IPU_PYTHON" scripts/audit_t01_hierarchical_outcome_critic.py
