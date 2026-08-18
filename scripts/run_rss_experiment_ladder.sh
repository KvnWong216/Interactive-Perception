#!/usr/bin/env bash
# One-command, dependency-gated RSS experiment ladder.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
IPU_PYTHON="${IPU_PYTHON:-$WORKSPACE/.conda/envs/ipu/bin/python}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
EXPERIMENT_GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
EXPERIMENT_ALLOW_LOCAL_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-1}"
MODE="${1:-preflight}"

case "$MODE" in
  preflight|smoke|development|audit) ;;
  *) echo "usage: $0 [preflight|smoke|development|audit]" >&2; exit 2 ;;
esac

cd "$REPO"
env -u PYTHONPATH "$IPU_PYTHON" -m pytest -q
env -u PYTHONPATH "$IPU_PYTHON" -m py_compile \
  src/interactive_perception/pipeline.py \
  src/interactive_perception/minimal_pipeline.py \
  src/interactive_perception/seed_registry.py \
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
  scripts/audit_t01_privileged_inputs.py \
  scripts/evaluate_t01_v9_clean_extension.py \
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
  SMOKE_MANIFEST="data/calibration/t01_open_and_observe_effect_v4_smoke.manifest.json"
  [ -f "$SMOKE_MANIFEST" ] || {
    echo "refusing clean development: v4 disposable smoke has not passed" >&2
    exit 1
  }
  env -u PYTHONPATH "$IPU_PYTHON" scripts/verify_dataset_manifest.py "$SMOKE_MANIFEST"
  env -u PYTHONPATH "$IPU_PYTHON" -c \
    "import json; m=json.load(open('$SMOKE_MANIFEST')); assert m['schema_version']=='interactive-perception.open-and-observe-manifest.v4' and m['phase']=='smoke' and m['seeds']==[1399] and m['samples']==3; print('Verified disposable v4 smoke prerequisite.')"
  PRIMARY_CANDIDATE="results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_visual.json"
  if [ ! -f "$PRIMARY_CANDIDATE" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/fit_t01_hierarchical_outcome_critic.py \
      --block visual_only_history --output "$PRIMARY_CANDIDATE"
  fi
  FULL_CANDIDATE="results/calibration/t01_open_and_observe_outcome_critic_v9_candidate.json"
  if [ ! -f "$FULL_CANDIDATE" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/fit_t01_hierarchical_outcome_critic.py \
      --block temporal_history --output "$FULL_CANDIDATE"
  fi
  GLOBAL_CANDIDATE="results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_global.json"
  if [ ! -f "$GLOBAL_CANDIDATE" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/fit_t01_hierarchical_outcome_critic.py \
      --block global_visual_history --output "$GLOBAL_CANDIDATE"
  fi
  NO_HISTORY_CANDIDATE="results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_no_history.json"
  if [ ! -f "$NO_HISTORY_CANDIDATE" ]; then
    env -u PYTHONPATH "$IPU_PYTHON" scripts/fit_t01_hierarchical_outcome_critic.py \
      --block no_history --output "$NO_HISTORY_CANDIDATE"
  fi
  bash scripts/run_t01_open_and_observe_pipeline.sh extension
  EMBEDDINGS="outputs/t01_open_and_observe_effect_v4_extension/pi05_temporal_embeddings_v5.npz"
  if [ ! -f "$EMBEDDINGS" ]; then
    EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" \
      EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$EXPERIMENT_ALLOW_LOCAL_RUSTDESK" \
      bash scripts/check_gpu_preflight.sh
    (
      cd "$OPENPI_DIR"
      CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_INDEX" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
        "$REPO/scripts/extract_pi05_transition_embeddings.py" \
        --dataset "$REPO/data/calibration/t01_open_and_observe_effect_v4_extension.jsonl" \
        --output "$REPO/$EMBEDDINGS" \
        --spatial-v2 --query-prompt "Find the butter" \
        --cognitive-query-v4 --temporal-v5
    )
  fi
  env -u PYTHONPATH "$IPU_PYTHON" scripts/evaluate_t01_v9_clean_extension.py
  echo "Clean v9 development GO; sealed audit remains untouched."
  exit 0
fi

CRITIC="results/calibration/t01_open_and_observe_outcome_critic_v9.json"
[ -f "$CRITIC" ] || { echo "missing frozen v9 critic; run development first" >&2; exit 1; }
env -u PYTHONPATH "$IPU_PYTHON" -c \
  "import json; d=json.load(open('$CRITIC')); assert d['development']['passed'], 'development NOT-GO'"
[ "${ALLOW_SEALED_AUDIT:-0}" = "1" ] || {
  echo "Refusing one-time seeds 900-999. Re-run with ALLOW_SEALED_AUDIT=1 after reviewing development." >&2
  exit 1
}
AUDIT_DATA="data/calibration/t01_open_and_observe_effect_v4_audit.jsonl"
AUDIT_MANIFEST="data/calibration/t01_open_and_observe_effect_v4_audit.manifest.json"
AUDIT_RESULT="results/calibration/t01_open_and_observe_outcome_audit_v9.json"
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

AUDIT_EMBEDDINGS="outputs/t01_open_and_observe_effect_v4_audit/pi05_temporal_embeddings_v5.npz"
if [ ! -f "$AUDIT_EMBEDDINGS" ]; then
  EXPERIMENT_GPU_INDEX="$EXPERIMENT_GPU_INDEX" \
    EXPERIMENT_ALLOW_LOCAL_RUSTDESK="$EXPERIMENT_ALLOW_LOCAL_RUSTDESK" \
    bash scripts/check_gpu_preflight.sh
  (
    cd "$OPENPI_DIR"
    CUDA_VISIBLE_DEVICES="$EXPERIMENT_GPU_INDEX" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run \
      "$REPO/scripts/extract_pi05_transition_embeddings.py" \
      --dataset "$REPO/data/calibration/t01_open_and_observe_effect_v4_audit.jsonl" \
      --output "$REPO/$AUDIT_EMBEDDINGS" \
      --spatial-v2 --query-prompt "Find the butter" \
      --cognitive-query-v4 --temporal-v5
  )
fi
env -u PYTHONPATH "$IPU_PYTHON" scripts/audit_t01_hierarchical_outcome_critic.py
