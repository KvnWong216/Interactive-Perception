#!/usr/bin/env bash
# Run the whole pi0.5 baseline experiment in one go.
#
# Stages, in order:
#   0  preflight        environment, LIBERO checkout, checkpoint
#   1  anchors          every hypothesis_anchors ref resolves in every scene
#   2  scenes           the v0.2 scene validators still pass
#   3  server           start the pi05_libero policy server and wait for it
#   4  gate             reproduction gate on stock LIBERO  <-- hard gate
#   5  sweep            the prompt ladder on the challenge scenes
#   6  figures          uncertainty curves, primitive evidence, demo videos
#   7  nbv              viewpoint-sufficiency certificates
#
# Stage 4 is a hard gate on purpose. It catches a silent wiring fault -- a
# missing image rotation, a mis-ordered state vector, a stale checkpoint --
# which would depress success on the challenge scenes and be indistinguishable
# from the information-seeking failure the benchmark exists to measure. If it
# fails, nothing downstream is evidence of anything, so the script stops.
#
# Usage:
#   scripts/run_full_experiment.sh              # full sweep, ~120 episodes
#   QUICK=1 scripts/run_full_experiment.sh      # one task, one seed, smoke test
#   SKIP_NBV=1 scripts/run_full_experiment.sh   # skip the camera sweep
#
# Every knob below is overridable from the environment.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"

CONDA_ENV="${CONDA_ENV:-$WORKSPACE/.conda/envs/ipu}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORKSPACE/checkpoints/checkpoints/pi05_libero}"
PORT="${PORT:-8000}"

SEEDS="${SEEDS:-0 1 2 3 4}"
VARIANTS="${VARIANTS:-implicit hinted explicit capability}"
TASKS="${TASKS:-}"                      # empty means every task in the spec
MAX_STEPS="${MAX_STEPS:-300}"
PROBE_EVERY="${PROBE_EVERY:-25}"
PROBE_SAMPLES="${PROBE_SAMPLES:-12}"
GATE_TRIALS="${GATE_TRIALS:-10}"
NBV_AZIMUTHS="${NBV_AZIMUTHS:-24}"
NBV_ELEVATIONS="${NBV_ELEVATIONS:-5}"
NBV_RADII="${NBV_RADII:-0.6 0.9 1.2}"

SKIP_SERVER="${SKIP_SERVER:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
SKIP_NBV="${SKIP_NBV:-0}"
SKIP_SCENES="${SKIP_SCENES:-0}"

if [ "${QUICK:-0}" = "1" ]; then
  TASKS="T01_drawer_retrieval T04_visible_direct"
  SEEDS="0"
  VARIANTS="implicit capability"
  MAX_STEPS=150
  PROBE_SAMPLES=6
  GATE_TRIALS=2
  NBV_AZIMUTHS=8
  NBV_ELEVATIONS=3
  NBV_RADII="0.6 1.0"
fi

OUT="$REPO/outputs"
LOG_DIR="$OUT/logs"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
SERVER_LOG="$LOG_DIR/policy_server.log"
mkdir -p "$LOG_DIR"

# ROS installs put /opt/ros/... on PYTHONPATH, which shadows packages in the
# conda environment and breaks imports. Clearing it is required, not cosmetic.
run() { env -u PYTHONPATH "$@"; }

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*" | tee -a "$RUN_LOG"; }
note() { printf '   %s\n' "$*" | tee -a "$RUN_LOG"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" | tee -a "$RUN_LOG"; exit 1; }

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    note "stopping policy server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- 0 preflight
say "0/7 preflight"
[ -d "$CONDA_ENV" ] || die "conda env not found: $CONDA_ENV"
[ -d "$REPO/third_party/LIBERO" ] || die "LIBERO checkout missing: $REPO/third_party/LIBERO"
[ -d "$CHECKPOINT_DIR/params" ] || die "checkpoint missing: $CHECKPOINT_DIR/params
  fetch it with: python scripts/download_pi05_libero.py --dest $WORKSPACE/checkpoints"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"

[ -f "$REPO/.libero/config.yaml" ] || run python scripts/setup_libero_config.py
note "repo:       $REPO"
note "checkpoint: $CHECKPOINT_DIR"
note "tasks:      ${TASKS:-<all>}"
note "variants:   $VARIANTS"
note "seeds:      $SEEDS"
note "log:        $RUN_LOG"

TASK_ARG=()
[ -n "$TASKS" ] && TASK_ARG=(--task-ids $TASKS)

# ------------------------------------------------------------------ 1 anchors
say "1/7 anchor references"
run python scripts/list_scene_handles.py "${TASK_ARG[@]}" 2>&1 \
  | grep -E "===|UNRESOLVED|    -|all anchor" | tee -a "$RUN_LOG" \
  || die "unresolved anchor references; fix hypothesis_anchors in the spec"

# ------------------------------------------------------------------- 2 scenes
if [ "$SKIP_SCENES" = "1" ]; then
  say "2/7 scene validation (skipped)"
else
  say "2/7 scene validation"
  run python scripts/validate_interactive_manipulation_v0.py \
    --seeds 0 --strict 2>&1 | grep -E "case_runs|case_passes|all_checks" \
    | tee -a "$RUN_LOG" || die "scene validation failed"
fi

# ------------------------------------------------------------------- 3 server
if [ "$SKIP_SERVER" = "1" ]; then
  say "3/7 policy server (assumed already running on port $PORT)"
else
  say "3/7 policy server"
  : > "$SERVER_LOG"
  PORT="$PORT" bash scripts/serve_pi05.sh "$OPENPI_DIR" "$CHECKPOINT_DIR" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  note "pid $SERVER_PID, log $SERVER_LOG"
  # Loading ~6 GiB of weights takes a while; poll rather than sleeping blindly.
  for _ in $(seq 1 120); do
    grep -q "server listening" "$SERVER_LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || die "server died during startup; see $SERVER_LOG"
    sleep 5
  done
  grep -q "server listening" "$SERVER_LOG" || die "server did not come up; see $SERVER_LOG"
  note "server ready on port $PORT"
fi

# --------------------------------------------------------------------- 4 gate
if [ "$SKIP_GATE" = "1" ]; then
  say "4/7 reproduction gate (skipped -- downstream results are unverified)"
else
  say "4/7 reproduction gate"
  run python scripts/run_repro_gate.py \
    --port "$PORT" --trials-per-task "$GATE_TRIALS" --strict 2>&1 \
    | grep -E "repro\]|success rate|success_rate|gate_passed" | tee -a "$RUN_LOG" \
    || die "reproduction gate failed -- the client/checkpoint wiring is wrong.
  No challenge-scenario result is evidence of anything until this passes."
fi

# -------------------------------------------------------------------- 5 sweep
say "5/7 prompt-ladder sweep"
note "this is the long stage; a probe step costs PROBE_SAMPLES inferences"
run python scripts/run_challenge_rollout.py \
  --port "$PORT" \
  "${TASK_ARG[@]}" \
  --variants $VARIANTS \
  --seeds $SEEDS \
  --max-steps "$MAX_STEPS" \
  --probe-every "$PROBE_EVERY" \
  --probe-samples "$PROBE_SAMPLES" \
  --save-frames 2>&1 | tee -a "$RUN_LOG" | grep -E "rollout\]|success=|skip" || true

[ -f "$OUT/challenge_rollout/rollout_summary.json" ] || die "sweep produced no summary"

# ------------------------------------------------------------------ 6 figures
say "6/7 figures and demo videos"
for metric in vacuity dissonance; do
  run python scripts/plot_uncertainty_curves.py --figures --metric "$metric" \
    2>&1 | tee -a "$RUN_LOG"
done
run python scripts/plot_uncertainty_curves.py --all-videos --metric vacuity \
  2>&1 | tee -a "$RUN_LOG"

# The server is only needed up to here; free the GPU before the camera sweep.
cleanup
SERVER_PID=""

# ---------------------------------------------------------------------- 7 nbv
if [ "$SKIP_NBV" = "1" ]; then
  say "7/7 viewpoint certification (skipped)"
else
  say "7/7 viewpoint certification"
  note "renders $((NBV_AZIMUTHS * NBV_ELEVATIONS * $(echo $NBV_RADII | wc -w))) poses per sweep, twice per scene"
  run python scripts/certify_nbv_insufficiency.py \
    "${TASK_ARG[@]}" \
    --azimuths "$NBV_AZIMUTHS" \
    --elevations "$NBV_ELEVATIONS" \
    --radii $NBV_RADII 2>&1 \
    | grep -E "certify\]|  NBV|  UNREV|  ERROR" | tee -a "$RUN_LOG" || true
fi

# ------------------------------------------------------------------- epilogue
say "done"
note "summary:      $OUT/challenge_rollout/rollout_summary.json"
note "figures:      $OUT/figures/"
note "certificates: $OUT/nbv_certificates/nbv_certificates.json"
note "full log:     $RUN_LOG"
cat <<'EOF'

   Read capability_minus_implicit in rollout_summary.json first. It compares
   information_endpoint_reached between the two conditions:

     capability high, implicit near zero  ->  the policy can do the information
                                              step but never elects to: a
                                              decision failure, which is the
                                              claim under test
     both near zero                       ->  the policy cannot do the step even
                                              when instructed: a skill gap, and
                                              the scenario cannot support any
                                              claim about information seeking
EOF
