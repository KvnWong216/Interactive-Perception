#!/usr/bin/env bash
# Launch the pi05_libero policy server on a GPU host.
#
# The scenario process and the model process are separate: this script runs on
# the machine with the GPU, while run_challenge_rollout.py and run_repro_gate.py
# can run anywhere that can reach it over the network. Point them at this host
# with --host/--port.
#
# Usage:
#   scripts/serve_pi05.sh [OPENPI_DIR] [CHECKPOINT_DIR]
#
# Defaults assume the sibling layout produced by scripts/download_pi05_libero.py:
#   <workspace>/openpi
#   <workspace>/checkpoints/checkpoints/pi05_libero
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OPENPI_DIR="${1:-${OPENPI_DIR:-$WORKSPACE/openpi}}"
CHECKPOINT_DIR="${2:-${CHECKPOINT_DIR:-$WORKSPACE/checkpoints/checkpoints/pi05_libero}}"
PORT="${PORT:-8000}"

if [ ! -d "$OPENPI_DIR" ]; then
  echo "openpi checkout not found at: $OPENPI_DIR" >&2
  echo "  git clone https://github.com/Physical-Intelligence/openpi.git $OPENPI_DIR" >&2
  exit 1
fi

if [ ! -d "$CHECKPOINT_DIR/params" ]; then
  echo "checkpoint params not found at: $CHECKPOINT_DIR/params" >&2
  echo "  python scripts/download_pi05_libero.py --dest $(dirname "$(dirname "$CHECKPOINT_DIR")")" >&2
  exit 1
fi

# JAX preallocates most of the device by default; leaving headroom lets the
# LIBERO renderer share the GPU when both run on one machine.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
EXPERIMENT_GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$EXPERIMENT_GPU_INDEX}"
if [ "$CUDA_VISIBLE_DEVICES" != "$EXPERIMENT_GPU_INDEX" ]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must match physical GPU${EXPERIMENT_GPU_INDEX}." >&2
  exit 1
fi
if [ "${LAB_SERVER_MODE:-0}" = "1" ] && [ "$EXPERIMENT_GPU_INDEX" != "1" ]; then
  echo "Refusing to start: lab-server mode is authorized only on physical GPU1." >&2
  exit 1
fi

echo "openpi:     $OPENPI_DIR"
echo "checkpoint: $CHECKPOINT_DIR"
echo "port:       $PORT"
echo "device:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$OPENPI_DIR"
exec uv run scripts/serve_policy.py \
  --port "$PORT" \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$CHECKPOINT_DIR"
