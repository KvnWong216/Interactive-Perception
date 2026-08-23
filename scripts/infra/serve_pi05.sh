#!/usr/bin/env bash
# Launch the pi05_libero policy server on a GPU host.
#
# The LIBERO controller and policy model remain separate processes so their
# pinned Python stacks do not conflict.
#
# Usage:
#   scripts/infra/serve_pi05.sh [OPENPI_DIR] [CHECKPOINT_DIR]
#
# Defaults assume the sibling layout produced by scripts/infra/download_pi05.py:
#   <workspace>/openpi
#   <workspace>/checkpoints/checkpoints/pi05_libero
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "$PROJECT_ROOT")}"
OPENPI_DIR="${1:-${OPENPI_DIR:-$WORKSPACE/openpi}}"
CHECKPOINT_DIR="${2:-${CHECKPOINT_DIR:-$WORKSPACE/checkpoints/checkpoints/pi05_libero}}"
PORT="${PORT:-8002}"
HOST="${HOST:-127.0.0.1}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-local_identified_server}"

if [ ! -d "$OPENPI_DIR" ]; then
  echo "openpi checkout not found at: $OPENPI_DIR" >&2
  echo "  git clone https://github.com/Physical-Intelligence/openpi.git $OPENPI_DIR" >&2
  exit 1
fi

if [ ! -d "$CHECKPOINT_DIR/params" ]; then
  echo "checkpoint params not found at: $CHECKPOINT_DIR/params" >&2
  echo "  python scripts/infra/download_pi05.py --dest $(dirname "$(dirname "$CHECKPOINT_DIR")")" >&2
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
if [ "$DEPLOYMENT_MODE" != "local_identified_server" ] \
  && [ "$DEPLOYMENT_MODE" != "remote_identified_server" ]; then
  echo "Refusing to start: DEPLOYMENT_MODE must identify local or remote serving." >&2
  exit 1
fi
if [ "$DEPLOYMENT_MODE" = "local_identified_server" ] \
  && [ "$XLA_PYTHON_CLIENT_MEM_FRACTION" != "0.85" ]; then
  echo "Refusing to start: local empirical contract freezes XLA fraction at 0.85." >&2
  exit 1
fi
if [ "${LAB_SERVER_MODE:-0}" = "1" ] && [ "$EXPERIMENT_GPU_INDEX" != "1" ]; then
  echo "Refusing to start: lab-server mode is authorized only on physical GPU1." >&2
  exit 1
fi

echo "openpi:     $OPENPI_DIR"
echo "checkpoint: $CHECKPOINT_DIR"
echo "port:       $PORT"
echo "host:       $HOST"
echo "device:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "deployment: $DEPLOYMENT_MODE"

"$PROJECT_ROOT/scripts/infra/check_gpu.sh"

cd "$OPENPI_DIR"
exec uv run python "$PROJECT_ROOT/scripts/infra/serve_identified_pi05.py" \
  --checkpoint "$CHECKPOINT_DIR" \
  --policy-config pi05_libero \
  --port "$PORT" \
  --host "$HOST" \
  --physical-gpu-index "$EXPERIMENT_GPU_INDEX"
