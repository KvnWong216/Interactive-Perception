#!/usr/bin/env bash
# Launch the versioned frozen action + prefix server from the pinned OpenPI tree.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORKSPACE/checkpoints/checkpoints/pi05_libero}"
PORT="${PORT:-8000}"

[ -d "$OPENPI_DIR" ] || { echo "missing OpenPI checkout: $OPENPI_DIR" >&2; exit 1; }
[ -f "$CHECKPOINT_DIR/params/_METADATA" ] || {
  echo "missing pi05_libero checkpoint: $CHECKPOINT_DIR" >&2
  exit 1
}

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$OPENPI_DIR"
exec uv run "$REPO/scripts/serve_pi05_with_prefix.py" \
  --port "$PORT" --checkpoint "$CHECKPOINT_DIR"
