#!/usr/bin/env bash
# Refuse to start a project model server unless the selected physical GPU is
# present and has no unapproved compute process.
set -euo pipefail

GPU_INDEX="${EXPERIMENT_GPU_INDEX:-0}"
LAB_SERVER_MODE="${LAB_SERVER_MODE:-0}"
ALLOW_LOCAL_RUSTDESK="${EXPERIMENT_ALLOW_LOCAL_RUSTDESK:-0}"

if [ "$LAB_SERVER_MODE" = "1" ] && [ "$GPU_INDEX" != "1" ]; then
  echo "GPU preflight failed: lab-server mode is authorized only on physical GPU1." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU preflight failed: nvidia-smi is unavailable." >&2
  exit 1
fi

if ! nvidia-smi --id="$GPU_INDEX" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader; then
  echo "GPU preflight failed: physical GPU${GPU_INDEX} is unavailable." >&2
  exit 1
fi

compute_processes="$(nvidia-smi --id="$GPU_INDEX" \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)"
unapproved=""
approved_count=0
while IFS=',' read -r raw_pid raw_name raw_memory; do
  [ -n "$raw_pid" ] || continue
  pid="${raw_pid//[[:space:]]/}"
  name="${raw_name#${raw_name%%[![:space:]]*}}"
  name="${name%${name##*[![:space:]]}}"
  memory="${raw_memory//[[:space:]]/}"
  owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
  if [ "$ALLOW_LOCAL_RUSTDESK" = "1" ] \
    && [ "$name" = "/usr/share/rustdesk/rustdesk" ] \
    && [ "$owner" = "$(id -un)" ]; then
    approved_count=$((approved_count + 1))
    echo "GPU preflight note: allowing current-user RustDesk infrastructure (${memory} MiB)."
    continue
  fi
  unapproved="${unapproved}${pid}, ${name}, ${memory} MiB\n"
done <<< "$compute_processes"

if [ -n "$unapproved" ]; then
  echo "GPU preflight failed: GPU${GPU_INDEX} has an unapproved compute process; refusing to interfere." >&2
  printf '%b' "$unapproved" >&2
  exit 1
fi

echo "GPU preflight passed: physical GPU${GPU_INDEX} is available; approved infrastructure processes=${approved_count}."
