#!/usr/bin/env bash
# All writes stay in this repository. Preflight weights never enter the main run.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p runs .cache/tmp
exec 9> runs/stage2_gpu4567.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HOME="$repo_root/.cache/huggingface"
export HF_MODULES_CACHE="$repo_root/.cache/huggingface/modules"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$repo_root/.cache/tmp"

if [[ "${1:-}" == "--preflight" ]]; then
  shift
  .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_stage2.py \
    --output runs/stage2_transition_preflight_s17_gpu4567 --max-updates 2 \
    --validation-episodes 4 --preflight "$@"
  exec .venv/bin/python scripts/audit_stage2_checkpoint.py --run runs/stage2_transition_preflight_s17_gpu4567
fi

.venv/bin/python - <<'PY'
import json
from pathlib import Path
from grounded_interaction.predictive_vla.config import load_config
config = load_config('experiments/stage2_transition.yaml')
report = json.loads(Path('runs/stage2_transition_preflight_s17_gpu4567/preflight.json').read_text())
checkpoint = json.loads(Path('runs/stage2_transition_preflight_s17_gpu4567/checkpoint_audit.json').read_text())
if not report['passed'] or report['world_size'] != 4 or report['config'] != config.to_dict():
    raise RuntimeError('matching four-GPU preflight has not passed')
if report['manual_seed'] != config.manual_seed or not report['initialization']['exact_restoration']:
    raise RuntimeError('manual seed or exact stage-one restoration check failed')
if not checkpoint['passed'] or checkpoint['rank_random_states'] != 4:
    raise RuntimeError('four-rank checkpoint serialization audit has not passed')
print('Four-GPU preflight passed; restarting from stage-one best with a fresh optimizer.', flush=True)
PY

exec .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_stage2.py \
  --output runs/stage2_transition_s17_gpu4567 --max-updates 2000 \
  --warmup-updates 200 --prediction-warmup-updates 200 \
  --eval-every 100 --validation-episodes 198 "$@"
