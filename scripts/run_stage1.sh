#!/usr/bin/env bash
# Launch only after the actual model and finalized dataset checks passed.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p runs
exec 9> runs/stage1_gpu7.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export HF_HOME="$repo_root/.cache/huggingface"
export HF_MODULES_CACHE="$repo_root/.cache/huggingface/modules"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$repo_root/.cache/tmp"

.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path
from grounded_interaction.predictive_vla.config import load_config

config = load_config('experiments/stage1_policy.yaml')
sys.path.insert(0, 'scripts')
from gpu_devices import verify_cuda_target
device = verify_cuda_target(7)
preflight = json.loads(Path('data/preparation/preflight.json').read_text())
audit = json.loads(Path('data/prepared/stage1_full/final_audit.json').read_text())
preparation = json.loads(Path('data/prepared/stage1_full/preparation_report.json').read_text())
rollouts = json.loads(Path('data/preparation/native_rollouts/report.json').read_text())
if not preflight['passed'] or preflight['manual_seed'] != config.manual_seed:
    raise RuntimeError('real-model preflight for this manual_seed is missing')
if not audit['passed'] or audit['manual_seed'] != config.manual_seed:
    raise RuntimeError('final data audit for this manual_seed is missing')
if preparation['planned_episodes'] != 2000 or len(preparation['episodes']) != 2000 or audit['summary']['episodes'] < 1800:
    raise RuntimeError('full stage-one data preparation is incomplete or rejection rate exceeds 10%')
if not rollouts.get('integration_check_complete'):
    raise RuntimeError('native control integration check is incomplete')
print(json.dumps({'ready': True, 'manual_seed': config.manual_seed,
                  **device, 'data': audit['summary']}), flush=True)

PY

exec .venv/bin/python -m grounded_interaction.predictive_vla train \
  --config experiments/stage1_policy.yaml \
  --data data/prepared/stage1_full/manifest.json \
  --data-audit data/prepared/stage1_full/final_audit.json \
  --model-path checkpoints/base/MolmoAct2-LIBERO \
  --output runs/stage1_s17_gpu7 \
  --device cuda:0 --max-updates 2000 --warmup-updates 200 \
  --eval-every 100 --validation-examples 64 \
  --experiment-log docs/experiment_log.md "$@"
