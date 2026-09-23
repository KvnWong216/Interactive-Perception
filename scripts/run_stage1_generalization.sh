#!/usr/bin/env bash
# User-authorized GPU 0; training continues independently on GPUs 4/5/6/7.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p runs/stage1_generalization_s1217_gpu0
exec 9> runs/stage1_generalization_gpu0.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

.venv-sim/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('runs/stage1_generalization_s1217_gpu0/scene_audit/report.json')
audit = json.loads(path.read_text())
assert audit['passed'] and audit['cases'] == 96
history = json.loads(Path('runs/stage1_native_best_history_s217_gpu0/evaluation/report.json').read_text())
assert history['complete'] and history['modes'] == ['native', 'best']
print('Scene audits passed; the earlier GPU 0 history evaluation is complete.', flush=True)
PY
.venv-sim/bin/python scripts/summarize_stage1_generalization.py
.venv/bin/python -u scripts/evaluate_diagnostics.py \
  --manifest experiments/diagnostic_cases/stage1_generalization_s1217/manifest.json \
  --scene-audit runs/stage1_generalization_s1217_gpu0/scene_audit/report.json \
  --output runs/stage1_generalization_s1217_gpu0/evaluation \
  --physical-gpu 0 --modes native best --kinds S "$@"
.venv-sim/bin/python scripts/summarize_stage1_generalization.py
