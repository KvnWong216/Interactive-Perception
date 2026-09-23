#!/usr/bin/env bash
# Complete all 2,000 candidates before starting the bounded stage-one run.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p runs
exec 8> runs/stage1_pipeline.lock
flock -n 8
export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_EGL_DEVICE_ID=7
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$repo_root/.cache/tmp"

.venv/bin/python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path.cwd().resolve()
source = root / 'data/prepared/stage1'
target = root / 'data/prepared/stage1_full'
target.mkdir(parents=True, exist_ok=True)
reused = 0
for entry in json.loads((source / 'manifest.json').read_text())['episodes']:
    old = (source / entry['path']).resolve()
    new = (target / entry['path']).resolve()
    if not old.is_relative_to(source) or not new.is_relative_to(target):
        raise ValueError('prepared episode path escaped repository data directories')
    new.parent.mkdir(parents=True, exist_ok=True)
    if not new.exists():
        os.link(old, new)
        shutil.copy2(old.with_suffix('.json'), new.with_suffix('.json'))
    reused += 1
print(json.dumps({'phase': 'preparing_full_dataset', 'manual_seed': 17,
                  'candidate_episodes': 2000, 'reused_episodes': reused}), flush=True)
PY

if [ -f data/prepared/stage1_full/preparation_report.json ]; then
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('data/prepared/stage1_full/preparation_report.json').read_text())
if report['audit_only'] or report['planned_episodes'] != 2000 or len(report['episodes']) != 2000:
    raise RuntimeError('existing full preparation report is incomplete')
print(json.dumps({'phase': 'reusing_complete_preparation', 'manual_seed': 17,
                  'accepted': report['accepted_episodes']}), flush=True)
PY
else
  .venv-sim/bin/python scripts/prepare_libero.py \
    --output data/prepared/stage1_full --episodes-per-task 50 \
    --image-size 256 --workers 32 --manual-seed 17 --max-output-gb 400
fi

.venv/bin/python scripts/finalize_stage1.py \
  --data data/prepared/stage1_full --manual-seed 17 --minimum-episodes 1800

.venv/bin/python - <<'PY'
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

audit = json.loads(Path('data/prepared/stage1_full/final_audit.json').read_text())
summary = audit['summary']
timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
with Path('docs/experiment_log.md').open('a') as stream:
    stream.write(f"\n{timestamp}：完整阶段一数据检查通过，实际保留 {summary['episodes']} 条轨迹、"
                 f"{summary['actions']} 个动作，划分 {summary['splits']}；manual_seed=17。"
                 "现在启动 GPU 7 正式训练，更新预算 2,000 次，有效 batch=32。\n")
print(json.dumps({'phase': 'starting_full_stage1', 'manual_seed': 17, **summary}), flush=True)
PY

exec bash scripts/run_stage1.sh
