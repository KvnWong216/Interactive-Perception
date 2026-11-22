#!/usr/bin/env bash
# Plug-and-play ActiveArena bootstrap for IP-VLA Stage-2 Evaluation.
# Clones pinned ActiveArena + ActiveArena-VLA (if not already present/linked),
# downloads RoboTwin2.0 + ActiveArena assets, downloads official OFT checkpoints
# via resumable curl (bypassing HF xet proxy issues), and runs a dry-run check.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT_DIR="${REPO_ROOT}/eval/activearena/external"
mkdir -p "${EXT_DIR}"

AA_SIM="${ACTIVEARENA_ROOT:-${EXT_DIR}/ActiveArena}"
AA_VLA="${ACTIVEARENA_VLA_ROOT:-${EXT_DIR}/ActiveArena-VLA}"
AA_WEIGHTS="${ACTIVEARENA_VLA_WEIGHTS_ROOT:-${EXT_DIR}/ActiveArena-VLA-weights}"

echo "=== [1/4] Checking ActiveArena & ActiveArena-VLA checkouts ==="
if [[ ! -d "${AA_SIM}" ]]; then
  git clone https://github.com/leeibo/ActiveArena.git "${AA_SIM}"
else
  echo "  Found ActiveArena: ${AA_SIM}"
fi

if [[ ! -d "${AA_VLA}" ]]; then
  git clone https://github.com/leeibo/ActiveArena-VLA.git "${AA_VLA}"
else
  echo "  Found ActiveArena-VLA: ${AA_VLA}"
fi

echo "=== [2/4] Downloading official ActiveArena-VLA checkpoints (resumable curl) ==="
REV_VLA="c265fe089dc23656e7ff82eb0fe8012adc0644aa"
BASE_VLA="https://huggingface.co/leeibo/ActiveArena-VLA/resolve/${REV_VLA}"
mkdir -p "${AA_WEIGHTS}/oft_subtask_action_12_ws/checkpoints" "${AA_WEIGHTS}/oft_instruction_action_12_ws/checkpoints"
for rel in \
  "manifest.json" \
  "oft_subtask_action_12_ws/config.yaml" \
  "oft_subtask_action_12_ws/config.full.yaml" \
  "oft_subtask_action_12_ws/dataset_statistics.json" \
  "oft_subtask_action_12_ws/checkpoints/steps_100000_pytorch_model.pt" \
  "oft_instruction_action_12_ws/config.yaml" \
  "oft_instruction_action_12_ws/config.full.yaml" \
  "oft_instruction_action_12_ws/dataset_statistics.json" \
  "oft_instruction_action_12_ws/checkpoints/steps_100000_pytorch_model.pt"; do
  target="${AA_WEIGHTS}/${rel}"
  if [[ -f "${target}" && "${rel}" != *.pt ]]; then
    continue
  fi
  if [[ -f "${target}" && "$(stat -c%s "${target}")" -eq 4967033387 ]]; then
    echo "  Already complete: ${rel}"
    continue
  fi
  echo "  Downloading ${rel}..."
  curl -L -C - --fail --retry 8 --retry-delay 5 --connect-timeout 30 -o "${target}" "${BASE_VLA}/${rel}"
done

echo "=== [3/4] Downloading RoboTwin2.0 + ActiveArena simulation assets ==="
REV_RT="981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"
BASE_RT="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/${REV_RT}"
mkdir -p "${AA_SIM}/assets"
for pair in "embodiments.zip:embodiments" "objects.zip:objects" "background_texture.zip:background_texture"; do
  arc="${pair%%:*}"
  dir="${pair##*:}"
  if [[ -d "${AA_SIM}/assets/${dir}" ]]; then
    echo "  Already extracted: assets/${dir}"
    continue
  fi
  echo "  Downloading ${arc}..."
  curl -L -C - --fail --retry 8 --retry-delay 5 --connect-timeout 30 -o "${AA_SIM}/assets/${arc}" "${BASE_RT}/${arc}"
  unzip -q -o "${AA_SIM}/assets/${arc}" -d "${AA_SIM}/assets"
  rm -f "${AA_SIM}/assets/${arc}"
done

SIM_PY="${ACTIVEARENA_PYTHON:-python}"
(cd "${AA_SIM}" && "${SIM_PY}" ./script/update_embodiment_config_path.py)
(cd "${AA_SIM}" && env -u ALL_PROXY -u all_proxy "${SIM_PY}" ./script/download_activearena_assets.py)

echo "=== [4/4] Running IP-VLA ActiveArena preflight dry-run ==="
cd "${REPO_ROOT}"
python3 -m eval.activearena.run_eval --mode oracle_subtask_oft --tasks check_block_color --episodes 5 --dry-run
