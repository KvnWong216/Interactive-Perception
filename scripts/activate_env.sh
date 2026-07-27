#!/bin/zsh

# Source from any directory:
#   source scripts/activate_env.sh

SCRIPT_PATH="${(%):-%N}"
SCRIPT_DIR="${SCRIPT_PATH:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
ENV_PREFIX="${PROJECT_ROOT}/.conda/envs/libero-mac"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    /opt/anaconda3/etc/profile.d/conda.sh \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      source "${candidate}"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniforge/Anaconda or initialize conda for zsh."
  return 1
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Environment not found: ${ENV_PREFIX}"
  echo "Follow the Environment setup section in README.md first."
  return 1
fi

conda activate "${ENV_PREFIX}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero"
if [[ -z "${MUJOCO_GL:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export MUJOCO_GL="cgl"
  else
    export MUJOCO_GL="egl"
  fi
fi
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/.cache/numba"

mkdir -p \
  "${MPLCONFIGDIR}" \
  "${PIP_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${PROJECT_ROOT}/outputs" \
  "${PROJECT_ROOT}/data"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  python "${SCRIPT_DIR}/setup_libero_config.py"
fi

cd "${PROJECT_ROOT}"
echo "LIBERO environment active: ${CONDA_PREFIX}"
echo "Project root: ${PROJECT_ROOT}"
