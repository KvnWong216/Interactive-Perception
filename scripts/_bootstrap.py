"""Shared path and environment setup for the local LIBERO starter scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_SOURCE = PROJECT_ROOT / "third_party" / "LIBERO"
LOCAL_CONFIG = PROJECT_ROOT / ".libero"

if str(LIBERO_SOURCE) not in sys.path:
    sys.path.insert(0, str(LIBERO_SOURCE))

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LOCAL_CONFIG))
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()
