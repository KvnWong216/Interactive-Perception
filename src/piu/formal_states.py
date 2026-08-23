"""Validation for opaque simulator states frozen before formal outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_state_archive(path: Path, *, state_key: str) -> tuple[list[int], str]:
    """Validate one numeric finite NPZ transport artifact without exposing it online."""

    if path.suffix != ".npz" or not path.is_file():
        raise ValueError(f"formal source state must be an existing NPZ: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {state_key}:
            raise ValueError(
                f"formal source state must contain exactly key {state_key!r}: {path}"
            )
        state = np.asarray(archive[state_key])
    if state.size == 0 or not np.issubdtype(state.dtype, np.number):
        raise ValueError(f"formal source state must be a nonempty numeric array: {path}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"formal source state contains nonfinite values: {path}")
    return list(state.shape), str(state.dtype)
