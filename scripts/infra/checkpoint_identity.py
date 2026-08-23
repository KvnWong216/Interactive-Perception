"""Content identity for a checkpoint directory without loading the model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "piu.checkpoint-tree-sha256.v1"


def checkpoint_identity(
    directory: Path, *, chunk_bytes: int = 8 * 1024 * 1024
) -> dict[str, Any]:
    """Hash relative paths, sizes, and bytes for every checkpoint file."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {root}")
    tree_hash = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        total_bytes += size
        tree_hash.update(len(relative).to_bytes(8, "big"))
        tree_hash.update(relative)
        tree_hash.update(size.to_bytes(16, "big"))
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                file_hash.update(chunk)
        tree_hash.update(file_hash.digest())
    return {
        "schema_version": SCHEMA_VERSION,
        "sha256": tree_hash.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
