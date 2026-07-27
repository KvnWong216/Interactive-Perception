#!/usr/bin/env python3
"""Create a repository-local LIBERO config with portable absolute paths."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = PROJECT_ROOT / "third_party" / "LIBERO"
LIBERO_PACKAGE = LIBERO_ROOT / "libero" / "libero"
CONFIG_DIR = PROJECT_ROOT / ".libero"
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    if not LIBERO_PACKAGE.exists():
        raise FileNotFoundError(
            "LIBERO source was not found. Clone it to "
            f"{LIBERO_ROOT} before running this script."
        )

    config = {
        "assets": str(LIBERO_PACKAGE / "assets"),
        "bddl_files": str(LIBERO_PACKAGE / "bddl_files"),
        "benchmark_root": str(LIBERO_PACKAGE),
        "datasets": str(DATA_DIR),
        "init_states": str(LIBERO_PACKAGE / "init_files"),
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = CONFIG_DIR / "config.yaml"
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
