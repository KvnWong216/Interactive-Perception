#!/usr/bin/env python3
"""Check the simulation-only LIBERO installation."""

from __future__ import annotations

import os
import platform
import sys

from _bootstrap import PROJECT_ROOT

import mujoco
import numpy as np
import robosuite
import torch
from libero.libero import benchmark


def main() -> None:
    suites = sorted(benchmark.get_benchmark_dict().keys())
    print("LIBERO simulation installation check")
    print(f"  project:    {PROJECT_ROOT}")
    print(f"  machine:    {platform.machine()}")
    print(f"  python:     {sys.version.split()[0]}")
    print(f"  numpy:      {np.__version__}")
    print(f"  torch:      {torch.__version__}")
    print(f"  mujoco:     {mujoco.__version__}")
    print(f"  robosuite:  {robosuite.__version__}")
    print(f"  MUJOCO_GL:  {os.environ.get('MUJOCO_GL')}")
    print(f"  suites:     {', '.join(suites)}")

    if sys.platform == "darwin":
        assert (
            platform.machine() == "arm64"
        ), "The tested macOS setup expects a native Apple-silicon Python."
    assert "libero_object" in suites, "LIBERO benchmark registry was not loaded."
    print("OK: imports, architecture, and benchmark registry are ready.")


if __name__ == "__main__":
    main()
