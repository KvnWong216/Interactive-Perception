#!/usr/bin/env python3
"""Run the public RGB inference pipeline with the S03-v2 frontend adapter."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/perception"), str(ROOT / "src")]

import build_scene_packets as frozen_frontend  # noqa: E402
from build_scene_packets_v2 import PublicObjectFrontend  # noqa: E402
from interaction_uncertainty.grounding_dino_compat import (  # noqa: E402
    grounding_dino_post_process_identity,
)


def _argument_path(flag: str) -> Path:
    try:
        value = sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"S03 v2 backend requires {flag}") from exc
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _append_backend_identity(path: Path, identity: dict[str, str]) -> None:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"public backend artifact is not a mapping: {path}")
    if "grounding_dino_post_process" in value or "backend_identity" in value:
        raise ValueError("public backend artifact already contains an unfrozen v2 identity")
    if path.name == "report.json" or "pipeline" in value:
        value["backend_identity"] = {"grounding_dino_post_process": identity}
    else:
        value["grounding_dino_post_process"] = identity
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    # ``infer.py`` imports this already-loaded module name, so changing the
    # class binding selects v2 without changing the frozen v1 source bytes.
    frozen_frontend.PublicObjectFrontend = PublicObjectFrontend
    runpy.run_path(str(ROOT / "scripts/pipeline/infer.py"), run_name="__main__")
    import transformers

    identity = grounding_dino_post_process_identity(
        transformers.GroundingDinoProcessor.post_process_grounded_object_detection
    )
    output = _argument_path("--output")
    scene_packet = _argument_path("--asset-dir") / "scene_packet.json"
    _append_backend_identity(scene_packet, identity)
    _append_backend_identity(output, identity)


if __name__ == "__main__":
    main()
