#!/usr/bin/env python3
"""Run the S03 v3 processor/config readiness probe without inference or writes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_v3_amendment import (  # noqa: E402
    V3_CERTIFICATE_PATH,
    V3_OUTPUT_ROOT,
    V3_RUNTIME_READINESS_PATH,
    build_s03_v3_runtime_readiness,
    probe_s03_v3_backend_readiness,
    validate_s03_v3_runtime_readiness,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-readiness", type=Path, default=V3_RUNTIME_READINESS_PATH)
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the deterministic prospective readiness JSON to stdout",
    )
    args = parser.parse_args()
    if (ROOT / V3_OUTPUT_ROOT).exists() or (ROOT / V3_CERTIFICATE_PATH).exists():
        raise RuntimeError("backend readiness preflight is authorized only before v3 execution")
    if args.render:
        value = build_s03_v3_runtime_readiness(repository_root=ROOT)
        status = "RENDERED_WITHOUT_WRITE"
    else:
        path = _resolve(args.runtime_readiness)
        value = validate_s03_v3_runtime_readiness(path, repository_root=ROOT)
        status = "VERIFIED"
    live = probe_s03_v3_backend_readiness(repository_root=ROOT)
    if live != value["backend_readiness"]:
        raise ValueError("live S03 v3 readiness differs from the frozen contract")
    print(
        json.dumps(
            {
                "status": status,
                "execution_version": value["execution_version"],
                "sentencepiece": live["package_versions"]["sentencepiece"],
                "protobuf": live["package_versions"]["protobuf"],
                "siglip_tokenizer": live["siglip_tokenizer"],
                "grounding_dino_branch": live["grounding_dino_processor_api"][
                    "score_threshold_keyword"
                ],
                "processors": live["processor_classes"],
                "configs": live["config_classes"],
                "model_weights_loaded": False,
                "inference_executed": False,
                "outcome_present": False,
                "files_written": False,
                "paper_claim_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
