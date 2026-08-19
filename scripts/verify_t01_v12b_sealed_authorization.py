#!/usr/bin/env python3
"""Verify the immutable v12b clean-GO authorization before sealed access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = (
    ROOT / "results/calibration/t01_open_and_observe_v12b_sealed_authorization.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    authorization = json.loads(AUTHORIZATION.read_text())
    if authorization["decision"] != "AUTHORIZED":
        raise ValueError("sealed audit is not authorized")
    if not all(authorization["clean_gates"].values()):
        raise ValueError("one or more clean gates did not pass")
    for key in ("clean_result", "clean_manifest", "frozen_composite"):
        reference = authorization[key]
        path = ROOT / reference["path"]
        observed = digest(path)
        if observed != reference["sha256"]:
            raise ValueError(
                f"sealed authorization dependency changed: {path}; "
                f"expected {reference['sha256']}, observed {observed}"
            )
    clean = json.loads((ROOT / authorization["clean_result"]["path"]).read_text())
    if clean["decision"] != "GO" or not all(clean["gates"].values()):
        raise ValueError("clean result no longer proves every preregistered gate")
    print(
        json.dumps(
            {
                "decision": "AUTHORIZED",
                "sealed_seed_block": authorization["sealed_seed_block"],
                "sealed_seeds": authorization["sealed_seeds"],
                "frozen_composite_sha256": authorization["frozen_composite"][
                    "sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
