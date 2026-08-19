#!/usr/bin/env python3
"""Verify frozen train/cal PIU artifacts before clean scene access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORIZATION = (
    ROOT / "results/calibration/piu_v1_replacement_clean_authorization.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_authorization(path: Path = DEFAULT_AUTHORIZATION) -> dict[str, Any]:
    if not path.is_absolute():
        path = ROOT / path
    authorization = json.loads(path.read_text())
    if authorization["decision"] != "AUTHORIZED":
        raise ValueError("clean scene-disjoint development is not authorized")
    if not all(authorization["pre_clean_gates"].values()):
        raise ValueError("one or more pre-clean gates did not pass")
    for group in ("frozen_artifacts", "frozen_source"):
        for name, reference in authorization[group].items():
            dependency = ROOT / reference["path"]
            observed = digest(dependency)
            if observed != reference["sha256"]:
                raise ValueError(
                    f"clean authorization dependency changed ({name}): {dependency}; "
                    f"expected {reference['sha256']}, observed {observed}"
                )

    protocol_path = ROOT / authorization["frozen_artifacts"]["protocol"]["path"]
    protocol = yaml.safe_load(protocol_path.read_text())
    clean = protocol["splits"]["clean_scene_disjoint_development"]
    if clean["open_now"] is not False:
        raise ValueError("clean split must remain closed in the protocol")
    if str(clean["seeds"]) != authorization["clean_seeds"]:
        raise ValueError("clean authorization seed range differs from protocol")
    if "prototype_train" in protocol["splits"]:
        train_paths = {
            item["bddl"] for item in protocol["splits"]["prototype_train"]["scenarios"]
        }
        calibration_paths = {
            item["bddl"]
            for item in protocol["splits"]["conformal_calibration"]["scenarios"]
        }
    else:
        train_paths = set(protocol["provenance_scenario_paths"]["train"])
        calibration_paths = set(protocol["provenance_scenario_paths"]["calibration"])
    clean_paths = {item["bddl"] for item in clean["scenarios"]}
    if train_paths & calibration_paths or train_paths & clean_paths or calibration_paths & clean_paths:
        raise ValueError("scenario paths overlap across train/calibration/clean")

    training_path = ROOT / authorization["frozen_artifacts"]["training_report"]["path"]
    training = json.loads(training_path.read_text())
    calibration = training["calibration_metrics"]
    if training.get("online_oracle_inputs"):
        raise ValueError("training report contains online oracle inputs")
    if calibration["location"]["coverage"] < 0.90:
        raise ValueError("calibration location coverage is below 0.90")
    if calibration["action"]["coverage"] < 0.90:
        raise ValueError("calibration action coverage is below 0.90")
    if min(
        value["accuracy"] for value in calibration["location"]["per_class"].values()
    ) < 0.90:
        raise ValueError("calibration location class accuracy is below 0.90")
    if min(
        value["accuracy"] for value in calibration["action"]["per_class"].values()
    ) < 0.90:
        raise ValueError("calibration action class accuracy is below 0.90")
    if calibration["optimizer_accuracy"] < 0.90:
        raise ValueError("calibration optimizer accuracy is below 0.90")
    for name in ("train_frontend_audit", "calibration_frontend_audit"):
        audit_path = ROOT / authorization["frozen_artifacts"][name]["path"]
        audit = json.loads(audit_path.read_text())
        if audit["query_leaks"] != 0 or audit["truth_node_proposal_rate"] < 1.0:
            raise ValueError(f"frontend audit is not clean: {name}")
    if "failed_clean_attempt" in authorization["frozen_artifacts"]:
        failed_path = ROOT / authorization["frozen_artifacts"]["failed_clean_attempt"]["path"]
        failed = json.loads(failed_path.read_text())
        if failed["decision"] != "NOT-GO" or failed["model_inference_performed"]:
            raise ValueError("invalid provenance for the retired clean attempt")
    for name, reference in authorization["frozen_artifacts"].items():
        if name.startswith("hidden_") and name.endswith("_preflight"):
            smoke_path = ROOT / reference["path"]
            if not json.loads(smoke_path.read_text())["none_resolvable"]:
                raise ValueError(f"replacement hidden visibility preflight failed: {name}")
        if name.startswith("visible_") and name.endswith("_preflight"):
            smoke_path = ROOT / reference["path"]
            if not json.loads(smoke_path.read_text())["all_resolvable"]:
                raise ValueError(f"replacement visible visibility preflight failed: {name}")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    args = parser.parse_args()
    authorization = verify_authorization(args.authorization)
    print(
        json.dumps(
            {
                "decision": authorization["decision"],
                "clean_seed_block": authorization["clean_seed_block"],
                "clean_seeds": authorization["clean_seeds"],
                "model_sha256": authorization["frozen_artifacts"]["model"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
