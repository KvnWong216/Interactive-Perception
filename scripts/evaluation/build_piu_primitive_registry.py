#!/usr/bin/env python3
"""Build a threshold-free primitive reliability registry from retained rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import reliability_record


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def raw_contact(
    retained: dict[str, Any], *, condition: str
) -> tuple[bool, dict[str, str]]:
    reference = retained["reports"][condition]
    path = ROOT / reference["path"]
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"raw primitive report hash drift: {path}")
    report = json.loads(path.read_text())
    if report["controller"].get("online_oracle_inputs"):
        raise ValueError("primitive evidence contains online oracle inputs")
    evaluator = report["evaluator"]
    target = evaluator["target_object"]
    return (
        int(evaluator["objects"][target]["grasp_contact_steps"]) > 0,
        {"path": portable(path), "sha256": reference["sha256"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/piu_primitive_registry_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("primitive registry artifacts are immutable")
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "piu.primitive-registry-protocol.v1":
        raise ValueError("unsupported primitive registry protocol")
    source_path = resolve(Path(config["source"]))
    source = json.loads(source_path.read_text())
    if source.get("schema_version") != "calibrated-interaction.paper-cycle-summary.v2":
        raise ValueError("unexpected primitive evidence source")
    open_values = []
    visible_pick_values = []
    post_open_pick_values = []
    place_values = []
    raw_sources = []
    seeds = []
    for retained in source["per_seed"]:
        seeds.append(int(retained["seed"]))
        open_values.append(bool(retained["open_closed_drawer"]["drawer_open"]))
        visible_contact, visible_source = raw_contact(
            retained, condition="direct_visible_cream_cheese"
        )
        post_open_contact, post_open_source = raw_contact(
            retained, condition="direct_after_actual_open"
        )
        visible_pick_values.append(visible_contact)
        post_open_pick_values.append(post_open_contact)
        place_values.append(
            bool(
                retained["direct_visible_cream_cheese"][
                    "target_destination_final"
                ]
            )
        )
        raw_sources.extend((visible_source, post_open_source))
    evaluated = {
        "OPEN": {
            "middle_drawer": {
                "status": "RETROSPECTIVE_PILOT_NOT_FORMALLY_QUALIFIED",
                "estimate": reliability_record(open_values),
                "success_definition": config["success_definitions"][
                    "OPEN.middle_drawer"
                ],
            }
        },
        "PICK": {
            "visible_work_surface": {
                "status": "RETROSPECTIVE_PILOT_NOT_FORMALLY_QUALIFIED",
                "estimate": reliability_record(visible_pick_values),
                "success_definition": config["success_definitions"][
                    "PICK.visible_work_surface"
                ],
            },
            "post_open_middle_drawer": {
                "status": "RETROSPECTIVE_PILOT_NOT_FORMALLY_QUALIFIED",
                "estimate": reliability_record(post_open_pick_values),
                "success_definition": config["success_definitions"][
                    "PICK.post_open_middle_drawer"
                ],
            },
        },
        "PLACE": {
            "visible_work_surface": {
                "status": "RETROSPECTIVE_PILOT_NOT_FORMALLY_QUALIFIED",
                "estimate": reliability_record(place_values),
                "success_definition": config["success_definitions"][
                    "PLACE.visible_work_surface"
                ],
            }
        },
    }
    result = {
        "schema_version": "piu.primitive-reliability-registry.v1",
        "claim_scope": config["claim_scope"],
        "scenario": config["scenario"],
        "config": {"path": portable(config_path), "sha256": sha256(config_path)},
        "source": {"path": portable(source_path), "sha256": sha256(source_path)},
        "seeds": seeds,
        "evaluated": evaluated,
        "not_evaluated": config["unevaluated_contexts"],
        "formal_qualification": config["formal_qualification"],
        "raw_contact_sources": raw_sources,
        "global_primitive_reliability_collapsed_across_contexts": False,
        "historical_count_gates_used": False,
        "paper_method_action_set_authorized": [],
        "paper_method_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
