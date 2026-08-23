from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from piu.splits import LEARNING_SPLIT_ROLES, load_split_manifest

ROOT = Path(__file__).resolve().parents[1]


def _write_split(path: Path, roles: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "required_roles": list(roles),
                "assignments": [
                    {
                        "initial_state_group": f"excluded-{role}",
                        "seed": 200 + index,
                        "split_role": role,
                    }
                    for index, role in enumerate(roles)
                ],
            }
        )
    )


def test_learning_split_comes_only_from_external_budget_and_exclusions(
    tmp_path: Path,
) -> None:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.learning-collection-budget.v1",
                "status": "FROZEN_BEFORE_SUCCESSOR_COLLECTION",
                "scenario": "fixed drawer",
                "groups_per_role": {role: 1 for role in LEARNING_SPLIT_ROLES},
                "seed_start": 1400,
                "authority": "test resource owner",
                "rationale": "bounded test collection",
                "outcomes_loaded": False,
                "model_predictions_loaded": False,
            },
            sort_keys=False,
        )
    )
    qualification = tmp_path / "qualification.json"
    oracle = tmp_path / "oracle.json"
    _write_split(qualification, ("primitive_qualification",))
    _write_split(oracle, ("oracle_formal",))
    output = tmp_path / "learning.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_learning_split_manifest.py"),
            "--budget",
            str(budget),
            "--exclude-split",
            str(qualification),
            "--exclude-split",
            str(oracle),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = load_split_manifest(output)
    assert result["required_roles"] == list(LEARNING_SPLIT_ROLES)
    assert {row["seed"] for row in result["assignments"]}.isdisjoint(range(1400, 1410))


def test_main_formal_split_count_is_read_from_plan_not_cli(tmp_path: Path) -> None:
    prior_roles = (
        *LEARNING_SPLIT_ROLES,
        "oracle_formal",
        "primitive_qualification",
    )
    prior = tmp_path / "prior.json"
    _write_split(prior, prior_roles)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-paired-test-plan.v1",
                "status": "PROSPECTIVE_GROUP_COUNT_FROZEN",
                "design": {"prospective_group_count": 2},
            }
        )
    )
    output = tmp_path / "formal.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_planned_split_manifest.py"),
            "--purpose",
            "sealed_test",
            "--plan",
            str(plan),
            "--seed-start",
            "500",
            "--group-prefix",
            "sealed-main",
            "--exclude-split",
            str(prior),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = load_split_manifest(output)
    assert result["required_roles"] == ["sealed_test"]
    assert len(result["assignments"]) == 2
