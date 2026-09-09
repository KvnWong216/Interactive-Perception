from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grounded_interaction.collect_outcomes import (
    SceneResetSpec,
    VerifiedDecisionFreeze,
    _inventory_binding_for_scene_spec,
    _proposal_population_from_terminals,
    _validate_serialized_proposal_population,
    freeze_collection_plan,
    freeze_reset_inventory,
    load_verified_collection_plan,
    load_verified_reset_inventory,
)
from grounded_interaction.contracts import canonical_sha256
from grounded_interaction.train_outcomes import (
    _collection_plan_proposal_population_summary,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, tuple[Path, Path, Path]]:
    source_config = json.loads(
        (Path(__file__).parents[1] / "experiments" / "method_v1.yaml").read_text()
    )
    source_config["collection"]["reset_groups"] = {
        "train": 3,
        "validation": 0,
        "calibration": 0,
        "test": 0,
    }
    source_config["collection"]["information_stratum_counts"] = {
        "train": {
            "INFORMATION_NECESSARY": 1,
            "INFORMATION_SUFFICIENT": 1,
            "INFORMATION_ACTION_NO_HELP": 1,
        },
        "validation": {
            "INFORMATION_NECESSARY": 0,
            "INFORMATION_SUFFICIENT": 0,
            "INFORMATION_ACTION_NO_HELP": 0,
        },
        "calibration": {
            "INFORMATION_NECESSARY": 0,
            "INFORMATION_SUFFICIENT": 0,
            "INFORMATION_ACTION_NO_HELP": 0,
        },
        "test": {
            "INFORMATION_NECESSARY": 0,
            "INFORMATION_SUFFICIENT": 0,
            "INFORMATION_ACTION_NO_HELP": 0,
        },
    }
    config_path = tmp_path / "method_v1.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    bddl = tmp_path / "scene.bddl"
    bddl.write_text("(define (problem public-reset))", encoding="utf-8")
    paths: list[Path] = []
    for index, stratum in enumerate(
        (
            "INFORMATION_NECESSARY",
            "INFORMATION_SUFFICIENT",
            "INFORMATION_ACTION_NO_HELP",
        )
    ):
        init_state = tmp_path / f"init-{index}.npz"
        init_state.write_bytes(f"finite-reset-{index}".encode())
        spec = {
            "schema_version": "method-v1-scene-reset-spec-v2",
            "scene_id": f"scene-{index}",
            "layout_id": "layout-a",
            "initial_state_group": f"reset-{index}",
            "decision_group_id": f"decision-{index}",
            "split_group_id": f"hidden-family-{index}",
            "split": "train",
            "information_stratum": stratum,
            "prompt": "Place the requested object in the basket.",
            "bddl_file": bddl.name,
            "init_states_file": init_state.name,
            "init_state_index": 0,
            "env_seed": index,
            "reset_state_sha256": str(index + 1) * 64,
            "evaluator_predicate": ["In", f"object_{index}", "basket_region"],
            "model_seeds": [2 * index, 2 * index + 1],
            "image_size": 256,
            "settle_steps": 10,
            "control_mode": "relative",
        }
        path = tmp_path / f"scene-{index}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        paths.append(path)
    return config_path, (paths[0], paths[1], paths[2])


def test_reset_inventory_freezes_and_reopens_exact_public_population(
    tmp_path: Path,
) -> None:
    config, specs = _write_inputs(tmp_path)
    output = tmp_path / "reset_inventory.json"
    document = freeze_reset_inventory(
        scene_spec_paths=tuple(reversed(specs)),
        config_path=config,
        inventory_id="method-v1-population",
        output=output,
    )

    assert document["row_count"] == 3
    assert [row["decision_group_id"] for row in document["rows"]] == [
        "decision-0",
        "decision-1",
        "decision-2",
    ]
    assert all("evaluator_predicate" not in row for row in document["rows"])
    assert all("bddl_file" not in row for row in document["rows"])
    verified = load_verified_reset_inventory(
        output,
        config_path=config,
        scene_spec_paths=specs,
    )
    assert verified.inventory_sha256 == document["reset_inventory_sha256"]
    assert len({row["inventory_row_sha256"] for row in verified.rows}) == 3

    with pytest.raises(FileExistsError):
        freeze_reset_inventory(
            scene_spec_paths=specs,
            config_path=config,
            inventory_id="method-v1-population",
            output=output,
        )


def test_inventory_rejects_source_drift_and_nonmember(tmp_path: Path) -> None:
    config, specs = _write_inputs(tmp_path)
    output = tmp_path / "reset_inventory.json"
    document = freeze_reset_inventory(
        scene_spec_paths=specs,
        config_path=config,
        inventory_id="method-v1-population",
        output=output,
    )
    original = json.loads(specs[0].read_text())
    changed = copy.deepcopy(original)
    changed["prompt"] = "A changed task prompt."
    specs[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from re-opened scene specs"):
        load_verified_reset_inventory(
            output,
            config_path=config,
            scene_spec_paths=specs,
        )
    with pytest.raises(ValueError, match="not an exact member"):
        _inventory_binding_for_scene_spec(
            inventory_path=output,
            config_path=config,
            spec=SceneResetSpec.from_mapping(changed),
            spec_path=specs[0],
        )

    specs[0].write_text(json.dumps(original), encoding="utf-8")
    binding = _inventory_binding_for_scene_spec(
        inventory_path=output,
        config_path=config,
        spec=SceneResetSpec.from_mapping(original),
        spec_path=specs[0],
    )
    assert binding["reset_inventory_sha256"] == document["reset_inventory_sha256"]


def test_inventory_rejects_replacement_reset_and_tampering(tmp_path: Path) -> None:
    config, specs = _write_inputs(tmp_path)
    duplicate = json.loads(specs[1].read_text())
    duplicate["decision_group_id"] = "decision-replacement"
    duplicate["initial_state_group"] = "reset-replacement"
    duplicate["split_group_id"] = "hidden-family-replacement"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate physical_reset"):
        freeze_reset_inventory(
            scene_spec_paths=(*specs, replacement),
            config_path=config,
            inventory_id="bad-population",
            output=tmp_path / "bad.json",
        )

    output = tmp_path / "reset_inventory.json"
    freeze_reset_inventory(
        scene_spec_paths=specs,
        config_path=config,
        inventory_id="method-v1-population",
        output=output,
    )
    tampered = json.loads(output.read_text())
    tampered["rows"][0]["prompt"] = "tampered"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_verified_reset_inventory(output, config_path=config)


def _inventory_freeze(
    *,
    root: Path,
    row: dict[str, object],
    inventory_sha256: str,
    status: str,
    marker: str,
) -> VerifiedDecisionFreeze:
    root.mkdir()
    manifest_sha256 = marker * 64
    manifest = SimpleNamespace(
        scene_id=row["scene_id"],
        layout_id=row["layout_id"],
        initial_state_group=row["initial_state_group"],
        decision_group_id=row["decision_group_id"],
        split_group_id=row["split_group_id"],
        split=row["split"],
        information_stratum=SimpleNamespace(value=row["information_stratum"]),
        context=SimpleNamespace(prompt=row["prompt"]),
        reset_state_sha256=row["reset_state_sha256"],
        model_seeds=tuple(row["model_seeds"]),
        fingerprint=lambda: manifest_sha256,
    )
    return VerifiedDecisionFreeze(
        freeze_dir=root.resolve(),
        manifest=manifest,
        schedule=(),
        freeze_receipt_sha256=marker * 64,
        reset_inventory_sha256=inventory_sha256,
        inventory_row_sha256=str(row["inventory_row_sha256"]),
        proposal_population_status=status,
    )


def _write_proposal_failure(
    root: Path,
    *,
    row: dict[str, object],
    inventory_sha256: str,
) -> None:
    root.mkdir()
    body = {
        "schema_version": "method-v1-proposal-failure-v2",
        "status": "PROPOSAL_FAILURE",
        "scene_spec_sha256": row["scene_spec_sha256"],
        "context_fingerprint": "public-context",
        "proposer_id": "qwen-proposer",
        "failure": {"code": "no_valid_candidate"},
        "runtime_source_sha256": "f" * 64,
        "reset_inventory_sha256": inventory_sha256,
        "inventory_row_sha256": row["inventory_row_sha256"],
    }
    document = {**body, "proposal_failure_sha256": canonical_sha256(body)}
    (root / "proposal_failure.json").write_text(json.dumps(document), encoding="utf-8")


def test_population_requires_one_terminal_per_preproposal_reset(tmp_path: Path) -> None:
    config, specs = _write_inputs(tmp_path)
    inventory_path = tmp_path / "reset_inventory.json"
    freeze_reset_inventory(
        scene_spec_paths=specs,
        config_path=config,
        inventory_id="method-v1-population",
        output=inventory_path,
    )
    inventory = load_verified_reset_inventory(inventory_path, config_path=config)
    rows = [dict(row) for row in inventory.rows]
    freezes = (
        _inventory_freeze(
            root=tmp_path / "choice",
            row=rows[0],
            inventory_sha256=inventory.inventory_sha256,
            status="VALID_CHOICE_SET",
            marker="a",
        ),
        _inventory_freeze(
            root=tmp_path / "single",
            row=rows[1],
            inventory_sha256=inventory.inventory_sha256,
            status="VALID_SINGLE_PRIMITIVE_SET",
            marker="b",
        ),
    )
    failed = tmp_path / "failed"
    _write_proposal_failure(
        failed,
        row=rows[2],
        inventory_sha256=inventory.inventory_sha256,
    )

    with pytest.raises(ValueError, match="without a proposal terminal"):
        _proposal_population_from_terminals(
            inventory=inventory,
            freezes=freezes,
            proposal_failure_dirs=(),
        )
    population = _proposal_population_from_terminals(
        inventory=inventory,
        freezes=freezes,
        proposal_failure_dirs=(failed,),
    )
    assert population["population_group_count"] == 3
    assert population["proposal_covered_group_count"] == 2
    assert population["choice_eligible_group_count"] == 1
    assert population["status_counts"] == {
        "VALID_CHOICE_SET": 1,
        "VALID_SINGLE_PRIMITIVE_SET": 1,
        "PROPOSAL_FAILURE": 1,
    }
    _validate_serialized_proposal_population(population, freezes=freezes)
    with pytest.raises(ValueError, match="every serialized VALID"):
        _validate_serialized_proposal_population(
            population,
            freezes=freezes[:1],
        )

    forged = copy.deepcopy(population)
    forged["terminal_rows"][2]["manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="must not claim"):
        _validate_serialized_proposal_population(forged, freezes=freezes)

    conflicting_failure = tmp_path / "conflicting-failure"
    _write_proposal_failure(
        conflicting_failure,
        row=rows[0],
        inventory_sha256=inventory.inventory_sha256,
    )
    with pytest.raises(ValueError, match="duplicate/replacement terminals"):
        _proposal_population_from_terminals(
            inventory=inventory,
            freezes=freezes,
            proposal_failure_dirs=(failed, conflicting_failure),
        )


def test_population_summary_exposes_unconditional_and_legacy_scope() -> None:
    terminal_rows = [
        {
            "split": "test",
            "terminal_status": "VALID_CHOICE_SET",
        },
        {
            "split": "test",
            "terminal_status": "VALID_SINGLE_PRIMITIVE_SET",
        },
        {"split": "test", "terminal_status": "PROPOSAL_FAILURE"},
    ]
    summary = _collection_plan_proposal_population_summary(
        {
            "proposal_population": {
                "reset_inventory_sha256": "a" * 64,
                "population_group_count": 3,
                "proposal_covered_group_count": 2,
                "choice_eligible_group_count": 1,
                "terminal_rows": terminal_rows,
            }
        }
    )
    assert summary["population_denominator_available"] is True
    assert summary["overall"]["proposal_coverage"] == pytest.approx(2 / 3)
    assert summary["by_split"]["test"]["proposal_failure_group_count"] == 1

    legacy = _collection_plan_proposal_population_summary({"groups": []})
    assert legacy["population_denominator_available"] is False
    assert legacy["metric_scope"] == "POST_PROPOSAL_CONDITIONAL_LEGACY_PLAN"


def test_all_proposal_failures_still_freeze_zero_coverage_plan(
    tmp_path: Path,
) -> None:
    config, specs = _write_inputs(tmp_path)
    inventory_path = tmp_path / "reset_inventory.json"
    freeze_reset_inventory(
        scene_spec_paths=specs,
        config_path=config,
        inventory_id="all-failed-population",
        output=inventory_path,
    )
    inventory = load_verified_reset_inventory(inventory_path, config_path=config)
    failures: list[Path] = []
    for index, row in enumerate(inventory.rows):
        root = tmp_path / f"failed-{index}"
        _write_proposal_failure(
            root,
            row=dict(row),
            inventory_sha256=inventory.inventory_sha256,
        )
        failures.append(root)
    key = tmp_path / "scorer.key"
    key.write_bytes(b"a sufficiently long test-only authentication key")
    plan_path = tmp_path / "all-failed-plan.json"
    plan = freeze_collection_plan(
        freeze_dirs=(),
        config_path=config,
        plan_id="all-failed-plan",
        scorer_auth_key_path=key,
        output=plan_path,
        reset_inventory_path=inventory_path,
        proposal_failure_dirs=failures,
    )

    assert plan["group_count"] == 0
    assert plan["schedule_entry_count"] == 0
    assert plan["proposal_population"]["population_group_count"] == 3
    assert plan["proposal_population"]["proposal_covered_group_count"] == 0
    assert plan["proposal_population"]["status_counts"] == {
        "VALID_CHOICE_SET": 0,
        "VALID_SINGLE_PRIMITIVE_SET": 0,
        "PROPOSAL_FAILURE": 3,
    }
    reopened = load_verified_collection_plan(
        plan_path,
        freeze_dirs=(),
        config_path=config,
    )
    assert reopened.collection_plan_sha256 == plan["collection_plan_sha256"]
