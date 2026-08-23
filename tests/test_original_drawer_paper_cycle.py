from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from calibrated_interaction.data import (
    CounterfactualSample,
    validate_group_splits,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/original_drawer_paper_cycle_v2.yaml"
RESULT = ROOT / "results/method/original_drawer_paper_cycle_v2.json"
RELABEL = ROOT / "results/method/original_drawer_open_cream_relabel_v2.json"
DIRECT_FINAL_RELABEL = (
    ROOT / "results/method/original_drawer_direct_cream_final_relabel_v2.json"
)
DATA = (
    ROOT
    / "data/calibrated_interaction/original_drawer_executed_v2/development.jsonl"
)
MANIFEST = DATA.with_name("development.manifest.json")
CV_RESULT = ROOT / "results/method/original_drawer_executed_effect_cv_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_physical_matrix_freezes_separate_information_and_execution_gates() -> None:
    result = json.loads(RESULT.read_text())
    assert sha256(CONFIG) == result["config"]["sha256"]
    assert result["online_oracle_input_count"] == 0
    aggregates = result["aggregates"]
    assert aggregates["direct_closed_butter"]["wrong_object_contact"]["successes"] == 9
    assert aggregates["direct_closed_butter"]["task_success"]["successes"] == 0
    assert aggregates["open_closed_drawer"]["drawer_open"]["successes"] == 9
    assert aggregates["open_closed_drawer"]["information_acquired"]["successes"] == 8
    assert aggregates["direct_after_actual_open"]["target_pick"]["successes"] == 0
    assert aggregates["direct_visible_cream_cheese"]["target_pick"]["successes"] == 10
    assert (
        aggregates["direct_visible_cream_cheese"]["target_destination"]["successes"]
        == 3
    )
    assert (
        aggregates["direct_visible_cream_cheese"]["target_destination_final"][
            "successes"
        ]
        == 3
    )
    assert result["stage_conversion"][
        "post_open_target_pick_given_information"
    ] == {
        "successes": 0,
        "trials": 8,
        "rate": 0.0,
        "wilson_95": [0.0, 0.32440756488388023],
    }
    assert result["overall_executor_gate_passed"] is False
    assert (
        result["paired_comparisons"][
            "open_information_vs_direct_closed_visibility"
        ]["exact_two_sided_p"]
        == 0.0078125
    )
    for row in result["per_seed"]:
        for source in row["reports"].values():
            assert sha256(ROOT / source["path"]) == source["sha256"]
            action = source["action_history"]
            assert sha256(ROOT / action["path"]) == action["sha256"]


def test_task_specific_relabel_is_offline_and_trajectory_preserving() -> None:
    result = json.loads(RELABEL.read_text())
    assert sha256(CONFIG) == result["config"]["sha256"]
    assert result["evaluator_only"] is True
    assert result["policy_rerun"] is False
    assert result["policy_inputs_changed"] is False
    assert len(result["rows"]) == 10
    for row in result["rows"]:
        for key in ("source_report", "source_action_history"):
            assert sha256(ROOT / row[key]["path"]) == row[key]["sha256"]
    direct = json.loads(DIRECT_FINAL_RELABEL.read_text())
    assert sha256(CONFIG) == direct["config"]["sha256"]
    assert direct["source_condition"] == "direct_visible_cream_cheese"
    assert sum(
        row["evaluator"]["target_in_destination_final"] for row in direct["rows"]
    ) == 3


def test_executed_effect_dataset_is_grouped_and_policy_clean() -> None:
    rows = read_jsonl(DATA)
    manifest = json.loads(MANIFEST.read_text())
    assert sha256(CONFIG) == manifest["sources"]["config"]["sha256"]
    assert len(rows) == manifest["rows"] == 60
    assert sha256(DATA) == manifest["dataset"]["sha256"]
    assert manifest["initial_state_groups"] == 10
    assert manifest["online_oracle_input_count"] == 0
    assert set(manifest["constant_unsupported_factors"]) == {
        "candidate_rejected",
        "region_confirmed_empty",
    }
    validate_group_splits(rows)
    counts = Counter(row["initial_state_id"] for row in rows)
    assert set(counts.values()) == {6}
    for row in rows:
        sample = CounterfactualSample.from_mapping(row)
        assert len(sample.observation_frames) == 2
        assert sample.policy_input().keys() == {
            "prompt",
            "observation_frames",
            "history",
            "candidate_actions",
        }
        for path in (*sample.observation_frames, *sample.post_action_frames):
            assert (ROOT / path).is_file()


def test_cpu_group_cv_rejects_effect_head_as_route_contribution() -> None:
    result = json.loads(CV_RESULT.read_text())
    assert result["status"] == "DEVELOPMENT_GROUP_CV_ONLY"
    assert result["device"] == "cpu"
    assert result["gpu_visible_to_process"] is False
    assert result["formal_calibration_claim"] is False
    assert result["decision"] == "REJECT_EFFECT_HEAD_AS_ROUTE_CONTRIBUTION"
    comparison = result["effect_route_comparison"]
    assert comparison["mean_delta"] == 0.0
    assert comparison["tied_folds"] == 5
    assert result["summary"]["B6_route_only"]["route"]["accuracy"]["mean"] == 1.0
    assert (
        result["summary"]["B7_executed_effect_route"]["route"]["accuracy"]["mean"]
        == 1.0
    )
    assert (
        result["summary"]["B7_executed_effect_route"]["prompt_ablation_accuracy"][
            "prompt_removed"
        ]["mean"]
        == 0.5
    )
    for source in result["sources"].values():
        assert sha256(ROOT / source["path"]) == source["sha256"]
