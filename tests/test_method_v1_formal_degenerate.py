from __future__ import annotations

from types import SimpleNamespace

import pytest

import grounded_interaction.evaluate_policy as evaluation
from grounded_interaction import method_v1_data, train_outcomes
from grounded_interaction.contracts import Primitive
from grounded_interaction.evaluate_policy import Baseline, BranchPrediction


def _rows_with_one_degenerate_primitive(
    *, degenerate_primitive: Primitive, degenerate_outcome: bool
) -> tuple[BranchPrediction, ...]:
    rows: list[BranchPrediction] = []
    for primitive_index, primitive in enumerate((Primitive.DIRECT, Primitive.OPEN)):
        outcomes = (
            (degenerate_outcome, degenerate_outcome)
            if primitive is degenerate_primitive
            else (False, True)
        )
        for repeat_index, observed_outcome in enumerate(outcomes):
            rows.append(
                BranchPrediction(
                    record_id=f"{primitive.value.lower()}-{repeat_index}",
                    decision_group_id="decision-0",
                    split_group_id="split-family-0",
                    candidate_id=f"candidate-{primitive.value.lower()}",
                    candidate_fingerprint=str(primitive_index + 1) * 64,
                    primitive=primitive,
                    repeat_index=repeat_index,
                    observed_outcome=observed_outcome,
                    scorer_probability=0.3 + 0.4 * primitive_index,
                    ensemble_probability=0.3 + 0.4 * primitive_index,
                    frozen_vlm_rank=primitive_index,
                )
            )
    return tuple(rows)


def _coverage_report(
    rows: tuple[BranchPrediction, ...],
) -> dict[str, object]:
    counts = {
        Primitive.DIRECT.value: {"failure": 0, "success": 0},
        Primitive.OPEN.value: {"failure": 0, "success": 0},
    }
    for row in rows:
        outcome = "success" if row.observed_outcome else "failure"
        counts[row.primitive.value][outcome] += 1
    return {
        "schema_version": "method-v1-primitive-outcome-coverage-v1",
        "splits": {
            "test": {
                "decision_groups_by_information_stratum": {
                    "INFORMATION_NECESSARY": 1,
                    "INFORMATION_SUFFICIENT": 0,
                    "INFORMATION_ACTION_NO_HELP": 0,
                },
                "outcome_evaluated_by_information_stratum": {
                    "INFORMATION_NECESSARY": len(rows),
                    "INFORMATION_SUFFICIENT": 0,
                    "INFORMATION_ACTION_NO_HELP": 0,
                },
                "primitive_by_outcome": counts,
                "outcome_evaluated": len(rows),
            }
        },
    }


@pytest.mark.parametrize(
    ("degenerate_primitive", "degenerate_outcome", "expected_status", "missing"),
    (
        (
            Primitive.DIRECT,
            False,
            "DEGENERATE_ALL_FAILURE",
            "test:DIRECT:success",
        ),
        (
            Primitive.DIRECT,
            True,
            "DEGENERATE_ALL_SUCCESS",
            "test:DIRECT:failure",
        ),
        (
            Primitive.OPEN,
            False,
            "DEGENERATE_ALL_FAILURE",
            "test:OPEN:success",
        ),
        (
            Primitive.OPEN,
            True,
            "DEGENERATE_ALL_SUCCESS",
            "test:OPEN:failure",
        ),
    ),
)
def test_formal_reports_retain_all_success_or_all_failure_primitives(
    monkeypatch: pytest.MonkeyPatch,
    degenerate_primitive: Primitive,
    degenerate_outcome: bool,
    expected_status: str,
    missing: str,
) -> None:
    rows = _rows_with_one_degenerate_primitive(
        degenerate_primitive=degenerate_primitive,
        degenerate_outcome=degenerate_outcome,
    )
    raw_coverage = _coverage_report(rows)
    artifact = SimpleNamespace(
        dataset_id="formal-dataset",
        dataset_sha256="a" * 64,
        artifact_sha256="b" * 64,
        checkpoints=(object(),),
    )
    dataset = SimpleNamespace(
        proposal_population_summary={
            "population_denominator_available": True,
            "metric_scope": "FULL_PREPROPOSAL_RESET_POPULATION",
            "by_split": {
                "test": {
                    "population_group_count": 2,
                    "proposal_covered_group_count": 1,
                    "choice_eligible_group_count": 1,
                    "single_primitive_group_count": 0,
                    "proposal_failure_group_count": 1,
                    "proposal_coverage": 0.5,
                    "choice_eligibility_rate": 0.5,
                }
            },
        }
    )

    monkeypatch.setattr(
        train_outcomes, "require_receipt_backed_dataset", lambda value: value
    )
    monkeypatch.setattr(
        method_v1_data, "primitive_outcome_coverage", lambda value: raw_coverage
    )
    monkeypatch.setattr(
        evaluation,
        "validate_canonical_prediction_artifact",
        lambda artifact_value, dataset_value, **kwargs: artifact_value,
    )
    monkeypatch.setattr(
        evaluation,
        "_canonical_rows_for_split",
        lambda artifact_value, dataset_value, **kwargs: (
            rows,
            {"decision-0": "split-family-0"},
        ),
    )

    probabilities = evaluation.evaluate_formal_probabilities(
        artifact,
        dataset,
        cache_roots=(),
        checkpoint_paths=(),
        num_bins=2,
    )
    matrix = evaluation.evaluate_formal_branch_matrix(
        artifact,
        dataset,
        cache_roots=(),
        checkpoint_paths=(),
        baseline=Baseline.SCORER,
        bootstrap_samples=10,
    )

    coverage = probabilities.primitive_outcome_coverage
    degenerate = coverage["primitive_label_support"][degenerate_primitive.value]
    other_primitive = (
        Primitive.OPEN if degenerate_primitive is Primitive.DIRECT else Primitive.DIRECT
    )
    other = coverage["primitive_label_support"][other_primitive.value]

    assert probabilities.metrics.count == len(rows)
    assert matrix.decision_group_count == 1
    assert matrix.primitive_outcome_coverage == coverage
    assert coverage["degenerate_label_support"] is True
    assert coverage["all_primitives_have_both_labels"] is False
    assert coverage["missing_outcome_cells"] == [missing]
    assert coverage["proposal_population_denominator_available"] is True
    assert coverage["proposal_population"]["proposal_coverage"] == 0.5
    assert coverage["probability_metric_scope"] == ("CONDITIONAL_ON_VALID_PROPOSAL_SET")
    assert degenerate == {
        "outcome_evaluated": 2,
        "both_labels_observed": False,
        "degenerate_label_support": True,
        "label_support_status": expected_status,
        "missing_outcomes": [missing.rsplit(":", 1)[-1]],
    }
    assert other["both_labels_observed"] is True
    assert other["degenerate_label_support"] is False
    assert other["label_support_status"] == "TWO_SIDED"
    assert other["missing_outcomes"] == []


def test_formal_coverage_still_rejects_non_test_split() -> None:
    with pytest.raises(ValueError, match="restricted to the test split"):
        evaluation._formal_primitive_outcome_coverage(object(), split="validation")
