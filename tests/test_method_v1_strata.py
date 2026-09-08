from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
)
from grounded_interaction.data import ObservedBranch
from grounded_interaction.method_v1_config import (
    load_method_v1_config,
    method_v1_information_stratum_counts,
    validate_method_v1_config,
)
from grounded_interaction.method_v1_data import (
    BranchScheduleEntry,
    CollectionAttempt,
    CollectionAttemptStatus,
    DecisionGroupManifest,
    InformationStratum,
    MethodV1OutcomeDataset,
    build_branch_schedule,
    make_method_v1_outcome_contract,
    primitive_outcome_coverage,
    validate_information_stratum_counts,
    validate_manifest_information_strata,
    validate_primitive_outcome_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "method_v1.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context(suffix: str) -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            PublicFrame(
                frame_id=f"agent-{suffix}",
                camera="agentview",
                frame_index=0,
                image_sha256=_digest(f"image-{suffix}"),
                width=320,
                height=240,
            ),
        ),
        proprioception=(0.0,) * 8,
    )


def _candidate(
    *,
    context: PolicyContext,
    candidate_id: str,
    primitive: Primitive,
    x: float,
) -> GroundedIntervention:
    frame = context.frames[0]
    referent = "middle drawer" if primitive is Primitive.OPEN else "butter"
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent=referent,
        parameters=(("canonical instruction", f"{primitive.value} {referent}"),),
        grounding=GroundingReference(
            camera=frame.camera,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(x - 0.1, 0.3, x + 0.1, 0.7),
            point_xy=(x, 0.5),
        ),
    )


def _manifest(
    suffix: str,
    *,
    split: str = "train",
    stratum: InformationStratum = InformationStratum.INFORMATION_NECESSARY,
) -> DecisionGroupManifest:
    context = _context(suffix)
    contract = make_method_v1_outcome_contract(
        continuation_policy_id=_digest("continuation"),
        executor_id="molmoact2-libero-pinned",
        serializer_id="grounded-precise-text-v1",
    )
    return DecisionGroupManifest(
        manifest_id=f"manifest-{suffix}",
        experiment_id="method-v1-development",
        initial_state_group=f"state-{suffix}",
        decision_group_id=f"decision-{suffix}",
        split_group_id=f"family-{suffix}",
        split=split,
        information_stratum=stratum,
        scene_id=f"scene-{suffix}",
        layout_id=f"layout-{suffix}",
        reset_state_sha256=_digest(f"reset-{suffix}"),
        context=context,
        candidates=(
            _candidate(
                context=context,
                candidate_id=f"direct-{suffix}",
                primitive=Primitive.DIRECT,
                x=0.3,
            ),
            _candidate(
                context=context,
                candidate_id=f"open-{suffix}",
                primitive=Primitive.OPEN,
                x=0.7,
            ),
        ),
        outcome_contract=contract,
        proposal_provider_id="qwen-provider",
        proposal_request_sha256=_digest(f"proposal-{suffix}"),
        token_provider_id="qwen-token-provider",
        token_cache_sha256=_digest(f"token-{suffix}"),
        configuration_sha256=_digest("configuration"),
        model_seeds=(101, 202),
    )


def _branch(
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
    *,
    outcome: bool,
) -> ObservedBranch:
    pre_frame = manifest.context.frames[0]
    post_frame = PublicFrame(
        frame_id=f"post-{entry.entry_id}",
        camera=pre_frame.camera,
        frame_index=1,
        image_sha256=_digest(f"post-{entry.entry_id}"),
        width=pre_frame.width,
        height=pre_frame.height,
    )
    return ObservedBranch(
        branch_id=f"branch-{entry.entry_id}",
        initial_state_group=entry.initial_state_group,
        decision_group_id=entry.decision_group_id,
        split=entry.split,
        reset_state_sha256=entry.reset_state_sha256,
        repeat_index=entry.repeat_index,
        context=manifest.context,
        executed_intervention=manifest.candidate(entry.candidate_id),
        post_action_frames=(post_frame,),
        outcome_contract=manifest.outcome_contract,
        observed_outcome=outcome,
        execution_status=ExecutionStatus.COMPLETED,
        execution_receipt_id=f"receipt-{entry.entry_id}",
    )


def _dataset(*, direct_failure_present: bool = True) -> MethodV1OutcomeDataset:
    manifest = _manifest("coverage")
    schedule = build_branch_schedule((manifest,), schedule_id="coverage-schedule")
    attempts = []
    for entry in schedule:
        candidate = manifest.candidate(entry.candidate_id)
        outcome = entry.repeat_index == 0
        if candidate.primitive is Primitive.DIRECT and not direct_failure_present:
            outcome = True
        attempts.append(
            CollectionAttempt(
                attempt_id=f"attempt-{entry.entry_id}",
                schedule_entry=entry,
                status=CollectionAttemptStatus.OUTCOME_EVALUATED,
                artifact_tree_sha256=_digest(f"artifact-{entry.entry_id}"),
                branch=_branch(manifest, entry, outcome=outcome),
            )
        )
    return MethodV1OutcomeDataset(
        dataset_id="coverage-dataset",
        manifests=(manifest,),
        schedule=schedule,
        attempts=tuple(attempts),
        complete=True,
    )


def test_config_freezes_exact_balanced_information_strata() -> None:
    config = load_method_v1_config(CONFIG)
    counts = method_v1_information_stratum_counts(config)
    assert counts["train"] == {
        "INFORMATION_NECESSARY": 67,
        "INFORMATION_SUFFICIENT": 67,
        "INFORMATION_ACTION_NO_HELP": 66,
    }
    assert counts["validation"] == {
        "INFORMATION_NECESSARY": 17,
        "INFORMATION_SUFFICIENT": 17,
        "INFORMATION_ACTION_NO_HELP": 16,
    }
    assert counts["calibration"] == {
        "INFORMATION_NECESSARY": 0,
        "INFORMATION_SUFFICIENT": 0,
        "INFORMATION_ACTION_NO_HELP": 0,
    }
    assert counts["test"] == {
        "INFORMATION_NECESSARY": 34,
        "INFORMATION_SUFFICIENT": 33,
        "INFORMATION_ACTION_NO_HELP": 33,
    }

    broken = load_method_v1_config(CONFIG)
    broken["collection"]["information_stratum_counts"]["train"][
        "INFORMATION_NECESSARY"
    ] = 66
    with pytest.raises(ValueError, match="sum to"):
        validate_method_v1_config(broken)


def test_count_contract_rejects_missing_strata_and_boolean_counts() -> None:
    config = load_method_v1_config(CONFIG)
    counts = config["collection"]["information_stratum_counts"]
    reset_groups = config["collection"]["reset_groups"]
    missing = {split: dict(values) for split, values in counts.items()}
    del missing["test"]["INFORMATION_ACTION_NO_HELP"]
    with pytest.raises(ValueError, match="must name exactly"):
        validate_information_stratum_counts(missing, reset_groups=reset_groups)

    boolean = {split: dict(values) for split, values in counts.items()}
    boolean["calibration"]["INFORMATION_NECESSARY"] = False
    with pytest.raises(ValueError, match="non-negative integers"):
        validate_information_stratum_counts(boolean, reset_groups=reset_groups)


def test_manifest_binds_stratum_outside_model_input_and_digest() -> None:
    necessary = _manifest("necessary")
    assert necessary.to_dict()["information_stratum"] == "INFORMATION_NECESSARY"
    assert set(necessary.model_input()) == {"context", "candidates"}
    assert "information_stratum" not in necessary.model_input()
    assert DecisionGroupManifest.from_mapping(necessary.to_dict()) == necessary

    sufficient = dataclasses.replace(
        necessary,
        information_stratum=InformationStratum.INFORMATION_SUFFICIENT,
    )
    assert sufficient.model_input() == necessary.model_input()
    assert sufficient.fingerprint() != necessary.fingerprint()

    invalid = necessary.to_dict()
    invalid["information_stratum"] = "UNKNOWN"
    with pytest.raises(ValueError, match="information_stratum"):
        DecisionGroupManifest.from_mapping(invalid)


def test_manifest_population_matches_pre_registered_strata() -> None:
    manifests = tuple(
        _manifest(stratum.value.lower(), stratum=stratum)
        for stratum in InformationStratum
    )
    expected = {
        "train": {stratum.value: 1 for stratum in InformationStratum},
        "validation": {stratum.value: 0 for stratum in InformationStratum},
        "calibration": {stratum.value: 0 for stratum in InformationStratum},
        "test": {stratum.value: 0 for stratum in InformationStratum},
    }
    assert (
        validate_manifest_information_strata(manifests, expected_counts=expected)
        == expected
    )
    with pytest.raises(ValueError, match="allocation differs"):
        validate_manifest_information_strata(manifests[:-1], expected_counts=expected)


def test_post_outcome_coverage_reports_and_validates_primitive_by_label() -> None:
    dataset = _dataset()
    report = primitive_outcome_coverage(dataset)
    train = report["splits"]["train"]
    assert train["primitive_by_outcome"] == {
        "DIRECT": {"failure": 1, "success": 1},
        "OPEN": {"failure": 1, "success": 1},
    }
    assert train["decision_groups_by_information_stratum"] == {
        "INFORMATION_NECESSARY": 1,
        "INFORMATION_SUFFICIENT": 0,
        "INFORMATION_ACTION_NO_HELP": 0,
    }
    assert (
        validate_primitive_outcome_coverage(dataset, required_splits=("train",))
        == report
    )

    with pytest.raises(ValueError, match="train:DIRECT:failure"):
        validate_primitive_outcome_coverage(
            _dataset(direct_failure_present=False),
            required_splits=("train",),
        )
