from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

import grounded_interaction.collect_outcomes as collector_module
from grounded_interaction.collect_outcomes import (
    VerifiedDecisionFreeze,
    freeze_collection_plan,
    load_verified_collection_plan,
)
from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
)
from grounded_interaction.data import ObservedBranch
from grounded_interaction.method_v1_data import (
    BranchScheduleEntry,
    CollectionAttempt,
    CollectionAttemptStatus,
    DecisionGroupManifest,
    InformationStratum,
    InfrastructureFailure,
    MethodV1OutcomeDataset,
    ProposalRejection,
    build_branch_schedule,
    executed_only_target,
    make_method_v1_outcome_contract,
    validate_branch_schedule,
    validate_collection_attempts,
    validate_manifest_splits,
)


def _digest(character: str) -> str:
    return character * 64


def _frame(frame_id: str, camera: str, index: int, character: str) -> PublicFrame:
    return PublicFrame(
        frame_id=frame_id,
        camera=camera,
        frame_index=index,
        image_sha256=_digest(character),
        width=320,
        height=240,
    )


def _context() -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            _frame("agent-0", "agentview", 0, "a"),
            _frame("wrist-0", "wrist", 0, "b"),
        ),
        proprioception=(0.0,) * 8,
    )


def _candidate(
    candidate_id: str,
    primitive: Primitive,
    *,
    context: PolicyContext | None = None,
    x: float = 0.5,
) -> GroundedIntervention:
    current = context or _context()
    frame = max(
        (item for item in current.frames if item.camera == "agentview"),
        key=lambda item: item.frame_index,
    )
    grounding = GroundingReference(
        camera=frame.camera,
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        image_sha256=frame.image_sha256,
        box_xyxy=(x - 0.1, 0.3, x + 0.1, 0.7),
        point_xy=(x, 0.5),
    )
    referent = "middle drawer" if primitive is Primitive.OPEN else "butter package"
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent=referent,
        parameters=(("canonical instruction", f"{primitive.value} {referent}"),),
        grounding=grounding,
    )


def _manifest(
    *,
    manifest_id: str = "manifest-001",
    decision_group_id: str = "decision-001",
    initial_state_group: str = "state-001",
    split_group_id: str = "hidden-pair-001",
    split: str = "train",
    scene_id: str = "drawer-scene",
    layout_id: str = "layout-a",
) -> DecisionGroupManifest:
    context = _context()
    contract = make_method_v1_outcome_contract(
        continuation_policy_id=_digest("c"),
        executor_id="molmoact2-libero-pinned",
        serializer_id="grounded-precise-text-v1",
    )
    return DecisionGroupManifest(
        manifest_id=manifest_id,
        experiment_id="method-v1-development",
        initial_state_group=initial_state_group,
        decision_group_id=decision_group_id,
        split_group_id=split_group_id,
        information_stratum=InformationStratum.INFORMATION_NECESSARY,
        split=split,
        scene_id=scene_id,
        layout_id=layout_id,
        reset_state_sha256=_digest("d"),
        context=context,
        candidates=(
            _candidate("direct-butter", Primitive.DIRECT, context=context, x=0.3),
            _candidate("open-drawer", Primitive.OPEN, context=context, x=0.7),
        ),
        outcome_contract=contract,
        proposal_provider_id="qwen25vl-provider-revision-x",
        proposal_request_sha256=_digest("e"),
        token_provider_id="qwen25vl-token-provider-revision-x",
        token_cache_sha256=_digest("f"),
        configuration_sha256=_digest("0"),
        model_seeds=(101, 202),
        proposal_rejections=(
            ProposalRejection(
                proposal_index=2,
                reason_code="zero_area_bbox",
                proposal_sha256=_digest("1"),
            ),
        ),
    )


def _post_frame(branch_suffix: str) -> PublicFrame:
    return _frame(f"agent-post-{branch_suffix}", "agentview", 1, "2")


def _branch(
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
    *,
    outcome: bool,
) -> ObservedBranch:
    return ObservedBranch(
        branch_id=f"branch-{entry.execution_index}",
        initial_state_group=entry.initial_state_group,
        decision_group_id=entry.decision_group_id,
        split=entry.split,
        reset_state_sha256=entry.reset_state_sha256,
        repeat_index=entry.repeat_index,
        context=manifest.context,
        executed_intervention=manifest.candidate(entry.candidate_id),
        post_action_frames=(_post_frame(str(entry.execution_index)),),
        outcome_contract=manifest.outcome_contract,
        observed_outcome=outcome,
        execution_status=ExecutionStatus.COMPLETED,
        execution_receipt_id=f"receipt-{entry.execution_index}",
        diagnostics={"control_steps": 300},
        private_evaluator_metadata={"final predicate": outcome},
    )


def test_manifest_round_trip_binds_public_context_candidate_set_and_identities() -> (
    None
):
    manifest = _manifest()
    assert set(manifest.model_input()) == {"context", "candidates"}
    assert "reset_state_sha256" not in str(manifest.model_input())
    assert "final predicate" not in str(manifest.model_input())
    encoded = manifest.to_dict()
    decoded = DecisionGroupManifest.from_mapping(encoded)
    assert decoded.fingerprint() == manifest.fingerprint()
    assert decoded.candidate_set_sha256 == manifest.candidate_set_sha256

    tampered = dict(encoded)
    tampered["manifest_sha256"] = _digest("9")
    with pytest.raises(ValueError, match="manifest digest"):
        DecisionGroupManifest.from_mapping(tampered)


def test_schedule_is_exact_candidate_by_paired_seed_matrix() -> None:
    manifest = _manifest()
    schedule = build_branch_schedule((manifest,), schedule_id="method-v1-schedule")
    assert len(schedule) == 4
    assert [row.model_seed for row in schedule] == [101, 202, 101, 202]
    assert [row.repeat_index for row in schedule] == [0, 1, 0, 1]
    assert BranchScheduleEntry.from_mapping(schedule[0].to_dict()) == schedule[0]

    with pytest.raises(ValueError, match="complete candidate x seed matrix"):
        validate_branch_schedule((manifest,), schedule[:-1])
    changed_seed = dataclasses.replace(schedule[1], model_seed=999)
    with pytest.raises(ValueError, match="model seed"):
        validate_branch_schedule(
            (manifest,), (schedule[0], changed_seed, *schedule[2:])
        )


def test_collection_attempt_never_turns_infrastructure_failure_into_label() -> None:
    manifest = _manifest()
    schedule = build_branch_schedule((manifest,), schedule_id="method-v1-schedule")
    branch = _branch(manifest, schedule[0], outcome=True)
    evaluated = CollectionAttempt(
        attempt_id="attempt-0",
        schedule_entry=schedule[0],
        status=CollectionAttemptStatus.OUTCOME_EVALUATED,
        artifact_tree_sha256=_digest("3"),
        branch=branch,
    )
    failed = CollectionAttempt(
        attempt_id="attempt-1",
        schedule_entry=schedule[1],
        status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
        artifact_tree_sha256=_digest("4"),
        infrastructure_failure=InfrastructureFailure(
            failure_type="transport_error",
            message="endpoint unavailable",
            diagnostics_sha256=_digest("5"),
        ),
    )
    assert evaluated.has_training_label is True
    assert failed.has_training_label is False
    assert failed.to_dict()["branch"] is None
    with pytest.raises(ValueError, match="must not carry an outcome"):
        dataclasses.replace(failed, branch=branch)

    validate_collection_attempts(
        (manifest,), schedule, (evaluated, failed), require_complete=False
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_collection_attempts(
            (manifest,), schedule, (evaluated, failed), require_complete=True
        )


def test_executed_only_target_leaves_every_unexecuted_outcome_unknown() -> None:
    manifest = _manifest()
    schedule = build_branch_schedule((manifest,), schedule_id="method-v1-schedule")
    branch = _branch(manifest, schedule[0], outcome=True)
    target = executed_only_target(manifest, branch)
    assert target.labels == (True, None)
    assert target.executed_mask == (True, False)
    assert target.executed_index == 0

    with pytest.raises(ValueError, match="unexecuted candidate outcomes"):
        dataclasses.replace(target, labels=(True, False))


def test_split_firewall_groups_reset_hidden_variant_and_scene_layout() -> None:
    train = _manifest()
    cross_hidden = _manifest(
        manifest_id="manifest-002",
        decision_group_id="decision-002",
        initial_state_group="state-002",
        split_group_id=train.split_group_id,
        split="test",
        scene_id="another-scene",
        layout_id="layout-b",
    )
    with pytest.raises(ValueError, match="split_group_id"):
        validate_manifest_splits((train, cross_hidden))

    cross_layout = _manifest(
        manifest_id="manifest-003",
        decision_group_id="decision-003",
        initial_state_group="state-003",
        split_group_id="hidden-pair-003",
        split="test",
        scene_id=train.scene_id,
        layout_id=train.layout_id,
    )
    with pytest.raises(ValueError, match="scene/layout"):
        validate_manifest_splits((train, cross_layout))


def test_dataset_returns_only_outcome_evaluated_branches() -> None:
    manifest = _manifest()
    schedule = build_branch_schedule((manifest,), schedule_id="method-v1-schedule")
    attempts: list[CollectionAttempt] = []
    for index, entry in enumerate(schedule):
        if index == 1:
            attempts.append(
                CollectionAttempt(
                    attempt_id=f"attempt-{index}",
                    schedule_entry=entry,
                    status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
                    artifact_tree_sha256=_digest("6"),
                    infrastructure_failure=InfrastructureFailure(
                        failure_type="environment_error",
                        message="reset failed",
                        diagnostics_sha256=_digest("7"),
                    ),
                )
            )
        else:
            attempts.append(
                CollectionAttempt(
                    attempt_id=f"attempt-{index}",
                    schedule_entry=entry,
                    status=CollectionAttemptStatus.OUTCOME_EVALUATED,
                    artifact_tree_sha256=_digest("8"),
                    branch=_branch(manifest, entry, outcome=(index % 2 == 0)),
                )
            )
    dataset = MethodV1OutcomeDataset(
        dataset_id="method-v1-dataset",
        manifests=(manifest,),
        schedule=schedule,
        attempts=tuple(attempts),
        complete=False,
    )
    assert len(dataset.observed_branches) == 3
    assert len(dataset.executed_targets()) == 3
    assert dataset.summary()["infrastructure_failures"] == 1
    encoded = dataset.to_dict()
    encoded_branches = [
        item["branch"] for item in encoded["attempts"] if item["branch"] is not None
    ]
    assert all(
        "private_evaluator_metadata" not in branch for branch in encoded_branches
    )
    decoded = MethodV1OutcomeDataset.from_mapping(encoded)
    assert decoded.fingerprint() == dataset.fingerprint()
    with pytest.raises(ValueError, match="infrastructure failures"):
        dataclasses.replace(dataset, complete=True)


def test_global_collection_plan_freezes_exact_portable_group_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_manifest = _manifest()
    second_manifest = _manifest(
        manifest_id="manifest-002",
        decision_group_id="decision-002",
        initial_state_group="state-002",
        split_group_id="hidden-pair-002",
        scene_id="drawer-scene-2",
        layout_id="layout-b",
    )
    second_manifest = dataclasses.replace(
        second_manifest,
        information_stratum=InformationStratum.INFORMATION_SUFFICIENT,
    )
    third_manifest = _manifest(
        manifest_id="manifest-003",
        decision_group_id="decision-003",
        initial_state_group="state-003",
        split_group_id="hidden-pair-003",
        scene_id="drawer-scene-3",
        layout_id="layout-c",
    )
    third_manifest = dataclasses.replace(
        third_manifest,
        information_stratum=InformationStratum.INFORMATION_ACTION_NO_HELP,
    )
    roots = (
        tmp_path / "freeze-a",
        tmp_path / "freeze-b",
        tmp_path / "freeze-c",
    )
    for root in roots:
        root.mkdir()
    freezes = {
        roots[0].resolve(): VerifiedDecisionFreeze(
            freeze_dir=roots[0].resolve(),
            manifest=first_manifest,
            schedule=build_branch_schedule((first_manifest,), schedule_id="schedule-a"),
            freeze_receipt_sha256=hashlib.sha256(b"receipt-a").hexdigest(),
        ),
        roots[1].resolve(): VerifiedDecisionFreeze(
            freeze_dir=roots[1].resolve(),
            manifest=second_manifest,
            schedule=build_branch_schedule(
                (second_manifest,), schedule_id="schedule-b"
            ),
            freeze_receipt_sha256=hashlib.sha256(b"receipt-b").hexdigest(),
        ),
        roots[2].resolve(): VerifiedDecisionFreeze(
            freeze_dir=roots[2].resolve(),
            manifest=third_manifest,
            schedule=build_branch_schedule((third_manifest,), schedule_id="schedule-c"),
            freeze_receipt_sha256=hashlib.sha256(b"receipt-c").hexdigest(),
        ),
    }
    monkeypatch.setattr(
        collector_module,
        "load_verified_decision_freeze",
        lambda path: freezes[Path(path).expanduser().resolve()],
    )
    configured = {
        "train": 3,
        "validation": 0,
        "calibration": 0,
        "test": 0,
    }
    configured_strata = {
        split: {
            InformationStratum.INFORMATION_NECESSARY.value: int(count > 0),
            InformationStratum.INFORMATION_SUFFICIENT.value: int(count > 0),
            InformationStratum.INFORMATION_ACTION_NO_HELP.value: int(count > 0),
        }
        for split, count in configured.items()
    }
    monkeypatch.setattr(
        collector_module,
        "_collection_source_config_contract",
        lambda _path, _freezes: (
            _digest("8"),
            configured,
            configured_strata,
        ),
    )
    key = tmp_path / "scorer.key"
    key.write_bytes(b"global-plan-test-secret-key-material")
    plan_path = tmp_path / "collection_plan.json"
    plan = freeze_collection_plan(
        freeze_dirs=roots,
        plan_id="method-v1-study",
        scorer_auth_key_path=key,
        config_path=tmp_path / "config.json",
        output=plan_path,
    )
    assert plan["expected_split_counts"]["train"]["decision_groups"] == 3
    assert plan["group_count"] == 3
    assert all("freeze_dir" not in row for row in plan["groups"])
    verified = load_verified_collection_plan(
        plan_path,
        freeze_dirs=tuple(reversed(roots)),
        config_path=tmp_path / "config.json",
    )
    assert verified.collection_plan_sha256 == plan["collection_plan_sha256"]

    with pytest.raises(ValueError, match="population does not match"):
        load_verified_collection_plan(
            plan_path,
            freeze_dirs=roots[:2],
            config_path=tmp_path / "config.json",
        )
    with pytest.raises(ValueError, match="must be unique"):
        load_verified_collection_plan(
            plan_path,
            freeze_dirs=(roots[0], roots[0]),
            config_path=tmp_path / "config.json",
        )

    replacement_root = tmp_path / "freeze-replacement"
    replacement_root.mkdir()
    replacement_manifest = _manifest(
        manifest_id="manifest-replacement",
        decision_group_id="decision-replacement",
        initial_state_group="state-replacement",
        split_group_id="hidden-replacement",
        scene_id="replacement-scene",
        layout_id="replacement-layout",
    )
    replacement_manifest = dataclasses.replace(
        replacement_manifest,
        information_stratum=InformationStratum.INFORMATION_ACTION_NO_HELP,
    )
    freezes[replacement_root.resolve()] = VerifiedDecisionFreeze(
        freeze_dir=replacement_root.resolve(),
        manifest=replacement_manifest,
        schedule=build_branch_schedule(
            (replacement_manifest,), schedule_id="schedule-replacement"
        ),
        freeze_receipt_sha256=hashlib.sha256(b"replacement").hexdigest(),
    )
    with pytest.raises(ValueError, match="differs from re-opened"):
        load_verified_collection_plan(
            plan_path,
            freeze_dirs=(roots[0], roots[1], replacement_root),
            config_path=tmp_path / "config.json",
        )

    changed_split = dataclasses.replace(third_manifest, split="validation")
    split_root = tmp_path / "freeze-split-change"
    split_root.mkdir()
    freezes[split_root.resolve()] = VerifiedDecisionFreeze(
        freeze_dir=split_root.resolve(),
        manifest=changed_split,
        schedule=build_branch_schedule((changed_split,), schedule_id="schedule-split"),
        freeze_receipt_sha256=hashlib.sha256(b"split-change").hexdigest(),
    )
    with pytest.raises(ValueError, match="population does not match"):
        load_verified_collection_plan(
            plan_path,
            freeze_dirs=(roots[0], roots[1], split_root),
            config_path=tmp_path / "config.json",
        )

    with pytest.raises(FileExistsError):
        freeze_collection_plan(
            freeze_dirs=roots,
            plan_id="method-v1-study",
            scorer_auth_key_path=key,
            config_path=tmp_path / "config.json",
            output=plan_path,
        )
