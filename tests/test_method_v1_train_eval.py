from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

import grounded_interaction.collect_outcomes as collector_module
from grounded_interaction.collect_outcomes import (
    VerifiedCollectionPlan,
    VerifiedDecisionFreeze,
    _tree_sha256,
)
from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_sha256,
)
from grounded_interaction.data import ObservedBranch
from grounded_interaction.evaluate_policy import (
    Baseline,
    BranchPrediction,
    CanonicalPredictionArtifact,
    PolicyEpisodeResult,
    _canonical_rows_for_split,
    build_canonical_prediction_artifact,
    evaluate_formal_branch_matrix,
    evaluate_formal_probabilities,
    evaluate_offline_branch_matrix,
    probability_metrics,
    summarize_closed_loop_episodes,
    validate_canonical_prediction_artifact,
)
from grounded_interaction.method_v1_data import (
    CollectionAttempt,
    CollectionAttemptStatus,
    DecisionGroupManifest,
    InformationStratum,
    InfrastructureFailure,
    MethodV1OutcomeDataset,
    build_branch_schedule,
    make_method_v1_outcome_contract,
)
from grounded_interaction.qwen_provider import (
    QwenFeatureCache,
    candidate_feature_cache_key,
)
from grounded_interaction.tokens import CandidateTokenField, FrozenTokenField
from grounded_interaction.train_outcomes import (
    EncodedOutcomeRecord,
    GroupUniformSampler,
    ReceiptBackedMethodV1Dataset,
    TrainingConfig,
    _config_from_mapping,
    build_method_v1_outcome_dataset,
    collate_outcome_records,
    encoded_records_from_method_v1_dataset,
    fit_positive_temperature,
    load_checkpoint_model,
    load_method_v1_outcome_dataset,
    load_qwen_token_fields,
    method_v1_training_records,
    select_manifest_candidate,
    train_one_seed,
)

torch = pytest.importorskip("torch")


def _digest(character: str) -> str:
    return character * 64


def _field(
    *,
    context_character: str,
    candidate_characters: tuple[str, ...] = ("b", "c"),
    token_count: int = 4,
    provider_id: str = "qwen-test@revision+processor-v1",
    candidate_ids: tuple[str, ...] | None = None,
    candidate_fingerprints: tuple[str, ...] | None = None,
    use_state: bool = False,
    state_offset: float = 0.0,
) -> CandidateTokenField:
    candidate_ids = candidate_ids or tuple(
        f"candidate-{index}" for index in range(len(candidate_characters))
    )
    candidate_fingerprints = candidate_fingerprints or tuple(
        _digest(character) for character in candidate_characters
    )
    context_tokens = torch.arange(token_count * 8, dtype=torch.float32).reshape(
        1, token_count, 8
    )
    valid = torch.ones(1, token_count, dtype=torch.bool)
    current = torch.zeros(1, token_count, dtype=torch.bool)
    current[0, :2] = True
    boxes = torch.zeros(1, token_count, 4)
    boxes[0, 0] = torch.tensor([0.0, 0.0, 0.5, 1.0])
    boxes[0, 1] = torch.tensor([0.5, 0.0, 1.0, 1.0])
    context = FrozenTokenField(
        tokens=context_tokens,
        valid_mask=valid,
        current_patch_mask=current,
        camera_ids=(
            tuple("wrist" if index < 2 else None for index in range(token_count)),
        ),
        frame_ids=(
            tuple(
                "frame-current" if index < 2 else None for index in range(token_count)
            ),
        ),
        patch_xyxy=boxes,
        context_fingerprints=(_digest(context_character),),
        provider_id=provider_id,
    )
    candidate_count = len(candidate_ids)
    support = torch.zeros(1, candidate_count, token_count, dtype=torch.bool)
    for index in range(candidate_count):
        support[0, index, index % 2] = True
    kwargs: dict[str, object] = {}
    if use_state:
        kwargs = {
            "public_state_values": (
                torch.arange(9, dtype=torch.float32).unsqueeze(0) + state_offset
            ),
            "public_state_valid_mask": torch.tensor([True]),
        }
    return CandidateTokenField(
        tokens=torch.arange(candidate_count * 8, dtype=torch.float32).reshape(
            1, candidate_count, 8
        ),
        valid_mask=torch.ones(1, candidate_count, dtype=torch.bool),
        candidate_ids=(candidate_ids,),
        candidate_fingerprints=(candidate_fingerprints,),
        primitives=(
            tuple(
                Primitive.DIRECT if index % 2 == 0 else Primitive.OPEN
                for index in range(candidate_count)
            ),
        ),
        grounding_support=support,
        public_context=context,
        **kwargs,
    )


def _record(
    *,
    record_id: str,
    decision_group_id: str,
    initial_state_group: str | None = None,
    split: str,
    field: CandidateTokenField,
    executed_index: int,
    outcome: bool,
) -> EncodedOutcomeRecord:
    return EncodedOutcomeRecord(
        record_id=record_id,
        experiment_id="method-v1-test",
        initial_state_group=initial_state_group or decision_group_id,
        decision_group_id=decision_group_id,
        split_group_id=f"split-{decision_group_id}",
        split=split,
        proposal_provider_id="qwen-proposer-test",
        configuration_sha256=_digest("d"),
        branch_fingerprint=hashlib.sha256(record_id.encode()).hexdigest(),
        outcome_contract_sha256=_digest("e"),
        token_cache_sha256=_digest("f"),
        field=field,
        executed_candidate_fingerprint=field.candidate_fingerprints[0][executed_index],
        observed_outcome=outcome,
    )


def test_group_uniform_sampler_does_not_overweight_candidate_rich_groups() -> None:
    small = _field(context_character="a")
    large = _field(
        context_character="1",
        candidate_characters=("2", "3", "4", "5", "6"),
    )
    rows = [
        _record(
            record_id="small-0",
            decision_group_id="small",
            split="train",
            field=small,
            executed_index=0,
            outcome=True,
        )
    ]
    rows.extend(
        _record(
            record_id=f"large-{index}",
            decision_group_id="large",
            split="train",
            field=large,
            executed_index=index,
            outcome=False,
        )
        for index in range(5)
    )
    sampler = GroupUniformSampler(rows, seed=11, samples_per_epoch=20)
    sampled_groups = [rows[index].decision_group_id for index in sampler]
    assert sampled_groups.count("small") == 10
    assert sampled_groups.count("large") == 10
    sampler.set_epoch(1)
    assert list(sampler) != list(
        GroupUniformSampler(rows, seed=11, samples_per_epoch=20)
    )


def test_group_uniform_sampler_uses_physical_reset_not_decision_count() -> None:
    field = _field(context_character="a")
    rows = [
        _record(
            record_id=f"shared-{index}",
            decision_group_id=f"decision-{index}",
            initial_state_group="shared-reset",
            split="train",
            field=field,
            executed_index=index % 2,
            outcome=True,
        )
        for index in range(3)
    ]
    rows.append(
        _record(
            record_id="other",
            decision_group_id="other-decision",
            initial_state_group="other-reset",
            split="train",
            field=field,
            executed_index=0,
            outcome=False,
        )
    )
    sampler = GroupUniformSampler(rows, seed=4, samples_per_epoch=20)
    selected_resets = [rows[index].initial_state_group for index in sampler]
    assert selected_resets.count("shared-reset") == 10
    assert selected_resets.count("other-reset") == 10


def test_collation_pads_fields_and_keeps_only_executed_outcomes() -> None:
    first = _field(context_character="a", token_count=3)
    second = _field(
        context_character="1",
        candidate_characters=("2", "3", "4"),
        token_count=5,
    )
    batch = collate_outcome_records(
        (
            _record(
                record_id="first",
                decision_group_id="group-a",
                split="train",
                field=first,
                executed_index=0,
                outcome=True,
            ),
            _record(
                record_id="second",
                decision_group_id="group-b",
                split="train",
                field=second,
                executed_index=2,
                outcome=False,
            ),
        )
    )
    assert batch.field.tokens.shape == (2, 3, 8)
    assert batch.field.public_context.tokens.shape == (2, 5, 8)
    assert batch.executed_mask.tolist() == [[True, False, False], [False, False, True]]
    assert batch.observed_outcomes[0, 0].item() == 1
    assert batch.observed_outcomes[1, 2].item() == 0
    assert torch.isnan(batch.observed_outcomes[~batch.executed_mask]).all()
    assert batch.field.candidate_ids[0][2] is None


def _canonical_method_v1_dataset() -> tuple[
    MethodV1OutcomeDataset, DecisionGroupManifest, CandidateTokenField
]:
    frame = PublicFrame(
        frame_id="wrist-0",
        camera="wrist",
        frame_index=0,
        image_sha256=_digest("a"),
        width=256,
        height=256,
    )
    context = PolicyContext(
        prompt="Place the butter in the basket.",
        frames=(frame,),
        proprioception=tuple(float(index) for index in range(8)),
    )
    groundings = (
        GroundingReference(
            camera="wrist",
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(0.05, 0.1, 0.45, 0.9),
            point_xy=(0.25, 0.5),
        ),
        GroundingReference(
            camera="wrist",
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(0.55, 0.1, 0.95, 0.9),
            point_xy=(0.75, 0.5),
        ),
    )
    candidates = (
        GroundedIntervention(
            candidate_id="direct-butter",
            primitive=Primitive.DIRECT,
            referent="visible butter candidate",
            parameters=(("instruction", "Place this item in the basket."),),
            grounding=groundings[0],
        ),
        GroundedIntervention(
            candidate_id="open-drawer",
            primitive=Primitive.OPEN,
            referent="middle drawer",
            parameters=(("instruction", "Open this drawer."),),
            grounding=groundings[1],
        ),
    )
    contract = make_method_v1_outcome_contract(
        continuation_policy_id="qwen-direct-continuation-v1",
        executor_id="molmoact2-libero@revision",
        serializer_id="grounded-precise-text-v1",
    )
    cache_key = candidate_feature_cache_key(
        provider_id="qwen-test@revision+processor-v1",
        context=context,
        candidates=candidates,
    )
    manifest = DecisionGroupManifest(
        manifest_id="manifest-0",
        experiment_id="method-v1-test",
        initial_state_group="reset-0",
        decision_group_id="decision-0",
        split_group_id="layout-0",
        information_stratum=InformationStratum.INFORMATION_NECESSARY,
        split="train",
        scene_id="scene",
        layout_id="layout",
        reset_state_sha256=_digest("b"),
        context=context,
        candidates=candidates,
        outcome_contract=contract,
        proposal_provider_id="qwen-proposer@revision",
        proposal_request_sha256=_digest("c"),
        token_provider_id="qwen-test@revision+processor-v1",
        token_cache_sha256=cache_key,
        configuration_sha256=_digest("e"),
        model_seeds=(7, 11),
    )
    schedule = build_branch_schedule((manifest,), schedule_id="schedule-0")
    scheduled = schedule[0]
    branch = ObservedBranch(
        branch_id="branch-0",
        initial_state_group=manifest.initial_state_group,
        decision_group_id=manifest.decision_group_id,
        split=manifest.split,
        reset_state_sha256=manifest.reset_state_sha256,
        repeat_index=scheduled.repeat_index,
        context=context,
        executed_intervention=candidates[0],
        post_action_frames=(
            PublicFrame(
                frame_id="wrist-300",
                camera="wrist",
                frame_index=300,
                image_sha256=_digest("f"),
                width=256,
                height=256,
            ),
        ),
        outcome_contract=contract,
        observed_outcome=True,
        execution_status=ExecutionStatus.COMPLETED,
        execution_receipt_id="receipt-0",
    )
    attempt = CollectionAttempt(
        attempt_id="attempt-0",
        schedule_entry=scheduled,
        status=CollectionAttemptStatus.OUTCOME_EVALUATED,
        artifact_tree_sha256=_digest("1"),
        branch=branch,
    )
    dataset = MethodV1OutcomeDataset(
        dataset_id="dataset-0",
        manifests=(manifest,),
        schedule=schedule,
        attempts=(attempt,),
        complete=False,
    )
    field = _field(
        context_character="0",
        provider_id=manifest.token_provider_id,
        use_state=True,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        candidate_fingerprints=tuple(
            candidate.fingerprint() for candidate in candidates
        ),
    )
    field = dataclasses.replace(
        field,
        public_context=dataclasses.replace(
            field.public_context,
            context_fingerprints=(context.fingerprint(),),
        ),
    )
    return dataset, manifest, field


def _admit_synthetic_dataset(
    dataset: MethodV1OutcomeDataset,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
) -> ReceiptBackedMethodV1Dataset:
    """Exercise the public admission factory with mocked collector validators.

    Formal scorer tests intentionally use tiny synthetic tensors.  Collector
    receipt/trace integrity has its own integration tests, so this fixture mocks
    only those expensive source validators while retaining the canonical
    schedule-consumption gate.
    """

    freeze_dirs = tuple(
        (tmp_path / f"{prefix}-freeze-{index}").resolve()
        for index, _ in enumerate(dataset.manifests)
    )
    output_root = (tmp_path / f"{prefix}-outcomes").resolve()
    attempt_paths = tuple(
        (output_root / attempt.schedule_entry.entry_id / "attempt.json").resolve()
        for attempt in dataset.attempts
    )
    freezes = {
        root: VerifiedDecisionFreeze(
            freeze_dir=root,
            manifest=manifest,
            schedule=tuple(
                entry
                for entry in dataset.schedule
                if entry.manifest_id == manifest.manifest_id
            ),
            freeze_receipt_sha256=hashlib.sha256(
                f"{manifest.manifest_id}-receipt".encode()
            ).hexdigest(),
        )
        for index, (root, manifest) in enumerate(
            zip(freeze_dirs, dataset.manifests, strict=True)
        )
    }
    attempts = dict(zip(attempt_paths, dataset.attempts, strict=True))
    monkeypatch.setattr(
        collector_module,
        "load_verified_collection_plan",
        lambda path, **_: VerifiedCollectionPlan(
            plan_path=Path(path).expanduser().resolve(),
            collection_plan_sha256=_digest("7"),
            scorer_verifier_auth_key_id=_digest("8"),
            freezes=tuple(freezes.values()),
            document={"groups": []},
        ),
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_collection_attempt",
        lambda path: attempts[Path(path).expanduser().resolve()],
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_execution_claim",
        lambda **_: {
            "output_root": str(output_root),
            "selection_sha256": None,
            "collection_plan_sha256": _digest("7"),
            "scorer_verifier_auth_key_id": _digest("8"),
        },
    )
    return build_method_v1_outcome_dataset(
        collection_plan_path=tmp_path / f"{prefix}-collection-plan.json",
        collection_config_path=tmp_path / f"{prefix}-config.json",
        freeze_dirs=freeze_dirs,
        attempt_paths=attempt_paths,
        dataset_id=dataset.dataset_id,
        require_complete=dataset.complete,
    )


def test_split_scoped_admission_never_opens_held_out_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, train_manifest, _ = _canonical_method_v1_dataset()
    test_manifest = dataclasses.replace(
        train_manifest,
        manifest_id="manifest-test",
        initial_state_group="reset-test",
        decision_group_id="decision-test",
        split_group_id="layout-test",
        split="test",
        scene_id="scene-test",
        layout_id="layout-test",
        reset_state_sha256=_digest("c"),
        proposal_request_sha256=_digest("d"),
    )
    train_freeze_dir = (tmp_path / "freeze-train").resolve()
    test_freeze_dir = (tmp_path / "freeze-test").resolve()
    test_schedule = build_branch_schedule(
        (test_manifest,), schedule_id="held-out-test-schedule"
    )
    plan = VerifiedCollectionPlan(
        plan_path=(tmp_path / "collection-plan.json").resolve(),
        collection_plan_sha256=_digest("7"),
        scorer_verifier_auth_key_id=_digest("8"),
        freezes=(
            VerifiedDecisionFreeze(
                freeze_dir=train_freeze_dir,
                manifest=train_manifest,
                schedule=dataset.schedule,
                freeze_receipt_sha256=_digest("5"),
            ),
            VerifiedDecisionFreeze(
                freeze_dir=test_freeze_dir,
                manifest=test_manifest,
                schedule=(test_schedule[0],),
                freeze_receipt_sha256=_digest("6"),
            ),
        ),
        document={"groups": []},
    )
    output_root = (tmp_path / "outcomes").resolve()
    train_attempt_paths = tuple(
        (output_root / entry.entry_id / "attempt.json").resolve()
        for entry in dataset.schedule
    )
    test_attempt_path = (
        output_root / test_schedule[0].entry_id / "attempt.json"
    ).resolve()
    opened: list[Path] = []
    train_attempts = {train_attempt_paths[0]: dataset.attempts[0]}
    for index, (path, entry) in enumerate(
        zip(train_attempt_paths[1:], dataset.schedule[1:], strict=True), start=1
    ):
        train_attempts[path] = CollectionAttempt(
            attempt_id=f"train-infrastructure-{index}",
            schedule_entry=entry,
            status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
            artifact_tree_sha256=hashlib.sha256(
                f"train-infrastructure-tree-{index}".encode()
            ).hexdigest(),
            infrastructure_failure=InfrastructureFailure(
                failure_type="SyntheticInfrastructureError",
                message="split-scoped admission fixture",
                diagnostics_sha256=hashlib.sha256(
                    f"train-infrastructure-diagnostics-{index}".encode()
                ).hexdigest(),
            ),
        )

    monkeypatch.setattr(
        collector_module,
        "load_verified_collection_plan",
        lambda *_args, **_kwargs: plan,
    )

    def load_attempt(path: str | Path) -> CollectionAttempt:
        source = Path(path).expanduser().resolve()
        opened.append(source)
        if source == test_attempt_path:
            raise AssertionError("held-out attempt was opened")
        return train_attempts[source]

    monkeypatch.setattr(
        collector_module, "load_verified_collection_attempt", load_attempt
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_execution_claim",
        lambda **_: {
            "output_root": str(output_root),
            "selection_sha256": None,
            "collection_plan_sha256": _digest("7"),
            "scorer_verifier_auth_key_id": _digest("8"),
        },
    )

    admitted = build_method_v1_outcome_dataset(
        collection_plan_path=plan.plan_path,
        collection_config_path=tmp_path / "config.json",
        freeze_dirs=(train_freeze_dir, test_freeze_dir),
        attempt_paths=train_attempt_paths,
        dataset_id="train-only",
        admitted_splits=("train",),
    )
    assert {manifest.split for manifest in admitted.manifests} == {"train"}
    assert opened == list(train_attempt_paths)

    with pytest.raises(ValueError, match="outside admitted_splits"):
        build_method_v1_outcome_dataset(
            collection_plan_path=plan.plan_path,
            collection_config_path=tmp_path / "config.json",
            freeze_dirs=(train_freeze_dir, test_freeze_dir),
            attempt_paths=(*train_attempt_paths, test_attempt_path),
            dataset_id="leaky-training-input",
            admitted_splits=("train",),
        )
    assert opened == list(train_attempt_paths)


def _formal_method_v1_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ReceiptBackedMethodV1Dataset, Path, tuple[Path, ...]]:
    """Build a complete train/validation/test matrix and real tiny checkpoints.

    This is intentionally more expensive than a mocked scorer test: formal
    evaluation must be exercised through the same checkpoint loader and
    content-addressed Qwen cache used by the command-line entry point.
    """

    provider_id = "qwen-test@revision+processor-v1"
    proposal_provider_id = "qwen-proposer@revision"
    configuration_sha256 = _digest("9")
    contract = make_method_v1_outcome_contract(
        continuation_policy_id="qwen-direct-continuation-v1",
        executor_id="molmoact2-libero@revision",
        serializer_id="grounded-precise-text-v1",
    )
    manifests: list[DecisionGroupManifest] = []
    fields: dict[str, CandidateTokenField] = {}
    split_characters = {
        "train": ("a", "1"),
        "validation": ("b", "2"),
        "test": ("c", "3"),
    }
    for split_index, (split, (image_character, context_character)) in enumerate(
        split_characters.items()
    ):
        frame = PublicFrame(
            frame_id=f"{split}-wrist-0",
            camera="wrist",
            frame_index=0,
            image_sha256=_digest(image_character),
            width=256,
            height=256,
        )
        context = PolicyContext(
            prompt="Place the butter in the basket.",
            frames=(frame,),
            proprioception=tuple(float(index + split_index) for index in range(8)),
        )
        groundings = (
            GroundingReference(
                camera="wrist",
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                image_sha256=frame.image_sha256,
                box_xyxy=(0.05, 0.1, 0.45, 0.9),
                point_xy=(0.25, 0.5),
            ),
            GroundingReference(
                camera="wrist",
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                image_sha256=frame.image_sha256,
                box_xyxy=(0.55, 0.1, 0.95, 0.9),
                point_xy=(0.75, 0.5),
            ),
        )
        candidates = (
            GroundedIntervention(
                candidate_id=f"{split}-direct-butter",
                primitive=Primitive.DIRECT,
                referent="visible butter candidate",
                parameters=(("instruction", "Place this item in the basket."),),
                grounding=groundings[0],
            ),
            GroundedIntervention(
                candidate_id=f"{split}-open-drawer",
                primitive=Primitive.OPEN,
                referent="middle drawer",
                parameters=(("instruction", "Open this drawer."),),
                grounding=groundings[1],
            ),
        )
        cache_key = candidate_feature_cache_key(
            provider_id=provider_id,
            context=context,
            candidates=candidates,
        )
        manifest = DecisionGroupManifest(
            manifest_id=f"{split}-manifest",
            experiment_id="method-v1-formal-test",
            initial_state_group=f"{split}-reset",
            decision_group_id=f"{split}-decision",
            split_group_id=f"{split}-family",
            information_stratum=InformationStratum.INFORMATION_NECESSARY,
            split=split,
            scene_id=f"{split}-scene",
            layout_id=f"{split}-layout",
            reset_state_sha256=hashlib.sha256(
                f"{split}-reset-state".encode()
            ).hexdigest(),
            context=context,
            candidates=candidates,
            outcome_contract=contract,
            proposal_provider_id=proposal_provider_id,
            proposal_request_sha256=hashlib.sha256(
                f"{split}-proposal-request".encode()
            ).hexdigest(),
            token_provider_id=provider_id,
            token_cache_sha256=cache_key,
            configuration_sha256=configuration_sha256,
            model_seeds=(101, 202),
        )
        field = _field(
            context_character=context_character,
            provider_id=provider_id,
            use_state=True,
            state_offset=float(split_index),
            candidate_ids=tuple(item.candidate_id for item in candidates),
            candidate_fingerprints=tuple(item.fingerprint() for item in candidates),
        )
        field = dataclasses.replace(
            field,
            public_context=dataclasses.replace(
                field.public_context,
                context_fingerprints=(context.fingerprint(),),
            ),
        )
        manifests.append(manifest)
        fields[cache_key] = field

    schedule = build_branch_schedule(tuple(manifests), schedule_id="formal-schedule")
    manifests_by_id = {item.manifest_id: item for item in manifests}
    attempts: list[CollectionAttempt] = []
    for entry in schedule:
        manifest = manifests_by_id[entry.manifest_id]
        candidate = next(
            item
            for item in manifest.candidates
            if item.fingerprint() == entry.candidate_fingerprint
        )
        branch = ObservedBranch(
            branch_id=f"branch-{entry.execution_index}",
            initial_state_group=manifest.initial_state_group,
            decision_group_id=manifest.decision_group_id,
            split=manifest.split,
            reset_state_sha256=manifest.reset_state_sha256,
            repeat_index=entry.repeat_index,
            context=manifest.context,
            executed_intervention=candidate,
            post_action_frames=(
                PublicFrame(
                    frame_id=f"post-{entry.execution_index}",
                    camera="wrist",
                    frame_index=300,
                    image_sha256=hashlib.sha256(
                        f"post-{entry.execution_index}".encode()
                    ).hexdigest(),
                    width=256,
                    height=256,
                ),
            ),
            outcome_contract=contract,
            observed_outcome=entry.repeat_index == 0,
            execution_status=ExecutionStatus.COMPLETED,
            execution_receipt_id=f"receipt-{entry.execution_index}",
        )
        attempts.append(
            CollectionAttempt(
                attempt_id=f"attempt-{entry.execution_index}",
                schedule_entry=entry,
                status=CollectionAttemptStatus.OUTCOME_EVALUATED,
                artifact_tree_sha256=hashlib.sha256(
                    f"tree-{entry.execution_index}".encode()
                ).hexdigest(),
                branch=branch,
            )
        )
    dataset = MethodV1OutcomeDataset(
        dataset_id="formal-dataset",
        manifests=tuple(manifests),
        schedule=schedule,
        attempts=tuple(attempts),
        complete=True,
    )

    cache_root = tmp_path / "feature-cache"
    cache = QwenFeatureCache(cache_root)
    for key, field in fields.items():
        cache.put(key, field)
    admitted_dataset = _admit_synthetic_dataset(
        dataset,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        prefix="formal",
    )
    records = method_v1_training_records(admitted_dataset, cache_roots=(cache_root,))
    config = TrainingConfig(
        learning_rate=1e-3,
        effective_batch_size=2,
        micro_batch_size=2,
        max_epochs=1,
        early_stopping_patience=1,
        hidden_dim=8,
        num_heads=2,
        feedforward_dim=16,
        seeds=(5, 6),
    )
    checkpoints = tuple(
        Path(
            train_one_seed(
                records,
                output_dir=tmp_path / f"checkpoint-{seed}",
                config=config,
                seed=seed,
                training_admission_evidence_sha256=(
                    admitted_dataset.evidence_for_splits(("train", "validation"))
                ),
                collection_plan_sha256=admitted_dataset.collection_plan_sha256,
                scorer_verifier_auth_key_id=(
                    admitted_dataset.scorer_verifier_auth_key_id
                ),
            ).checkpoint_path
        )
        for seed in config.seeds
    )
    return admitted_dataset, cache_root, checkpoints


def _formal_artifact_with_probability(
    artifact: CanonicalPredictionArtifact,
    *,
    record_id: str,
    value: float,
) -> CanonicalPredictionArtifact:
    """Return a self-consistent in-memory artifact with one declared score changed."""

    records = tuple(
        dataclasses.replace(
            item,
            checkpoint_probabilities=(value, *item.checkpoint_probabilities[1:]),
        )
        if item.record_id == record_id
        else item
        for item in artifact.records
    )
    return dataclasses.replace(artifact, records=records)


def test_formal_evaluation_replays_real_checkpoints_on_complete_held_out_matrix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, cache_root, checkpoints = _formal_method_v1_fixture(tmp_path, monkeypatch)
    artifact = build_canonical_prediction_artifact(
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )
    assert artifact.dataset_admission_evidence_sha256 == dataset.evidence_sha256
    assert {
        item.training_admission_evidence_sha256 for item in artifact.checkpoints
    } == {dataset.evidence_for_splits(("train", "validation"))}
    assert len(artifact.checkpoints) == 2
    assert len(artifact.records) == len(dataset.schedule)
    assert {item.split for item in artifact.records} == {
        "train",
        "validation",
        "test",
    }
    assert all(
        len(item.checkpoint_probabilities) == len(checkpoints)
        for item in artifact.records
    )

    replayed = validate_canonical_prediction_artifact(
        artifact,
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )
    assert replayed.to_dict() == artifact.to_dict()
    probabilities = evaluate_formal_probabilities(
        artifact,
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
        prediction_source="deep_ensemble",
        num_bins=2,
    )
    assert probabilities.evidence_scope == (
        "FORMAL_CHECKPOINT_REPLAYED_EXECUTED_BRANCH_PROBABILITIES"
    )
    assert probabilities.split == "test"
    assert probabilities.checkpoint_indices == (0, 1)
    assert probabilities.metrics.count == 4
    assert probabilities.primitive_outcome_coverage["primitive_by_outcome"] == {
        "DIRECT": {"failure": 1, "success": 1},
        "OPEN": {"failure": 1, "success": 1},
    }
    matrix = evaluate_formal_branch_matrix(
        artifact,
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
        baseline=Baseline.ENSEMBLE,
        bootstrap_samples=20,
    )
    assert matrix.evidence_scope == "FORMAL_OFFLINE_RESET_CONTROLLED_BRANCH_MATRIX"
    assert matrix.decision_group_count == 1
    assert matrix.primitive_outcome_coverage == (
        probabilities.primitive_outcome_coverage
    )


def test_formal_report_uses_replayed_scores_not_tolerated_artifact_numbers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, cache_root, checkpoints = _formal_method_v1_fixture(tmp_path, monkeypatch)
    artifact = build_canonical_prediction_artifact(
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )
    test_record = next(item for item in artifact.records if item.split == "test")
    original = test_record.checkpoint_probabilities[0]
    tiny_delta = -1e-8 if original > 0.5 else 1e-8
    tolerated = _formal_artifact_with_probability(
        artifact,
        record_id=test_record.record_id,
        value=original + tiny_delta,
    )
    clean_report = evaluate_formal_probabilities(
        artifact,
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
        num_bins=2,
    )
    tolerated_report = evaluate_formal_probabilities(
        tolerated,
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
        num_bins=2,
    )
    assert tolerated.artifact_sha256 != artifact.artifact_sha256
    assert tolerated_report.metrics == clean_report.metrics
    assert tolerated_report.prediction_artifact_sha256 == artifact.artifact_sha256
    assert tolerated_report.prediction_artifact_sha256 != tolerated.artifact_sha256

    obvious_tamper = _formal_artifact_with_probability(
        artifact,
        record_id=test_record.record_id,
        value=0.0 if original > 0.5 else 1.0,
    )
    with pytest.raises(ValueError, match="probabilities differ from checkpoint"):
        evaluate_formal_probabilities(
            obvious_tamper,
            dataset,
            cache_roots=(cache_root,),
            checkpoint_paths=checkpoints,
        )


def test_formal_evaluation_rejects_overlap_and_incomplete_candidate_seed_matrix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, cache_root, checkpoints = _formal_method_v1_fixture(tmp_path, monkeypatch)
    artifact = build_canonical_prediction_artifact(
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )
    overlap_groups = (
        *artifact.checkpoints[0].training_split_group_ids,
        "test-family",
    )
    overlap_artifact = dataclasses.replace(
        artifact,
        checkpoints=tuple(
            dataclasses.replace(item, training_split_group_ids=overlap_groups)
            for item in artifact.checkpoints
        ),
    )
    with pytest.raises(ValueError, match="overlap checkpoint training groups"):
        _canonical_rows_for_split(
            overlap_artifact,
            dataset,
            split="test",
            single_scorer_index=0,
        )

    missing_test_attempt = next(
        item for item in dataset.attempts if item.schedule_entry.split == "test"
    )
    missing_dataset = MethodV1OutcomeDataset(
        dataset_id="missing-formal-dataset",
        manifests=dataset.manifests,
        schedule=dataset.schedule,
        attempts=tuple(
            item
            for item in dataset.attempts
            if item.attempt_id != missing_test_attempt.attempt_id
        ),
        complete=False,
    )
    with pytest.raises(ValueError, match="every frozen schedule entry"):
        _admit_synthetic_dataset(
            missing_dataset,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            prefix="missing-formal",
        )

    infrastructure_attempt = dataclasses.replace(
        missing_test_attempt,
        status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
        branch=None,
        infrastructure_failure=InfrastructureFailure(
            failure_type="TransportError",
            message="registered test branch did not produce an outcome",
            diagnostics_sha256=_digest("8"),
        ),
    )
    infrastructure_dataset = MethodV1OutcomeDataset(
        dataset_id=dataset.dataset_id,
        manifests=dataset.manifests,
        schedule=dataset.schedule,
        attempts=tuple(
            infrastructure_attempt
            if item.attempt_id == missing_test_attempt.attempt_id
            else item
            for item in dataset.attempts
        ),
        complete=False,
    )
    admitted_infrastructure = _admit_synthetic_dataset(
        infrastructure_dataset,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        prefix="infrastructure-formal",
    )
    incomplete_artifact = build_canonical_prediction_artifact(
        admitted_infrastructure,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )
    with pytest.raises(ValueError, match="infrastructure failures"):
        evaluate_formal_branch_matrix(
            incomplete_artifact,
            admitted_infrastructure,
            cache_roots=(cache_root,),
            checkpoint_paths=checkpoints,
            baseline=Baseline.SCORER,
            bootstrap_samples=20,
        )


def test_formal_evaluation_rejects_artifact_cache_and_checkpoint_tamper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, cache_root, checkpoints = _formal_method_v1_fixture(tmp_path, monkeypatch)
    artifact = build_canonical_prediction_artifact(
        dataset,
        cache_roots=(cache_root,),
        checkpoint_paths=checkpoints,
    )

    identity_tamper = dataclasses.replace(
        artifact,
        checkpoints=(
            dataclasses.replace(
                artifact.checkpoints[0], configuration_sha256=_digest("0")
            ),
            artifact.checkpoints[1],
        ),
    )
    with pytest.raises(ValueError, match="checkpoint evidence mismatch"):
        validate_canonical_prediction_artifact(
            identity_tamper,
            dataset,
            cache_roots=(cache_root,),
            checkpoint_paths=checkpoints,
        )

    test_manifest = next(item for item in dataset.manifests if item.split == "test")
    cache_sidecar = (
        cache_root
        / "sha256"
        / test_manifest.token_cache_sha256[:2]
        / f"{test_manifest.token_cache_sha256}.json"
    )
    original_sidecar = cache_sidecar.read_bytes()
    sidecar = json.loads(original_sidecar)
    tensor_name = next(iter(sidecar["tensor_sha256"]))
    sidecar["tensor_sha256"][tensor_name] = _digest("0")
    cache_sidecar.write_text(json.dumps(sidecar))
    try:
        with pytest.raises(ValueError, match="digest mismatch"):
            validate_canonical_prediction_artifact(
                artifact,
                dataset,
                cache_roots=(cache_root,),
                checkpoint_paths=checkpoints,
            )
    finally:
        cache_sidecar.write_bytes(original_sidecar)

    checkpoint_path = checkpoints[0]
    original_checkpoint = checkpoint_path.read_bytes()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_identity = dict(payload["identity"])
    checkpoint_identity["proposal_provider_id"] = "tampered-proposer"
    payload["identity"] = checkpoint_identity
    payload["identity_sha256"] = canonical_sha256(checkpoint_identity)
    torch.save(payload, checkpoint_path)
    try:
        with pytest.raises(ValueError, match="proposal_provider_id differs"):
            validate_canonical_prediction_artifact(
                artifact,
                dataset,
                cache_roots=(cache_root,),
                checkpoint_paths=checkpoints,
            )
    finally:
        checkpoint_path.write_bytes(original_checkpoint)


def test_real_qwen_feature_cache_round_trip(tmp_path) -> None:
    _, manifest, field = _canonical_method_v1_dataset()
    cache = QwenFeatureCache(tmp_path)
    cache.put(manifest.token_cache_sha256, field)
    loaded = cache.load(
        manifest.token_cache_sha256,
        context=manifest.context,
        candidates=manifest.candidates,
        provider_id=manifest.token_provider_id,
    )
    assert loaded is not None
    assert loaded.candidate_fingerprints == field.candidate_fingerprints
    assert torch.equal(loaded.tokens, field.tokens)
    assert torch.equal(loaded.public_state_values, field.public_state_values)
    loaded_by_digest = load_qwen_token_fields((manifest,), cache_root=tmp_path)
    assert loaded_by_digest[manifest.token_cache_sha256].candidate_ids == (
        field.candidate_ids
    )

    manifest_path = (
        tmp_path
        / "sha256"
        / manifest.token_cache_sha256[:2]
        / f"{manifest.token_cache_sha256}.json"
    )
    sidecar = json.loads(manifest_path.read_text())
    sidecar["tensor_sha256"]["context_tokens"] = _digest("0")
    manifest_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="tensor digest mismatch"):
        cache.load(
            manifest.token_cache_sha256,
            context=manifest.context,
            candidates=manifest.candidates,
            provider_id=manifest.token_provider_id,
        )


def test_canonical_dataset_join_excludes_unlabelled_attempts() -> None:
    dataset, manifest, field = _canonical_method_v1_dataset()
    records = encoded_records_from_method_v1_dataset(
        dataset,
        token_fields_by_sha256={manifest.token_cache_sha256: field},
    )
    assert len(records) == 1
    assert records[0].observed_outcome is True
    assert records[0].split == "train"
    assert records[0].executed_candidate_fingerprint == (
        manifest.candidates[0].fingerprint()
    )


def test_collector_artifacts_join_directly_to_qwen_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        collector_module, "validate_method_v1_execution_trace", lambda **_: None
    )
    dataset, manifest, field = _canonical_method_v1_dataset()
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    (freeze_dir / "decision_manifest.json").write_text(json.dumps(manifest.to_dict()))
    schedule_body = {
        "schema_version": "method-v1-schedule-file-v1",
        "schedule_id": dataset.schedule[0].schedule_id,
        "manifest_sha256": manifest.fingerprint(),
        "entries": [entry.to_dict() for entry in dataset.schedule],
    }
    (freeze_dir / "branch_schedule.json").write_text(
        json.dumps(
            {
                **schedule_body,
                "schedule_sha256": canonical_sha256(schedule_body),
            }
        )
    )
    attempt_dir = tmp_path / "outcomes" / dataset.schedule[0].entry_id
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "public_execution_trace.json").write_text(json.dumps({"steps": 300}))
    (attempt_dir / "started.json").write_text(
        json.dumps(
            {
                "status": "STARTED",
                "entry": dataset.schedule[0].to_dict(),
                "manifest_sha256": manifest.fingerprint(),
                "selection_sha256": None,
                "collection_plan_sha256": _digest("7"),
                "scorer_verifier_auth_key_id": _digest("8"),
            }
        )
    )
    branch = dataset.attempts[0].branch
    assert branch is not None
    (attempt_dir / "observed_branch.public.json").write_text(
        json.dumps(branch.to_dict(include_private=False))
    )
    private_dir = attempt_dir / "private"
    private_dir.mkdir()
    (private_dir / "evaluator_sidecar.json").write_text(
        json.dumps(
            {
                "evaluator_predicate": ["In", "butter_1", "basket_1"],
                "final_task_success": branch.observed_outcome,
                "policy_input_changed_by_evaluator": False,
            }
        )
    )
    sealed_attempt = dataclasses.replace(
        dataset.attempts[0],
        artifact_tree_sha256=_tree_sha256(attempt_dir),
    )
    attempt_path = attempt_dir / "attempt.json"
    attempt_path.write_text(
        json.dumps(sealed_attempt.to_dict(include_private_branch=False))
    )
    attempt_paths = [attempt_path]
    for entry in dataset.schedule[1:]:
        failure_dir = tmp_path / "outcomes" / entry.entry_id
        failure_dir.mkdir(parents=True)
        failure_payload = {
            "status": "INFRASTRUCTURE_FAILURE",
            "failure_type": "SyntheticTransportError",
            "message": "synthetic receipt-validation fixture",
            "traceback": "not executed",
            "selection_sha256": None,
            "collection_plan_sha256": _digest("7"),
            "scorer_verifier_auth_key_id": _digest("8"),
            "public_trace_sha256": None,
        }
        (failure_dir / "infrastructure_failure.json").write_text(
            json.dumps(failure_payload)
        )
        failure_attempt = CollectionAttempt(
            attempt_id=entry.entry_id,
            schedule_entry=entry,
            status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
            artifact_tree_sha256=_tree_sha256(failure_dir),
            infrastructure_failure=InfrastructureFailure(
                failure_type="SyntheticTransportError",
                message="synthetic receipt-validation fixture",
                diagnostics_sha256=canonical_sha256(failure_payload),
            ),
        )
        failure_path = failure_dir / "attempt.json"
        failure_path.write_text(json.dumps(failure_attempt.to_dict()))
        attempt_paths.append(failure_path)
    cache = QwenFeatureCache(freeze_dir / "feature_cache")
    cache.put(manifest.token_cache_sha256, field)
    verified_freeze = VerifiedDecisionFreeze(
        freeze_dir=freeze_dir.resolve(),
        manifest=manifest,
        schedule=dataset.schedule,
        freeze_receipt_sha256=_digest("6"),
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_collection_plan",
        lambda path, **_: VerifiedCollectionPlan(
            plan_path=Path(path).expanduser().resolve(),
            collection_plan_sha256=_digest("7"),
            scorer_verifier_auth_key_id=_digest("8"),
            freezes=(verified_freeze,),
            document={"groups": []},
        ),
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_execution_claim",
        lambda **_: {
            "output_root": str((tmp_path / "outcomes").resolve()),
            "selection_sha256": None,
            "collection_plan_sha256": _digest("7"),
            "scorer_verifier_auth_key_id": _digest("8"),
        },
    )

    rebuilt = build_method_v1_outcome_dataset(
        collection_plan_path=tmp_path / "collection-plan.json",
        collection_config_path=tmp_path / "config.json",
        freeze_dirs=(freeze_dir,),
        attempt_paths=tuple(attempt_paths),
        dataset_id="rebuilt-dataset",
    )
    records = method_v1_training_records(
        rebuilt, cache_roots=(freeze_dir / "feature_cache",)
    )
    assert len(records) == 1
    assert records[0].token_cache_sha256 == manifest.token_cache_sha256

    (attempt_dir / "public_execution_trace.json").write_text(json.dumps({"steps": 299}))
    with pytest.raises(ValueError, match="artifact tree digest mismatch"):
        build_method_v1_outcome_dataset(
            collection_plan_path=tmp_path / "collection-plan.json",
            collection_config_path=tmp_path / "config.json",
            freeze_dirs=(freeze_dir,),
            attempt_paths=tuple(attempt_paths),
            dataset_id="tampered-dataset",
        )

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset.to_dict()))
    loaded = load_method_v1_outcome_dataset(dataset_path)
    assert loaded.fingerprint() == dataset.fingerprint()
    with pytest.raises(TypeError, match="ReceiptBackedMethodV1Dataset"):
        method_v1_training_records(loaded, cache_roots=(freeze_dir / "feature_cache",))
    with pytest.raises(TypeError, match="ReceiptBackedMethodV1Dataset"):
        build_canonical_prediction_artifact(
            loaded,
            cache_roots=(freeze_dir / "feature_cache",),
            checkpoint_paths=(),
        )
    with pytest.raises(TypeError, match="must be created"):
        ReceiptBackedMethodV1Dataset()


def test_training_uses_train_validation_and_roundtrips_checkpoint(tmp_path) -> None:
    train_field = _field(context_character="a", use_state=True)
    validation_field = _field(context_character="1", use_state=True, state_offset=2.0)
    records = (
        _record(
            record_id="train-direct",
            decision_group_id="train-group",
            split="train",
            field=train_field,
            executed_index=0,
            outcome=True,
        ),
        _record(
            record_id="train-open",
            decision_group_id="train-group",
            split="train",
            field=train_field,
            executed_index=1,
            outcome=False,
        ),
        _record(
            record_id="validation-direct",
            decision_group_id="validation-group",
            split="validation",
            field=validation_field,
            executed_index=0,
            outcome=True,
        ),
        _record(
            record_id="validation-open",
            decision_group_id="validation-group",
            split="validation",
            field=validation_field,
            executed_index=1,
            outcome=False,
        ),
    )
    config = TrainingConfig(
        learning_rate=1e-3,
        effective_batch_size=2,
        micro_batch_size=2,
        max_epochs=2,
        early_stopping_patience=1,
        hidden_dim=8,
        num_heads=2,
        feedforward_dim=16,
        seeds=(5,),
    )
    result = train_one_seed(
        records,
        output_dir=tmp_path / "checkpoints",
        config=config,
        seed=5,
    )
    assert result.trainable_parameter_count > 0
    assert 1 <= result.epochs_completed <= 2
    assert len(result.checkpoint_sha256) == 64
    model, identity = load_checkpoint_model(result.checkpoint_path)
    assert identity["provider_id"] == train_field.public_context.provider_id
    assert identity["training_admission_evidence_sha256"] is None
    assert identity["model"]["use_public_state"] is True
    prediction = model(validation_field)
    assert prediction.task_success_logits.shape == (1, 2)
    assert torch.isfinite(prediction.task_success_logits).all()

    test_record = dataclasses.replace(
        records[-1],
        record_id="held-out",
        initial_state_group="held-out-reset",
        split_group_id="held-out-group",
        decision_group_id="held-out-decision",
        split="test",
    )
    with pytest.raises(ValueError, match="only train/validation"):
        train_one_seed(
            (*records, test_record),
            output_dir=tmp_path / "invalid",
            config=config,
            seed=5,
        )


def test_checkpoint_scores_and_selects_manifest_bound_candidates(tmp_path) -> None:
    _, manifest, field = _canonical_method_v1_dataset()
    contract_sha256 = manifest.outcome_contract.fingerprint()
    rows = tuple(
        dataclasses.replace(
            _record(
                record_id=f"{split}-{index}",
                decision_group_id=f"{split}-decision",
                split=split,
                field=field,
                executed_index=index,
                outcome=index == 0,
            ),
            outcome_contract_sha256=contract_sha256,
            experiment_id=manifest.experiment_id,
            configuration_sha256=manifest.configuration_sha256,
            proposal_provider_id=manifest.proposal_provider_id,
        )
        for split in ("train", "validation")
        for index in range(2)
    )
    config = TrainingConfig(
        learning_rate=1e-3,
        effective_batch_size=2,
        micro_batch_size=2,
        max_epochs=1,
        early_stopping_patience=1,
        hidden_dim=8,
        num_heads=2,
        feedforward_dim=16,
        seeds=(5,),
    )
    trained = train_one_seed(
        rows,
        output_dir=tmp_path / "checkpoint",
        config=config,
        seed=5,
    )
    cache = QwenFeatureCache(tmp_path / "feature_cache")
    cache.put(manifest.token_cache_sha256, field)
    result = select_manifest_candidate(
        manifest,
        cache_root=tmp_path / "feature_cache",
        checkpoint_path=trained.checkpoint_path,
        temperature=1.7,
    )
    assert result.decision.status.value == "SELECTED"
    assert result.temperature == 1.7
    assert len(result.predictions) == len(manifest.candidates)
    assert {item.candidate_id for item in result.predictions} == {
        item.candidate_id for item in manifest.candidates
    }
    assert result.to_dict()["evidence_scope"] == "PRE_EXECUTION_MODEL_DECISION"


def test_repository_config_loads_nested_training_contract() -> None:
    raw = json.loads(
        (Path(__file__).parents[1] / "experiments" / "method_v1.yaml").read_text()
    )
    config = _config_from_mapping(raw)
    assert config.optimizer == "AdamW"
    assert config.early_stopping_metric == "validation_nll"
    assert config.early_stopping_patience == 5
    assert config.seeds == (0, 1, 2)
    assert config.micro_batch_size == 8
    assert config.hidden_dim == raw["model"]["hidden_dim"]
    assert config.use_public_state is True
    overridden = _config_from_mapping(raw, micro_batch_size=4)
    assert overridden.micro_batch_size == 4

    invalid = json.loads(json.dumps(raw))
    invalid["training"]["optimizer"] = "SGD"
    with pytest.raises(ValueError, match="AdamW"):
        _config_from_mapping(invalid)
    invalid = json.loads(json.dumps(raw))
    invalid["training"]["early_stopping"]["metric"] = "validation_accuracy"
    with pytest.raises(ValueError, match="validation_nll"):
        _config_from_mapping(invalid)


def test_temperature_is_positive_and_requires_disjoint_groups() -> None:
    logits = torch.tensor([6.0, -6.0])
    labels = torch.tensor([0.0, 1.0])
    fit = fit_positive_temperature(
        logits,
        labels,
        calibration_group_ids=("cal-a", "cal-b"),
        training_split_group_ids=("train-a", "validation-a"),
    )
    assert fit.temperature > 0
    assert fit.calibrated_nll <= fit.raw_nll
    scaled = logits / fit.temperature
    assert torch.argsort(scaled).tolist() == torch.argsort(logits).tolist()
    with pytest.raises(ValueError, match="overlap"):
        fit_positive_temperature(
            logits,
            labels,
            calibration_group_ids=("cal-a", "train-a"),
            training_split_group_ids=("train-a",),
        )


def _prediction_rows() -> tuple[BranchPrediction, ...]:
    specifications = (
        ("group-1", "a", Primitive.DIRECT, 0.2, (False, False)),
        ("group-1", "b", Primitive.OPEN, 0.8, (True, True)),
        ("group-2", "c", Primitive.DIRECT, 0.8, (True, True)),
        ("group-2", "d", Primitive.OPEN, 0.2, (False, False)),
    )
    rows: list[BranchPrediction] = []
    for group_id, character, primitive, probability, outcomes in specifications:
        for repeat_index, outcome in enumerate(outcomes):
            rows.append(
                BranchPrediction(
                    record_id=f"{group_id}-{character}-{repeat_index}",
                    decision_group_id=group_id,
                    split_group_id=f"split-{group_id}",
                    candidate_id=f"candidate-{character}",
                    candidate_fingerprint=_digest(character),
                    primitive=primitive,
                    repeat_index=repeat_index,
                    observed_outcome=outcome,
                    scorer_probability=probability,
                    ensemble_probability=probability,
                    frozen_vlm_rank=(0 if primitive is Primitive.DIRECT else 1),
                )
            )
    return tuple(rows)


def test_probability_metrics_and_baselines_have_clear_evidence_scope() -> None:
    metrics = probability_metrics((0.9, 0.1), (True, False), num_bins=2)
    assert metrics.nll == pytest.approx(-math.log(0.9))
    assert metrics.brier == pytest.approx(0.01)
    assert metrics.expected_calibration_error == pytest.approx(0.1)

    rows = _prediction_rows()
    scorer = evaluate_offline_branch_matrix(
        rows,
        baseline=Baseline.SCORER,
        bootstrap_samples=100,
    )
    always_open = evaluate_offline_branch_matrix(
        rows,
        baseline=Baseline.ALWAYS_OPEN,
        bootstrap_samples=100,
    )
    assert scorer.evidence_scope == "NON_FORMAL_UNVERIFIED_BRANCH_PREDICTIONS"
    assert scorer.task_success.estimate == 1.0
    assert scorer.information_action_rate.estimate == 0.5
    assert always_open.task_success.estimate == 0.5


def test_closed_loop_summary_reports_efficiency_and_group_bootstrap() -> None:
    episodes = (
        PolicyEpisodeResult(
            episode_id="episode-1",
            reset_group_id="reset-a",
            policy_id="scorer-seed-5",
            final_task_success=True,
            selected_primitive=Primitive.OPEN,
            control_steps=300,
            vlm_calls=2,
            vla_calls=30,
            latency_seconds=3.0,
            proposal_covered_successful_candidate=True,
        ),
        PolicyEpisodeResult(
            episode_id="episode-2",
            reset_group_id="reset-b",
            policy_id="scorer-seed-5",
            final_task_success=False,
            selected_primitive=Primitive.DIRECT,
            control_steps=300,
            vlm_calls=1,
            vla_calls=30,
            latency_seconds=2.0,
            proposal_covered_successful_candidate=False,
        ),
    )
    summary = summarize_closed_loop_episodes(episodes, bootstrap_samples=100)
    assert summary.evidence_scope == "LIVE_CLOSED_LOOP_POLICY_EPISODES"
    assert summary.final_task_success.estimate == 0.5
    assert summary.information_action_rate == 0.5
    summary = summarize_closed_loop_episodes(
        (
            PolicyEpisodeResult(
                episode_id="episode-0",
                reset_group_id="reset-0",
                policy_id="method-seed-5",
                final_task_success=True,
                selected_primitive=Primitive.OPEN,
                control_steps=300,
                vlm_calls=2,
                vla_calls=30,
                latency_seconds=1.5,
                proposal_covered_successful_candidate=True,
            ),
            PolicyEpisodeResult(
                episode_id="episode-1",
                reset_group_id="reset-1",
                policy_id="method-seed-5",
                final_task_success=False,
                selected_primitive=Primitive.DIRECT,
                control_steps=300,
                vlm_calls=1,
                vla_calls=30,
                latency_seconds=1.0,
                proposal_covered_successful_candidate=False,
            ),
        ),
        bootstrap_samples=100,
    )
    assert summary.evidence_scope == "LIVE_CLOSED_LOOP_POLICY_EPISODES"
    assert summary.final_task_success.estimate == 0.5
    assert summary.information_action_rate == 0.5
    assert summary.mean_vlm_calls == 1.5
    assert summary.proposal_coverage == 0.5
