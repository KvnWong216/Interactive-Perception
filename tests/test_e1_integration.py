from __future__ import annotations

import dataclasses
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from grounded_interaction.conditioning import (
    ReferentConditioner,
    draw_public_grounding_marker,
)
from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    canonical_sha256,
)
from grounded_interaction.e1 import (
    E1Plan,
    run_e1_trial,
    validate_e1_artifacts,
    validate_e1_runtime_preflight,
)
from grounded_interaction.e1_training import (
    collate_e1_outcome_targets,
    load_e1_training_record,
)
from grounded_interaction.libero_runtime import (
    LiberoPublicObservation,
    canonical_libero_state_sha256,
)
from grounded_interaction.molmoact2 import (
    MolmoAct2HTTPClient,
    MolmoAct2PolicyOutputError,
    MolmoAct2ServerIdentity,
    make_molmoact2_http_server,
)
from grounded_interaction.rgb import (
    RGBFrameStore,
    canonical_rgb_sha256,
    decode_rgb_png,
)
from grounded_interaction.serialization import (
    GroundedTextSerializer,
    SpatialConditioningMode,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments/e1_referent_ceiling/pilot_state0_v1.json"


def _rgb(value: int = 0) -> np.ndarray:
    return np.full((256, 256, 3), value, dtype=np.uint8)


def _context_and_candidate(
    store: RGBFrameStore,
    *,
    candidate_id: str,
    point_x: float,
) -> tuple[PolicyContext, GroundedIntervention]:
    agent = store.put(_rgb(10), frame_id="pre-agent", camera="agentview", frame_index=0)
    wrist = store.put(_rgb(20), frame_id="pre-wrist", camera="wrist", frame_index=0)
    context = PolicyContext(
        prompt="Put the selected moka pot on the stove.",
        frames=(agent, wrist),
        proprioception=(0.0,) * 8,
    )
    candidate = GroundedIntervention(
        candidate_id=candidate_id,
        primitive=Primitive.DIRECT,
        referent="moka pot",
        parameters=(("destination", "stove cook region"),),
        grounding=GroundingReference(
            camera="agentview",
            frame_id=agent.frame_id,
            frame_index=0,
            image_sha256=agent.image_sha256,
            box_xyxy=(point_x - 0.08, 0.42, point_x + 0.08, 0.68),
            point_xy=(point_x, 0.55),
        ),
    )
    return context, candidate


def test_rgb_store_hashes_decoded_pixels_and_rejects_tampering(tmp_path: Path) -> None:
    store = RGBFrameStore(tmp_path)
    frame = store.put(_rgb(17), frame_id="f0", camera="agentview", frame_index=0)
    assert canonical_rgb_sha256(store.resolve(frame)) == frame.image_sha256
    assert np.array_equal(decode_rgb_png(store.path_for(frame).read_bytes()), _rgb(17))
    store.path_for(frame).write_bytes(b"not a png")
    with pytest.raises(OSError):
        store.resolve(frame)


def test_precise_text_removes_the_reproduced_coarse_grid_collision(
    tmp_path: Path,
) -> None:
    store = RGBFrameStore(tmp_path)
    context, left = _context_and_candidate(store, candidate_id="left", point_x=0.4)
    _, right = _context_and_candidate(store, candidate_id="right", point_x=0.6)
    coarse = GroundedTextSerializer(spatial_mode=SpatialConditioningMode.COARSE_TEXT)
    precise = GroundedTextSerializer(spatial_mode=SpatialConditioningMode.PRECISE_TEXT)
    assert left.fingerprint() != right.fingerprint()
    assert (
        coarse.serialize(left, context).subtask_text
        == coarse.serialize(right, context).subtask_text
    )
    assert (
        precise.serialize(left, context).subtask_text
        != precise.serialize(right, context).subtask_text
    )


def test_visual_marker_is_public_and_changes_the_executed_rgb(tmp_path: Path) -> None:
    store = RGBFrameStore(tmp_path / "frames")
    context, candidate = _context_and_candidate(store, candidate_id="left", point_x=0.4)
    serializer = GroundedTextSerializer(
        spatial_mode=SpatialConditioningMode.VISUAL_MARKER
    )
    serialized = serializer.serialize(candidate, context)
    conditioned = ReferentConditioner(store).prepare(
        context=context,
        intervention=candidate,
        serialized=serialized,
        mode=SpatialConditioningMode.VISUAL_MARKER,
    )
    assert conditioned.provenance["manual_public_rgb_grounding"] is True
    assert conditioned.provenance["native_vla_spatial_api_used"] is False
    assert (
        canonical_rgb_sha256(conditioned.agentview_rgb)
        != context.frames[0].image_sha256
    )
    assert np.array_equal(conditioned.wrist_rgb, _rgb(20))
    direct = draw_public_grounding_marker(
        _rgb(10), box_xyxy=(0.32, 0.42, 0.48, 0.68), point_xy=(0.4, 0.55)
    )
    assert np.array_equal(direct, conditioned.agentview_rgb)


class _FakeBackend:
    def __init__(
        self,
        *,
        invalid: bool = False,
        identity: MolmoAct2ServerIdentity | None = None,
    ) -> None:
        self.identity = identity or MolmoAct2ServerIdentity()
        self.runtime_identity = {
            "backend_kind": "faithful-test-double",
            "snapshot_revision": self.identity.checkpoint_revision,
        }
        self.invalid = invalid
        self.calls: list[dict[str, object]] = []
        self.resets: list[str] = []

    def reset(self, session_id: str) -> None:
        self.resets.append(session_id)

    def predict(
        self,
        *,
        agentview_rgb: object,
        wrist_rgb: object,
        state: object,
        instruction: str,
        seed: int,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "agentview_sha256": canonical_rgb_sha256(agentview_rgb),
                "wrist_sha256": canonical_rgb_sha256(wrist_rgb),
                "state": tuple(state),
                "instruction": instruction,
                "seed": seed,
            }
        )
        count = 1 if self.invalid else 10
        return [[0.0] * 7 for _ in range(count)]


def _client_for_backend(backend: _FakeBackend):
    server = make_molmoact2_http_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = MolmoAct2HTTPClient(
        f"http://{host}:{port}", expected_identity=backend.identity, timeout_seconds=5
    )
    return server, thread, client


def test_http_boundary_transports_exact_public_inputs_and_ten_actions() -> None:
    backend = _FakeBackend()
    server, thread, client = _client_for_backend(backend)
    try:
        client.reset("session-e1")
        result = client.predict_action_chunk(
            agentview_rgb=_rgb(1),
            wrist_rgb=_rgb(2),
            state=(0.0,) * 8,
            instruction="Put the left moka pot on the stove.",
            session_id="session-e1",
            seed=7,
        )
        assert backend.resets == ["session-e1"]
        assert len(result.actions) == 10
        assert backend.calls == [
            {
                "agentview_sha256": canonical_rgb_sha256(_rgb(1)),
                "wrist_sha256": canonical_rgb_sha256(_rgb(2)),
                "state": (0.0,) * 8,
                "instruction": "Put the left moka pot on the stove.",
                "seed": 7,
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_boundary_classifies_invalid_model_chunk_as_policy_output() -> None:
    backend = _FakeBackend(invalid=True)
    server, thread, client = _client_for_backend(backend)
    try:
        with pytest.raises(MolmoAct2PolicyOutputError, match="exactly 10"):
            client.predict_action_chunk(
                agentview_rgb=_rgb(1),
                wrist_rgb=_rgb(2),
                state=(0.0,) * 8,
                instruction="Move.",
                session_id="session-e1",
                seed=7,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_e1_test_double_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_policy_output: bool = False,
) -> tuple[E1Plan, Path, dict[str, object] | None]:
    base_plan = E1Plan.load(PLAN_PATH)
    backend = _FakeBackend(
        invalid=invalid_policy_output,
        identity=base_plan.server_identity,
    )
    server, thread, _client = _client_for_backend(backend)
    initial_agent = _rgb(31)
    initial_wrist = _rgb(47)
    initial_state = (0.0,) * 8
    plan = dataclasses.replace(
        base_plan,
        reset_state_sha256="f" * 64,
        expected_agentview_sha256=canonical_rgb_sha256(initial_agent),
        expected_wrist_sha256=canonical_rgb_sha256(initial_wrist),
        expected_state=initial_state,
        expected_state_sha256=canonical_libero_state_sha256(initial_state),
    )

    class FakeEnvironment:
        def __init__(self, **_: object) -> None:
            self.reset_state_sha256 = plan.reset_state_sha256
            self.control_mode = "relative"
            self.step_count = 0
            self.public_observation = LiberoPublicObservation(
                initial_agent, initial_wrist, initial_state
            )

        def step(
            self, action: object
        ) -> tuple[LiberoPublicObservation, tuple[float, ...]]:
            applied = np.asarray(tuple(action), dtype=np.float32)
            assert applied.shape == (7,)
            applied[-1] = -1.0 if applied[-1] < 0 else 1.0
            self.step_count += 1
            return self.public_observation, tuple(float(item) for item in applied)

        def contacts(self, object_names: tuple[str, ...]) -> tuple[str, ...]:
            return (object_names[0],) if self.step_count == 1 else ()

        def is_grasping(self, object_name: str) -> bool:
            assert object_name
            return False

        def predicate(self, predicate: tuple[str, ...]) -> bool:
            assert predicate
            return False

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    import grounded_interaction.e1 as e1_module

    monkeypatch.setattr(e1_module, "LiberoE1Environment", FakeEnvironment)
    monkeypatch.setattr(
        e1_module,
        "_resolve_libero_paths",
        lambda *_args, **_kwargs: (tmp_path / "task.bddl", tmp_path / "states"),
    )
    try:
        host, port = server.server_address
        if invalid_policy_output:
            with pytest.raises(MolmoAct2PolicyOutputError, match="exactly 10"):
                run_e1_trial(
                    plan=plan,
                    trial_id="coarse-left",
                    libero_root=tmp_path,
                    endpoint=f"http://{host}:{port}",
                    output_root=tmp_path / "runs",
                    allow_test_backend=True,
                )
            report = None
        else:
            report = run_e1_trial(
                plan=plan,
                trial_id="coarse-left",
                libero_root=tmp_path,
                endpoint=f"http://{host}:{port}",
                output_root=tmp_path / "runs",
                allow_test_backend=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return plan, tmp_path / "runs" / plan.plan_id / "coarse-left", report


@pytest.fixture
def e1_test_double_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[E1Plan, Path, dict[str, object]]:
    plan, run_dir, report = _run_e1_test_double_artifact(tmp_path, monkeypatch)
    assert report is not None
    return plan, run_dir, report


def test_e1_runner_exports_a_sealed_trainable_observed_branch(
    e1_test_double_artifact: tuple[E1Plan, Path, dict[str, object]],
) -> None:
    plan, run_dir, report = e1_test_double_artifact
    assert report["empirical_execution"] is False
    assert report["diagnostics"]["first_contact_intended"] is True
    assert report["diagnostics"]["full_physical_goal"] is False
    verification = validate_e1_artifacts(
        run_dir,
        expected_plan=plan,
        expected_trial_id="coarse-left",
    )
    assert verification["status"] == "VERIFIED_COMPLETED"
    assert verification["evidence_class"] == "SOFTWARE_TEST_DOUBLE"
    with pytest.raises(ValueError, match="live MolmoAct2"):
        validate_e1_artifacts(
            run_dir,
            require_empirical=True,
            expected_plan=plan,
            expected_trial_id="coarse-left",
        )
    branch = json.loads((run_dir / "observed_branch.public.json").read_text())
    assert branch["supervision"]["observed_outcome"] is True
    assert "moka_pot_1" not in (run_dir / "public_execution_trace.json").read_text()

    with pytest.raises(ValueError, match="live MolmoAct2"):
        load_e1_training_record(
            run_dir,
            expected_plan=plan,
            expected_trial_id="coarse-left",
        )
    record = load_e1_training_record(
        run_dir,
        expected_plan=plan,
        expected_trial_id="coarse-left",
        require_empirical=False,
    )
    assert record.evidence_class == "SOFTWARE_TEST_DOUBLE"
    assert record.branch.observed_outcome is True
    with pytest.raises(ValueError, match="complete-intervention"):
        dataclasses.replace(
            record,
            branch=dataclasses.replace(
                record.branch,
                execution_status=ExecutionStatus.FAILED,
            ),
        )


def test_e1_runner_never_labels_malformed_policy_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, run_dir, report = _run_e1_test_double_artifact(
        tmp_path,
        monkeypatch,
        invalid_policy_output=True,
    )
    assert report is None
    verification = validate_e1_artifacts(
        run_dir,
        expected_plan=plan,
        expected_trial_id="coarse-left",
    )
    assert verification["status"] == "VERIFIED_INFRASTRUCTURE_FAILURE"
    assert not (run_dir / "completed.json").exists()
    assert not (run_dir / "observed_branch.public.json").exists()


def test_e1_training_record_reaches_actual_pytorch_loss(
    e1_test_double_artifact: tuple[E1Plan, Path, dict[str, object]],
) -> None:
    torch = pytest.importorskip("torch")
    plan, run_dir, _ = e1_test_double_artifact
    record = load_e1_training_record(
        run_dir,
        expected_plan=plan,
        expected_trial_id="coarse-left",
        require_empirical=False,
    )
    targets = collate_e1_outcome_targets((record,))
    assert not hasattr(targets, "candidate_ids")
    logits = torch.nn.Parameter(torch.zeros_like(targets.observed_outcomes))
    fingerprint_rows = (targets.candidate_fingerprints,)
    loss = targets.loss(
        logits,
        prediction_candidate_fingerprints=fingerprint_rows,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_e1_plan_is_byte_frozen_and_keeps_physical_candidate_identity(
    tmp_path: Path,
) -> None:
    plan = E1Plan.load(PLAN_PATH)
    assert len(plan.trials) == 6
    assert {trial.seed for trial in plan.trials} == {26090800}
    for target in {trial.target_object for trial in plan.trials}:
        rows = [trial for trial in plan.trials if trial.target_object == target]
        assert len({trial.candidate_id for trial in rows}) == 1
        assert len({(trial.box_xyxy, trial.point_xy) for trial in rows}) == 1
    raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    frozen = raw.pop("frozen_plan_sha256")
    assert frozen == canonical_sha256(raw)
    raw["prompt"] = "tampered"
    raw["frozen_plan_sha256"] = frozen
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        E1Plan.load(tampered)
    assert frozen != canonical_sha256(raw)


@pytest.mark.skipif(
    not os.environ.get("LIBERO_REPO_ROOT"),
    reason="external LIBERO checkout is not configured",
)
def test_real_libero_reset_and_rgb_preflight() -> None:
    plan = E1Plan.load(PLAN_PATH)
    result = validate_e1_runtime_preflight(
        plan, libero_root=os.environ["LIBERO_REPO_ROOT"]
    )
    assert result["status"] == "VALID"
    assert (
        result["scored_observation"]["agentview_sha256"]
        == plan.expected_agentview_sha256
    )
    assert result["scored_observation"]["wrist_sha256"] == plan.expected_wrist_sha256
