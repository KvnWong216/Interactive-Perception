"""Deterministic software smoke test for the formal-v1 pipeline.

The fixture contains no simulator rollout and no learned checkpoint. It checks
that the tensor path is finite and that one immutable candidate identity
survives an ``OPEN -> reobserve -> DIRECT -> reobserve -> STOP`` replay.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    OutcomeContract,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
)
from .execution import ReplayExecutor, ReplayOutcome
from .loop import ClosedLoop, PublicReplayObserver
from .selection import ValuePrediction


def _digest(character: str) -> str:
    return character * 64


def _frame(index: int, character: str) -> PublicFrame:
    return PublicFrame(
        frame_id=f"wrist-{index:03d}",
        camera="wrist",
        frame_index=index,
        image_sha256=_digest(character),
        width=256,
        height=256,
    )


def _candidate(
    context: PolicyContext,
    *,
    candidate_id: str,
    primitive: Primitive,
    referent: str,
    point: tuple[float, float],
    parameters: tuple[tuple[str, str], ...] = (),
) -> GroundedIntervention:
    frame = max(
        (item for item in context.frames if item.camera == "wrist"),
        key=lambda item: item.frame_index,
    )
    x, y = point
    half_width = 0.18
    half_height = 0.12
    grounding = GroundingReference(
        camera=frame.camera,
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        image_sha256=frame.image_sha256,
        box_xyxy=(
            max(0.0, x - half_width),
            max(0.0, y - half_height),
            min(1.0, x + half_width),
            min(1.0, y + half_height),
        ),
        point_xy=point,
    )
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent=referent,
        parameters=parameters,
        grounding=grounding,
    )


def _stop() -> GroundedIntervention:
    return GroundedIntervention(
        candidate_id="stop",
        primitive=Primitive.STOP,
        referent=None,
        parameters=(),
        grounding=None,
    )


class _ReplayProposer:
    def propose(self, context: PolicyContext) -> tuple[GroundedIntervention, ...]:
        step = len(context.public_history)
        if step == 0:
            return (
                _candidate(
                    context,
                    candidate_id="direct-unresolved",
                    primitive=Primitive.DIRECT,
                    referent="the currently unresolved butter target",
                    point=(0.50, 0.55),
                ),
                _candidate(
                    context,
                    candidate_id="open-middle",
                    primitive=Primitive.OPEN,
                    referent="the middle drawer below the countertop",
                    point=(0.50, 0.55),
                    parameters=(("direction", "pull outward"),),
                ),
                _stop(),
            )
        if step == 1:
            return (
                _candidate(
                    context,
                    candidate_id="direct-butter",
                    primitive=Primitive.DIRECT,
                    referent="the revealed butter in the middle drawer",
                    point=(0.58, 0.55),
                ),
                _candidate(
                    context,
                    candidate_id="open-bottom",
                    primitive=Primitive.OPEN,
                    referent="the bottom drawer below the countertop",
                    point=(0.50, 0.80),
                    parameters=(("direction", "pull outward"),),
                ),
                _stop(),
            )
        return (_stop(),)


class _ReplayScorer:
    _SCORES = (
        {"direct-unresolved": 0.15, "open-middle": 0.90, "stop": 0.05},
        {"direct-butter": 0.95, "open-bottom": 0.10, "stop": 0.05},
        {"stop": 1.00},
    )

    def score(
        self,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
    ) -> tuple[ValuePrediction, ...]:
        scores = self._SCORES[min(len(context.public_history), 2)]
        return tuple(
            ValuePrediction(
                candidate_id=candidate.candidate_id,
                candidate_fingerprint=candidate.fingerprint(),
                success_probability=scores[candidate.candidate_id],
                feasible=True,
            )
            for candidate in candidates
        )


def _token_model_smoke(
    context: PolicyContext,
    candidates: tuple[GroundedIntervention, ...],
) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - CLI dependency error.
        raise RuntimeError(
            "The formal-v1 tensor smoke requires `uv sync --extra learned`."
        ) from error

    from .losses import executed_candidate_bernoulli_nll
    from .model import GroundedOutcomeModel
    from .tokens import CandidateTokenField, FrozenTokenField

    torch.manual_seed(7)
    width = 16
    token_count = 6
    context_tokens = torch.linspace(
        -1.0,
        1.0,
        steps=token_count * width,
        dtype=torch.float32,
    ).reshape(1, token_count, width)
    candidate_tokens = torch.stack(
        [
            torch.tensor(
                [
                    (byte - 127.5) / 127.5
                    for byte in bytes.fromhex(item.fingerprint())[:width]
                ],
                dtype=torch.float32,
            )
            for item in candidates
        ]
    ).unsqueeze(0)
    latest = max(
        (frame for frame in context.frames if frame.camera == "wrist"),
        key=lambda frame: frame.frame_index,
    )
    patch_boxes = torch.zeros(1, token_count, 4)
    patch_boxes[0, 1:5] = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ]
    )
    token_field = FrozenTokenField(
        tokens=context_tokens,
        valid_mask=torch.ones(1, token_count, dtype=torch.bool),
        current_patch_mask=torch.tensor([[False, True, True, True, True, False]]),
        camera_ids=((None, "wrist", "wrist", "wrist", "wrist", None),),
        frame_ids=(
            (
                None,
                latest.frame_id,
                latest.frame_id,
                latest.frame_id,
                latest.frame_id,
                None,
            ),
        ),
        patch_xyxy=patch_boxes,
        context_fingerprints=(context.fingerprint(),),
        provider_id="synthetic-token-provider-v1",
    )
    support = torch.zeros(1, len(candidates), token_count, dtype=torch.bool)
    for candidate_index, candidate in enumerate(candidates):
        if candidate.grounding is None:
            continue
        x, y = candidate.grounding.point_xy
        column = min(int(x * 2), 1)
        row = min(int(y * 2), 1)
        support[0, candidate_index, 1 + row * 2 + column] = True
    candidate_field = CandidateTokenField(
        tokens=candidate_tokens,
        valid_mask=torch.ones(1, len(candidates), dtype=torch.bool),
        candidate_ids=(tuple(item.candidate_id for item in candidates),),
        candidate_fingerprints=(tuple(item.fingerprint() for item in candidates),),
        primitives=(tuple(item.primitive for item in candidates),),
        grounding_support=support,
        public_context=token_field,
    )
    model = GroundedOutcomeModel(
        context_dim=width,
        candidate_dim=width,
        hidden_dim=width,
        num_heads=4,
    )
    grounded = model.candidate_encoder(candidate_field)
    prediction = model.outcomes(grounded)
    executed = torch.zeros(1, len(candidates), dtype=torch.bool)
    executed[0, 1] = True
    labels = torch.full((1, len(candidates)), float("nan"))
    labels[0, 1] = 1.0
    loss = executed_candidate_bernoulli_nll(
        prediction.task_success_logits,
        labels,
        executed,
    )
    loss.backward()
    probabilities = torch.sigmoid(prediction.task_success_logits).detach()
    return {
        "context_token_shape": list(token_field.tokens.shape),
        "candidate_token_shape": list(candidate_field.tokens.shape),
        "grounded_token_shape": list(grounded.grounded_tokens.shape),
        "success_logit_shape": list(prediction.task_success_logits.shape),
        "probabilities_finite": bool(torch.isfinite(probabilities).all()),
        "executed_only_loss_finite": bool(torch.isfinite(loss)),
        "backward_completed": True,
    }


def build_smoke_report() -> dict[str, object]:
    initial_context = PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(_frame(0, "a"),),
        public_history=(),
        proprioception=(0.0, 0.0, 0.0),
    )
    proposer = _ReplayProposer()
    initial_candidates = proposer.propose(initial_context)
    open_candidate = initial_candidates[1]

    revealed_context = PolicyContext(
        prompt=initial_context.prompt,
        frames=initial_context.frames + (_frame(1, "b"),),
        public_history=(
            PublicActionEvent(
                step_index=0,
                primitive=Primitive.OPEN,
                subtask_text="Open the middle drawer below the countertop.",
                execution_status=ExecutionStatus.COMPLETED,
            ),
        ),
        proprioception=initial_context.proprioception,
    )
    direct_candidate = proposer.propose(revealed_context)[0]
    executor = ReplayExecutor(
        {
            open_candidate.fingerprint(): ReplayOutcome(
                status=ExecutionStatus.COMPLETED,
                post_frames=(_frame(1, "b"),),
            ),
            direct_candidate.fingerprint(): ReplayOutcome(
                status=ExecutionStatus.COMPLETED,
                post_frames=(_frame(2, "c"),),
            ),
        }
    )
    trace = ClosedLoop(
        observer=PublicReplayObserver(initial_context),
        proposer=proposer,
        scorer=_ReplayScorer(),
        executor=executor,
    ).run(max_steps=3)
    selected = [
        step.decision.candidate.candidate_id
        for step in trace.steps
        if step.decision.candidate is not None
    ]
    identity_chain_verified = all(
        step.request is None
        or (
            step.receipt is not None
            and step.request.candidate_id == step.receipt.candidate_id
            and step.request.candidate_fingerprint == step.receipt.candidate_fingerprint
            and step.request.request_digest == step.receipt.request_digest
        )
        for step in trace.steps
    )
    contract = OutcomeContract(
        outcome_name="synthetic_task_success_within_horizon",
        continuation_policy_id="synthetic-replay-continuation-v1",
        horizon=3,
        executor_id=executor.executor_id,
        serializer_id="grounded-text-v1",
        failure_handling="synthetic_fixture_only",
    )
    return {
        "schema_version": "grounded-interaction.smoke.v1",
        "software_verification_only": True,
        "empirical_evidence": False,
        "trained_checkpoint_used": False,
        "real_vlm_used": False,
        "real_vla_used": False,
        "outcome_contract": contract.to_dict(),
        "outcome_contract_fingerprint": contract.fingerprint(),
        "tensor_path": _token_model_smoke(initial_context, initial_candidates),
        "replay_path": {
            "expected_selected_sequence": [
                "open-middle",
                "direct-butter",
                "stop",
            ],
            "selected_sequence": selected,
            "sequence_verified": selected == ["open-middle", "direct-butter", "stop"],
            "identity_chain_verified": identity_chain_verified,
            "trace": trace.to_dict(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/formal_v1_smoke.json"),
    )
    args = parser.parse_args()
    report = build_smoke_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema_version",
                    "software_verification_only",
                    "empirical_evidence",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
