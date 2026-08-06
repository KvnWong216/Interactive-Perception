"""The episode loop: act, and at intervals, measure what the action distribution knows.

Control flow follows openpi's LIBERO example so that results stay comparable to
published numbers: settle the scene, then replan every ``replan_steps`` steps
from a fresh action chunk.  What is added is a probe.  At probe steps the policy
is queried ``probe_samples`` times on the *same frozen observation*, yielding an
action distribution rather than a single chunk.  The first sample is the one
executed, so probing changes what is measured and not what is done.

Probing is the dominant cost: a probe step is ``probe_samples`` inferences
instead of one.  ``probe_every`` trades temporal resolution of the uncertainty
curve against wall-clock, and should be set from the GPU budget rather than
left at the default.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from interaction_uncertainty.uncertainty import summarize_task_uncertainty
from interaction_uncertainty.v2.primitives import COARSE_ACTION_SPACE, CoarsePrimitive
from interaction_uncertainty.v2.vla_bridge import (
    ActionChunkSample,
    HypothesisAnchor,
    action_induced_belief,
)

from .anchors import (
    AnchorSpec,
    ResolvedAnchor,
    drawer_joint_value,
    eef_position,
    resolve_anchors,
    visible_pixels,
)
from .metrics import EpisodeOutcome
from .policy_client import PolicyBackend, build_observation
from .primitive_decoder import PrimitiveDecoderConfig, decode_primitive_evidence

__all__ = ["RolloutConfig", "StepRecord", "run_episode", "write_trace"]

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 400
    num_steps_wait: int = 10
    replan_steps: int = 5
    probe_samples: int = 16
    probe_every: int = 20
    resize_size: int = 224
    camera: str = "agentview"
    commitment_probability: float = 0.6
    # Pixels above which the target counts as visible. Not always zero: a
    # renderer can leak a pixel or two of a covered object, and the leak is
    # platform-dependent (T03 renders 1 px under EGL on Linux and 0 under CGL on
    # macOS). Treating a 1-px leak as "the information arrived" would mark the
    # endpoint reached at step 0 and silently void the metric. Tasks may
    # override this via `endpoint_visible_threshold` in the benchmark spec.
    endpoint_visible_threshold: int = 0
    decoder: PrimitiveDecoderConfig = dataclasses.field(
        default_factory=PrimitiveDecoderConfig
    )

    def __post_init__(self) -> None:
        if self.probe_samples < 2:
            raise ValueError("probe_samples must be >= 2 to form a distribution")
        if self.probe_every < self.replan_steps:
            raise ValueError("probe_every must be a multiple of replan_steps or larger")
        if not 0.0 < self.commitment_probability <= 1.0:
            raise ValueError("commitment_probability must lie in (0, 1]")


@dataclasses.dataclass
class StepRecord:
    """One probe point on the uncertainty-versus-time curve."""

    step: int
    eef_position: list[float]
    target_visible_pixels: int | None
    reveal_joint_value: float | None
    hypothesis_probabilities: dict[str, float]
    vacuity: float
    dissonance: float
    predictive_entropy: float
    normalized_predictive_entropy: float
    dirichlet_mutual_information: float
    sufficiency_mean: float
    primitive_evidence: dict[str, float]
    top_hypothesis: str
    top_probability: float
    mean_gripper_command: float
    mean_translation_norm: float
    # Fraction of samples whose decisiveness hit its ceiling. If this sits at
    # 1.0 for a whole run, `AttributionConfig.motion_scale` is miscalibrated for
    # this policy's action units: evidence becomes the sample count, S is
    # exactly K+N, and vacuity is a constant that measures nothing. The curve
    # still plots, which is why the diagnostic is recorded per probe.
    saturated_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _probe(
    policy: PolicyBackend,
    packet: Any,
    count: int,
) -> tuple[list[np.ndarray], list[ActionChunkSample]]:
    chunks = policy.sample_chunks(packet, count)
    return chunks, [ActionChunkSample.from_chunk(chunk) for chunk in chunks]


def _record_step(
    *,
    step: int,
    prompt: str,
    samples: Sequence[ActionChunkSample],
    origin: np.ndarray,
    anchors: Sequence[ResolvedAnchor],
    target_pixels: int | None,
    joint_value: float | None,
    config: RolloutConfig,
) -> StepRecord:
    attribution = config.decoder.attribution
    belief = action_induced_belief(
        prompt=prompt,
        samples=samples,
        eef_position=tuple(origin.tolist()),
        anchors=[HypothesisAnchor(item.label, item.position) for item in anchors],
        config=attribution,
        provenance="pi05_action_samples",
    )
    report = summarize_task_uncertainty(belief)
    norms = np.asarray([sample.translation_norm for sample in samples], dtype=np.float64)
    saturated = float(np.mean(norms >= attribution.motion_scale))
    probabilities = belief.hypotheses.mean
    top_label = max(probabilities, key=lambda key: probabilities[key])
    evidence = decode_primitive_evidence(
        samples,
        eef_position=tuple(origin.tolist()),
        anchors=anchors,
        config=config.decoder,
    )
    return StepRecord(
        step=step,
        eef_position=[float(value) for value in origin],
        target_visible_pixels=target_pixels,
        reveal_joint_value=joint_value,
        hypothesis_probabilities={key: float(value) for key, value in probabilities.items()},
        vacuity=report.vacuity,
        dissonance=report.dissonance,
        predictive_entropy=report.predictive_entropy,
        normalized_predictive_entropy=report.normalized_predictive_entropy,
        dirichlet_mutual_information=report.dirichlet_mutual_information,
        sufficiency_mean=report.sufficiency_mean,
        primitive_evidence={
            primitive.value: float(evidence[primitive])
            for primitive in COARSE_ACTION_SPACE
        },
        top_hypothesis=top_label,
        top_probability=float(probabilities[top_label]),
        mean_gripper_command=float(
            np.mean([sample.gripper_command for sample in samples])
        ),
        mean_translation_norm=float(norms.mean()),
        saturated_fraction=saturated,
    )


def run_episode(
    *,
    env: Any,
    policy: PolicyBackend,
    task: dict[str, Any],
    prompt: str,
    prompt_variant: str,
    anchor_specs: Sequence[AnchorSpec],
    seed: int,
    config: RolloutConfig | None = None,
    frames: list[np.ndarray] | None = None,
) -> tuple[EpisodeOutcome, list[StepRecord]]:
    """Run one scenario under one prompt variant and score it.

    ``frames``, when supplied, is filled with the policy-view RGB images so a
    demo video can be rendered alongside the uncertainty curve without a second
    rollout.
    """

    config = config or RolloutConfig()
    target = task.get("target")
    reveal_joint = task.get("reveal_joint_name")
    expected_terminal = str(task.get("expected_terminal", "TASK_SUCCESS"))
    visible_threshold = int(
        task.get("endpoint_visible_threshold", config.endpoint_visible_threshold)
    )

    env.seed(seed)
    obs = env.reset()

    records: list[StepRecord] = []
    max_pixels = 0
    steps_to_endpoint: int | None = None
    first_committed_anchor: str | None = None
    first_committed_step: int | None = None
    success = False
    error: str | None = None
    plan: list[np.ndarray] = []
    step = 0

    try:
        total_steps = config.max_steps + config.num_steps_wait
        while step < total_steps:
            if step < config.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                step += 1
                continue

            pixels = (
                visible_pixels(env, obs, camera=config.camera, instance=target)
                if target is not None and target in env.instance_to_id
                else None
            )
            if pixels is not None:
                max_pixels = max(max_pixels, pixels)
                if pixels > visible_threshold and steps_to_endpoint is None:
                    steps_to_endpoint = step

            if frames is not None:
                frames.append(
                    np.ascontiguousarray(np.flipud(obs[f"{config.camera}_image"]))
                )

            if not plan:
                packet = build_observation(obs, prompt, resize_size=config.resize_size)
                is_probe = (step - config.num_steps_wait) % config.probe_every == 0
                count = config.probe_samples if is_probe else 1
                chunks, samples = _probe(policy, packet, count)
                plan = list(chunks[0][: config.replan_steps])

                if is_probe:
                    anchors = resolve_anchors(env, anchor_specs)
                    record = _record_step(
                        step=step,
                        prompt=prompt,
                        samples=samples,
                        origin=eef_position(obs),
                        anchors=anchors,
                        target_pixels=pixels,
                        joint_value=(
                            drawer_joint_value(env, reveal_joint)
                            if reveal_joint
                            else None
                        ),
                        config=config,
                    )
                    records.append(record)
                    if (
                        first_committed_step is None
                        and record.top_probability >= config.commitment_probability
                    ):
                        first_committed_anchor = record.top_hypothesis
                        first_committed_step = step

            obs, _, done, _ = env.step(plan.pop(0).tolist())
            step += 1
            if done:
                success = True
                break
    except Exception as exc:  # noqa: BLE001 - recorded, not silently dropped
        error = f"{type(exc).__name__}: {exc}"

    endpoint_reached = max_pixels > visible_threshold
    committed_before_endpoint = bool(
        first_committed_step is not None
        and (steps_to_endpoint is None or first_committed_step < steps_to_endpoint)
    )
    not_found_evidence = float(
        np.mean(
            [record.primitive_evidence[CoarsePrimitive.NOT_FOUND.value] for record in records]
        )
        if records
        else 0.0
    )

    def _mean(attribute: str) -> float:
        return float(np.mean([getattr(item, attribute) for item in records])) if records else float("nan")

    outcome = EpisodeOutcome(
        task_id=str(task["id"]),
        prompt_variant=prompt_variant,
        seed=seed,
        steps=step,
        task_success=success,
        information_endpoint_reached=endpoint_reached,
        max_target_visible_pixels=max_pixels,
        steps_to_endpoint=steps_to_endpoint,
        first_committed_anchor=first_committed_anchor,
        first_committed_step=first_committed_step,
        committed_before_endpoint=committed_before_endpoint,
        # A continuous-action policy has no abstention channel, so it always
        # terminates by acting. Recording this explicitly is the measurement
        # behind the claim that baseline VLAs cannot decline a task.
        terminal_decision="TASK_SUCCESS" if success else "ACT_NO_ABSTENTION",
        expected_terminal=expected_terminal,
        mean_vacuity=_mean("vacuity"),
        min_vacuity=float(np.min([item.vacuity for item in records])) if records else float("nan"),
        mean_dissonance=_mean("dissonance"),
        mean_predictive_entropy=_mean("predictive_entropy"),
        not_found_evidence=not_found_evidence,
        saturated_fraction=(
            float(np.mean([item.saturated_fraction for item in records]))
            if records
            else 0.0
        ),
        error=error,
    )
    return outcome, records


def write_trace(
    path: Path,
    *,
    outcome: EpisodeOutcome,
    records: Sequence[StepRecord],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write one episode as JSONL: a header line, then one line per probe."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        header = {"kind": "episode", **outcome.to_dict()}
        if metadata:
            header["metadata"] = metadata
        file.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            file.write(
                json.dumps({"kind": "step", **record.to_dict()}, ensure_ascii=False) + "\n"
            )
