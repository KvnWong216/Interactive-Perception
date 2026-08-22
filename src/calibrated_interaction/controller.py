"""Calibrated route/effect arbitration and executor-agnostic closed loop."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Protocol

from .calibration import BinaryEffectCalibration, LACCalibrator
from .capabilities import CapabilityRegistry
from .contracts import CandidateAction, EffectFactor, Primitive, validate_candidate_set


@dataclasses.dataclass(frozen=True)
class ModelPrediction:
    candidate_ids: tuple[str, ...]
    route_probabilities: Mapping[str, float]
    effect_positive_probabilities: Mapping[str, Mapping[EffectFactor | str, float]]


class DecisionKind(str, Enum):
    EXECUTE = "EXECUTE"
    STOP = "STOP"
    ABSTAIN = "ABSTAIN"


@dataclasses.dataclass(frozen=True)
class CalibratedDecision:
    kind: DecisionKind
    candidate_id: str | None
    route_prediction_set: tuple[str, ...]
    effect_prediction_sets: Mapping[str, Mapping[str, tuple[str, ...]]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "candidate_id": self.candidate_id,
            "route_prediction_set": list(self.route_prediction_set),
            "effect_prediction_sets": {
                candidate_id: {
                    factor: list(values) for factor, values in factors.items()
                }
                for candidate_id, factors in self.effect_prediction_sets.items()
            },
            "reason": self.reason,
        }


class CalibratedSelector:
    """Set-valued decision rule with no confidence-weighted utility formula."""

    def __init__(
        self,
        route_calibrator: LACCalibrator,
        effect_calibration: BinaryEffectCalibration,
    ) -> None:
        self.route_calibrator = route_calibrator
        self.effect_calibration = effect_calibration

    @staticmethod
    def _effect_reliable(
        candidate: CandidateAction, sets: Mapping[EffectFactor, tuple[str, ...]]
    ) -> bool:
        if sets[EffectFactor.EXECUTION_SUCCEEDED] != ("1",):
            return False
        if candidate.primitive.is_information_action:
            return sets[EffectFactor.AMBIGUITY_REDUCED] == ("1",)
        return True

    def select(
        self,
        candidates: Sequence[CandidateAction],
        prediction: ModelPrediction,
    ) -> CalibratedDecision:
        options = validate_candidate_set(candidates)
        by_id = {candidate.candidate_id: candidate for candidate in options}
        if tuple(by_id) != prediction.candidate_ids:
            raise ValueError(
                "model candidate order differs from validated candidate order"
            )
        route_set = self.route_calibrator.predict(prediction.route_probabilities)
        effect_sets = {
            candidate_id: self.effect_calibration.predict(
                prediction.effect_positive_probabilities[candidate_id]
            )
            for candidate_id in prediction.candidate_ids
        }
        serialized_effect_sets = {
            candidate_id: {factor.value: values for factor, values in sets.items()}
            for candidate_id, sets in effect_sets.items()
        }

        if len(route_set) == 1:
            selected = by_id[route_set[0]]
            if selected.primitive is Primitive.STOP:
                return CalibratedDecision(
                    DecisionKind.STOP,
                    selected.candidate_id,
                    route_set,
                    serialized_effect_sets,
                    "singleton calibrated STOP route",
                )
            if self._effect_reliable(selected, effect_sets[selected.candidate_id]):
                return CalibratedDecision(
                    DecisionKind.EXECUTE,
                    selected.candidate_id,
                    route_set,
                    serialized_effect_sets,
                    "singleton calibrated route with singleton-required effect evidence",
                )
            return CalibratedDecision(
                DecisionKind.ABSTAIN,
                None,
                route_set,
                serialized_effect_sets,
                "route is singleton but the required effect remains non-singleton",
            )

        reliable_information = [
            by_id[candidate_id]
            for candidate_id in route_set
            if candidate_id in by_id
            and by_id[candidate_id].primitive.is_information_action
            and self._effect_reliable(by_id[candidate_id], effect_sets[candidate_id])
        ]
        if len(reliable_information) == 1:
            selected = reliable_information[0]
            return CalibratedDecision(
                DecisionKind.EXECUTE,
                selected.candidate_id,
                route_set,
                serialized_effect_sets,
                "route is ambiguous but exactly one in-set physical information action has calibrated useful effects",
            )
        return CalibratedDecision(
            DecisionKind.ABSTAIN,
            None,
            route_set,
            serialized_effect_sets,
            "no unique calibrated information action resolves the route set",
        )


class ClosedLoopBackend(Protocol):
    def predict(
        self, *, prompt: str, history: Sequence[Mapping[str, Any]]
    ) -> tuple[Sequence[CandidateAction], ModelPrediction, Sequence[str]]: ...

    def execute(
        self, *, candidate: CandidateAction, instruction: str
    ) -> Mapping[str, Any]: ...


class ClosedLoopController:
    """Observe -> select -> frozen executor -> history update -> re-observe."""

    def __init__(
        self,
        *,
        selector: CalibratedSelector,
        capabilities: CapabilityRegistry,
        backend: ClosedLoopBackend,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.selector = selector
        self.capabilities = capabilities
        self.backend = backend
        self.max_steps = max_steps

    def run(self, *, prompt: str) -> dict[str, Any]:
        history: list[Mapping[str, Any]] = []
        steps: list[dict[str, Any]] = []
        terminal = "MAX_STEPS"
        for index in range(self.max_steps):
            candidates, prediction, frames = self.backend.predict(
                prompt=prompt, history=history
            )
            for candidate in candidates:
                self.capabilities.validate(candidate)
            decision = self.selector.select(candidates, prediction)
            step: dict[str, Any] = {
                "step": index,
                "public_observation_frames": list(frames),
                "candidates": [candidate.to_dict() for candidate in candidates],
                "decision": decision.to_dict(),
            }
            if decision.kind is DecisionKind.STOP:
                terminal = "STOP"
                steps.append(step)
                break
            if decision.kind is DecisionKind.ABSTAIN:
                terminal = "ABSTAIN"
                steps.append(step)
                break
            selected = next(
                candidate
                for candidate in candidates
                if candidate.candidate_id == decision.candidate_id
            )
            instruction = self.capabilities.serialize(selected, task_prompt=prompt)
            if instruction is None:
                raise RuntimeError(
                    "an executable decision serialized to no instruction"
                )
            result = dict(
                self.backend.execute(candidate=selected, instruction=instruction)
            )
            public_result = dict(result.get("public_result", {}))
            event = {
                "candidate_id": selected.candidate_id,
                "primitive": selected.primitive.value,
                "instruction": instruction,
                "public_result": public_result,
            }
            history.append(event)
            step["execution"] = event
            steps.append(step)
            if bool(result.get("controller_terminal", False)):
                terminal = "BACKEND_TERMINAL"
                break
        return {
            "schema_version": "calibrated-interaction.closed-loop-trace.v1",
            "prompt": " ".join(prompt.split()),
            "terminal": terminal,
            "steps": steps,
            "online_oracle_inputs": [],
        }
