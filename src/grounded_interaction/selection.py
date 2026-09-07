"""Identity-safe selection over grounded intervention value predictions.

Selection is intentionally small: it validates that every prediction refers to
exactly one proposed physical intervention, removes candidates declared
infeasible by the execution stack, and maximizes predicted task-success
probability.  It does not require a singleton conformal set and therefore does
not abstain merely because several interventions are useful.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from enum import Enum

from .contracts import GroundedIntervention

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclasses.dataclass(frozen=True)
class ValuePrediction:
    """Stage-1 value prediction for one immutable candidate.

    ``success_probability`` has meaning only under the outcome contract used
    to train the scorer.  It is not a generic confidence or uncertainty score.
    ``feasible`` is an execution-capability decision, not a probability
    threshold applied by this selector.
    """

    candidate_id: str
    candidate_fingerprint: str
    success_probability: float
    feasible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _clean_text(self.candidate_id, name="candidate_id"),
        )
        fingerprint = str(self.candidate_fingerprint)
        if not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("candidate_fingerprint must be a lowercase SHA-256 digest")
        object.__setattr__(self, "candidate_fingerprint", fingerprint)

        probability = float(self.success_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("success_probability must be finite and in [0, 1]")
        object.__setattr__(self, "success_probability", probability)
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class SelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    ABSTAIN = "ABSTAIN"


@dataclasses.dataclass(frozen=True)
class SelectionDecision:
    """Result of identity validation, feasibility filtering, and argmax."""

    status: SelectionStatus
    candidate: GroundedIntervention | None
    prediction: ValuePrediction | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, SelectionStatus):
            raise TypeError("status must be a SelectionStatus")
        object.__setattr__(self, "reason", _clean_text(self.reason, name="reason"))
        if self.status is SelectionStatus.SELECTED:
            if not isinstance(self.candidate, GroundedIntervention):
                raise ValueError("a selected decision requires a candidate")
            if not isinstance(self.prediction, ValuePrediction):
                raise ValueError("a selected decision requires a prediction")
            if not self.prediction.feasible:
                raise ValueError("an infeasible prediction cannot be selected")
            if self.candidate.candidate_id != self.prediction.candidate_id:
                raise ValueError("selected candidate ID does not match prediction")
            self.candidate.assert_fingerprint(self.prediction.candidate_fingerprint)
        elif self.candidate is not None or self.prediction is not None:
            raise ValueError("an abstention cannot contain a candidate or prediction")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "candidate_id": (
                self.candidate.candidate_id if self.candidate is not None else None
            ),
            "candidate_fingerprint": (
                self.candidate.fingerprint() if self.candidate is not None else None
            ),
            "success_probability": (
                self.prediction.success_probability
                if self.prediction is not None
                else None
            ),
            "reason": self.reason,
        }


class ExpectedSuccessSelector:
    """Choose the feasible intervention with greatest predicted success.

    Exact ties preserve the proposal order.  ``STOP`` participates like any
    other candidate; the loop, rather than the selector, enforces that it never
    reaches a physical executor.
    """

    def select(
        self,
        candidates: Sequence[GroundedIntervention],
        predictions: Sequence[ValuePrediction],
    ) -> SelectionDecision:
        candidate_tuple = tuple(candidates)
        prediction_tuple = tuple(predictions)
        if any(
            not isinstance(candidate, GroundedIntervention)
            for candidate in candidate_tuple
        ):
            raise TypeError("candidates must contain GroundedIntervention values")
        if any(
            not isinstance(prediction, ValuePrediction)
            for prediction in prediction_tuple
        ):
            raise TypeError("predictions must contain ValuePrediction values")
        if not candidate_tuple:
            if prediction_tuple:
                raise ValueError("predictions must be empty when candidates are empty")
            return SelectionDecision(
                status=SelectionStatus.ABSTAIN,
                candidate=None,
                prediction=None,
                reason="no grounded intervention was proposed",
            )

        candidate_by_id: dict[str, GroundedIntervention] = {}
        for candidate in candidate_tuple:
            if candidate.candidate_id in candidate_by_id:
                raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r}")
            candidate_by_id[candidate.candidate_id] = candidate

        prediction_by_id: dict[str, ValuePrediction] = {}
        for prediction in prediction_tuple:
            if prediction.candidate_id in prediction_by_id:
                raise ValueError(
                    f"duplicate prediction for {prediction.candidate_id!r}"
                )
            prediction_by_id[prediction.candidate_id] = prediction

        candidate_ids = set(candidate_by_id)
        prediction_ids = set(prediction_by_id)
        if candidate_ids != prediction_ids:
            missing = sorted(candidate_ids - prediction_ids)
            extra = sorted(prediction_ids - candidate_ids)
            raise ValueError(
                "predictions must match candidates exactly; "
                f"missing={missing}, extra={extra}"
            )

        # Validate every identity before considering feasibility so a corrupt
        # prediction cannot be hidden behind ``feasible=False``.
        for candidate_id, candidate in candidate_by_id.items():
            candidate.assert_fingerprint(
                prediction_by_id[candidate_id].candidate_fingerprint
            )

        best_candidate: GroundedIntervention | None = None
        best_prediction: ValuePrediction | None = None
        for candidate in candidate_tuple:
            prediction = prediction_by_id[candidate.candidate_id]
            if not prediction.feasible:
                continue
            if (
                best_prediction is None
                or prediction.success_probability > best_prediction.success_probability
            ):
                best_candidate = candidate
                best_prediction = prediction

        if best_candidate is None or best_prediction is None:
            return SelectionDecision(
                status=SelectionStatus.ABSTAIN,
                candidate=None,
                prediction=None,
                reason="no feasible grounded intervention",
            )
        return SelectionDecision(
            status=SelectionStatus.SELECTED,
            candidate=best_candidate,
            prediction=best_prediction,
            reason="maximum predicted success among feasible interventions",
        )
