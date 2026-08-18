"""Explicit expected information-utility optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import ActionEffectForecast, CandidateAction, PIUDecision, Primitive, TaskBelief


class InformationUtilityOptimizer:
    def __init__(
        self,
        *,
        information_weight: float = 1.0,
        task_progress_weight: float = 1.0,
        cost_weight: float = 1.0,
        risk_weight: float = 1.0,
    ) -> None:
        self.information_weight = float(information_weight)
        self.task_progress_weight = float(task_progress_weight)
        self.cost_weight = float(cost_weight)
        self.risk_weight = float(risk_weight)

    def select(
        self,
        *,
        belief: TaskBelief,
        candidates: Sequence[CandidateAction],
        forecasts: Mapping[str, ActionEffectForecast],
        learned_progress: Mapping[str, float],
    ) -> PIUDecision:
        if not candidates:
            raise ValueError("candidate set cannot be empty")
        utilities: dict[str, dict[str, float]] = {}
        best: CandidateAction | None = None
        best_value = float("-inf")
        for candidate in candidates:
            if candidate.primitive is Primitive.ABSTAIN:
                eig = progress = execution = value = 0.0
            elif candidate.primitive in {Primitive.STOP_NOT_FOUND, Primitive.COMPLETE}:
                eig = execution = 0.0
                progress = float(learned_progress.get(candidate.candidate_id, 0.0))
                value = self.task_progress_weight * progress
            else:
                forecast = forecasts.get(candidate.candidate_id)
                if forecast is None:
                    raise ValueError(f"physical candidate lacks learned forecast: {candidate.candidate_id}")
                eig = belief.task_uncertainty - forecast.expected_future_uncertainty
                progress = float(learned_progress.get(candidate.candidate_id, forecast.expected_task_progress))
                execution = forecast.execution_success_probability
                value = execution * (
                    self.information_weight * eig
                    + self.task_progress_weight * progress
                ) - self.cost_weight * candidate.cost - self.risk_weight * (
                    candidate.physical_risk + (1.0 - execution)
                )
            utilities[candidate.candidate_id] = {
                "expected_information_gain": eig,
                "expected_task_progress": progress,
                "execution_success_probability": execution,
                "cost": candidate.cost,
                "physical_risk": candidate.physical_risk,
                "utility": value,
            }
            if value > best_value:
                best, best_value = candidate, value
        assert best is not None
        return PIUDecision(
            selected=best,
            utilities=utilities,
            task_uncertainty=belief.task_uncertainty,
            valid_candidate_ids=tuple(item.candidate_id for item in candidates),
            reason="maximum explicit expected information/task utility over hard-valid candidates",
        )
