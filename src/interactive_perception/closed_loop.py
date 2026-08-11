"""Public-observation routing and primitive-to-prompt execution.

The perception service receives RGB only. Simulator segmentation, object poses,
and evaluator-only target anchors are deliberately absent from this boundary.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np
from interaction_uncertainty.v2.target_belief import (
    InformationAction,
    LocationHypothesis,
    Reachability,
    TargetBelief,
    select_action,
)


@dataclasses.dataclass(frozen=True)
class PublicSceneEvidence:
    target_visible: bool
    target_sufficient: bool
    locations: Mapping[str, str]
    occluders: Mapping[str, str]
    confidence: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicSceneEvidence":
        allowed_location = {"closed", "open_unsearched", "searched_empty"}
        allowed_occluder = {"blocking", "cleared"}
        locations = {str(k): str(v) for k, v in dict(value.get("locations", {})).items()}
        occluders = {str(k): str(v) for k, v in dict(value.get("occluders", {})).items()}
        if any(state not in allowed_location for state in locations.values()):
            raise ValueError("invalid public location state")
        if any(state not in allowed_occluder for state in occluders.values()):
            raise ValueError("invalid public occluder state")
        confidence = float(value.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        return cls(
            target_visible=bool(value.get("target_visible", False)),
            target_sufficient=bool(value.get("target_sufficient", False)),
            locations=locations,
            occluders=occluders,
            confidence=confidence,
        )


class PublicPerception(Protocol):
    def infer(self, *, image: np.ndarray, prompt: str, task_id: str) -> PublicSceneEvidence: ...


@dataclasses.dataclass
class RemotePublicPerception:
    endpoint: str
    timeout_s: float = 30.0

    def infer(self, *, image: np.ndarray, prompt: str, task_id: str) -> PublicSceneEvidence:
        buffer = io.BytesIO()
        imageio.imwrite(buffer, np.asarray(image, dtype=np.uint8), format="png")
        payload = json.dumps(
            {
                "schema_version": "interactive-perception.public-evidence.v1",
                "task_id": task_id,
                "prompt": prompt,
                "rgb_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        return PublicSceneEvidence.from_dict(result)


@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    prompt: str | None
    primitive: str
    target: str | None
    terminal: str | None
    reason: str
    evidence: PublicSceneEvidence
    risks: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ClosedLoopPromptRouter:
    task: Mapping[str, Any]
    perception: PublicPerception
    minimum_confidence: float = 0.5
    decisions: list[RoutingDecision] = dataclasses.field(default_factory=list)

    def decide(
        self, *, obs: Mapping[str, Any], step: int, final_prompt: str, camera: str
    ) -> RoutingDecision:
        image = np.ascontiguousarray(np.flipud(np.asarray(obs[f"{camera}_image"])))
        evidence = self.perception.infer(
            image=image, prompt=final_prompt, task_id=str(self.task["id"])
        )
        if evidence.confidence < self.minimum_confidence:
            decision = RoutingDecision(
                prompt=None,
                primitive="SAFE_STOP",
                target=None,
                terminal="PERCEPTION_UNCERTAIN",
                reason="public perception confidence below the safety threshold",
                evidence=evidence,
            )
        elif evidence.target_visible and evidence.target_sufficient:
            decision = RoutingDecision(
                prompt=final_prompt,
                primitive="ACT",
                target=None,
                terminal=None,
                reason="target evidence is sufficient for the final task",
                evidence=evidence,
            )
        elif self.task.get("search_locations"):
            decision = self._route_locations(evidence, final_prompt)
        elif self.task.get("occluder_actions"):
            decision = self._route_occluders(evidence, final_prompt)
        else:
            decision = RoutingDecision(
                prompt=final_prompt,
                primitive="ACT",
                target=None,
                terminal=None,
                reason="task declares no routable information actions",
                evidence=evidence,
            )
        self.decisions.append(decision)
        return decision

    def _route_locations(
        self, evidence: PublicSceneEvidence, final_prompt: str
    ) -> RoutingDecision:
        candidates = []
        searched_prior = 0.0
        for item in self.task["search_locations"]:
            label = str(item["label"])
            state = evidence.locations.get(label, "closed")
            if state != "searched_empty":
                candidates.append((float(item["prior"]), item))
            else:
                searched_prior += float(item["prior"])
        if not candidates:
            return RoutingDecision(
                prompt=None,
                primitive="NOT_FOUND",
                target=None,
                terminal="NOT_FOUND",
                reason="every declared location was publicly observed and empty",
                evidence=evidence,
            )
        hypotheses = tuple(
            LocationHypothesis(
                label=str(item["label"]),
                reachability=Reachability.MANIPULATION_ONLY,
                resolving_action=InformationAction.REMOVE_OCCLUDER,
            )
            for _, item in candidates
        ) + (LocationHypothesis("ABSENT", Reachability.ABSENT),)
        raw_alpha = [prior for prior, _ in candidates] + [0.05 + searched_prior]
        normalizer = sum(raw_alpha)
        belief = TargetBelief(
            hypotheses=hypotheses,
            alpha=tuple(2.0 * value / normalizer for value in raw_alpha),
            prior_weight=2.0,
        )
        action, _, risks = select_action(belief, loss_false_absent=4.0)
        if action is InformationAction.NOT_FOUND:
            return RoutingDecision(
                prompt=None,
                primitive="NOT_FOUND",
                target=None,
                terminal="NOT_FOUND",
                reason="Bayes risk favors absence after the completed public search",
                evidence=evidence,
                risks=risks,
            )
        _, selected = max(candidates, key=lambda pair: pair[0])
        return RoutingDecision(
            prompt=str(selected["action_prompt"]),
            primitive="REMOVE_OCCLUDER",
            target=str(selected["label"]),
            terminal=None,
            reason="Bayes risk favors information; select the highest posterior unresolved location",
            evidence=evidence,
            risks=risks,
        )

    def _route_occluders(
        self, evidence: PublicSceneEvidence, final_prompt: str
    ) -> RoutingDecision:
        blocking = [
            item
            for item in self.task["occluder_actions"]
            if evidence.occluders.get(str(item["label"]), "blocking") != "cleared"
        ]
        if blocking:
            hypotheses = tuple(
                LocationHypothesis(
                    label=str(item["label"]),
                    reachability=Reachability.MANIPULATION_ONLY,
                    resolving_action=InformationAction.REMOVE_OCCLUDER,
                )
                for item in blocking
            ) + (LocationHypothesis("ABSENT", Reachability.ABSENT),)
            belief = TargetBelief(
                hypotheses=hypotheses,
                alpha=tuple([1.9 / len(blocking)] * len(blocking) + [0.1]),
                prior_weight=2.0,
            )
            _, _, risks = select_action(belief, loss_false_absent=4.0)
        else:
            risks = {}
        for item in blocking:
            label = str(item["label"])
            return RoutingDecision(
                prompt=str(item["action_prompt"]),
                primitive="REMOVE_OCCLUDER",
                target=label,
                terminal=None,
                reason="Bayes risk favors information and this occluder remains blocking",
                evidence=evidence,
                risks=risks,
            )
        return RoutingDecision(
            prompt=final_prompt,
            primitive="ACT",
            target=None,
            terminal=None,
            reason="all declared occluders are clear; retry final task",
            evidence=evidence,
        )
