"""Lightweight outcome recognition on frozen before/after VLA features."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from .prompt_state import BalancedRidgeBinary
from .active_risk import EffectOutcome
from .semantic_conformal import MondrianSemanticConformalCalibrator


# pi0.5 receives 256x256 policy images and represents each real camera with a
# 16x16 visual-token grid.  Evaluator masks smaller than one token footprint
# are useful visibility diagnostics, but are not treated as action-resolvable
# semantic evidence.
PI05_POLICY_IMAGE_SIDE = 256
PI05_VISUAL_TOKEN_GRID_SIDE = 16
PI05_PATCH_EQUIVALENT_TARGET_PIXELS = (
    PI05_POLICY_IMAGE_SIDE // PI05_VISUAL_TOKEN_GRID_SIDE
) ** 2


@dataclasses.dataclass(frozen=True)
class FeatureStandardizer:
    """Train-split-only centering and scaling for heterogeneous frozen features."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("mean and scale must be aligned and non-empty")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("standardizer mean must be finite")
        if not np.all(np.isfinite(self.scale)) or any(
            value <= 0.0 for value in self.scale
        ):
            raise ValueError("standardizer scale must be finite and positive")

    @classmethod
    def fit(cls, features: np.ndarray, *, minimum_scale: float = 1e-6) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("features must have shape [N >= 2, D >= 1]")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        if minimum_scale <= 0.0 or not np.isfinite(minimum_scale):
            raise ValueError("minimum_scale must be finite and positive")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale < minimum_scale, 1.0, scale)
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("features differ from the fitted standardizer dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "fit_scope": "prototype training split only",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureStandardizer":
        return cls(
            mean=tuple(float(item) for item in value["mean"]),
            scale=tuple(float(item) for item in value["scale"]),
        )


def label_effect_outcome(
    *,
    opened: bool,
    target_in_resolved_location: bool,
    target_pixels: Sequence[int],
    minimum_target_pixels: int,
) -> EffectOutcome:
    """Evaluator-only label for one physical information-action transition."""

    pixels = [int(value) for value in target_pixels]
    if not pixels or any(value < 0 for value in pixels):
        raise ValueError("target_pixels must be non-negative and non-empty")
    if minimum_target_pixels < 1:
        raise ValueError("minimum_target_pixels must be positive")
    if opened and target_in_resolved_location and max(pixels) >= minimum_target_pixels:
        return EffectOutcome.REVEALED
    if opened and not target_in_resolved_location:
        return EffectOutcome.EMPTY
    return EffectOutcome.FAILED


def label_observable_information_outcome(
    *,
    full_executor: bool,
    opened: bool,
    return_complete: bool,
    target_pixels: Sequence[int],
    minimum_target_pixels: int,
) -> EffectOutcome:
    """Legacy final-frame evaluator label kept for artifact reproduction.

    New temporal datasets must use :func:`label_temporal_information_outcome`;
    otherwise a final occlusion can erase evidence acquired earlier.
    """

    pixels = [int(value) for value in target_pixels]
    if not pixels or any(value < 0 for value in pixels):
        raise ValueError("target_pixels must be non-negative and non-empty")
    if minimum_target_pixels < 1:
        raise ValueError("minimum_target_pixels must be positive")
    if not full_executor:
        return EffectOutcome.FAILED
    if max(pixels) >= minimum_target_pixels:
        return EffectOutcome.REVEALED
    if opened and return_complete:
        return EffectOutcome.EMPTY
    return EffectOutcome.FAILED


def label_temporal_information_outcome(
    *,
    full_executor: bool,
    opened: bool,
    return_complete: bool,
    target_pixel_history: Sequence[Sequence[int]],
    minimum_target_pixels: int,
    empty_coverage_certified: bool,
) -> EffectOutcome:
    """Label information acquired anywhere along a public option history.

    A final frame cannot erase evidence acquired earlier. Conversely, absence
    of target pixels is not positive ``EMPTY`` evidence unless the searched
    region's visibility has an independently declared coverage certificate.
    """

    history = [[int(value) for value in point] for point in target_pixel_history]
    if not history or any(not point for point in history):
        raise ValueError("target_pixel_history must contain non-empty view counts")
    if any(value < 0 for point in history for value in point):
        raise ValueError("target pixel counts must be non-negative")
    if minimum_target_pixels < 1:
        raise ValueError("minimum_target_pixels must be positive")
    if not full_executor:
        return EffectOutcome.FAILED
    if max(value for point in history for value in point) >= minimum_target_pixels:
        return EffectOutcome.REVEALED
    if opened and return_complete and empty_coverage_certified:
        return EffectOutcome.EMPTY
    return EffectOutcome.FAILED


@dataclasses.dataclass(frozen=True)
class BalancedRidgeMulticlass:
    """One-vs-rest ridge heads with equal positive/negative class mass.

    Scores are converted to normalized evidence only for conformal set
    construction.  They are not advertised as calibrated transition
    probabilities; ex-ante probabilities come from the physical effect
    registry instead.
    """

    labels: tuple[str, ...]
    heads: tuple[BalancedRidgeBinary, ...]

    def __post_init__(self) -> None:
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("at least two unique labels are required")
        if len(self.heads) != len(self.labels):
            raise ValueError("one binary head is required per label")
        for label, head in zip(self.labels, self.heads, strict=True):
            if head.positive_label != label or head.negative_label != f"NOT_{label}":
                raise ValueError("binary head labels do not match multiclass labels")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        *,
        regularization: float,
    ) -> "BalancedRidgeMulticlass":
        truth = np.asarray(list(labels), dtype=str)
        ordered = tuple(sorted(set(truth.tolist())))
        if len(ordered) < 2:
            raise ValueError("at least two observed classes are required")
        heads = []
        for label in ordered:
            binary = np.where(truth == label, label, f"NOT_{label}")
            heads.append(
                BalancedRidgeBinary.fit(
                    features,
                    binary,
                    negative_label=f"NOT_{label}",
                    positive_label=label,
                    regularization=regularization,
                )
            )
        return cls(ordered, tuple(heads))

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.stack([head.score(features) for head in self.heads], axis=1)

    def evidence(self, features: np.ndarray) -> list[dict[str, float]]:
        scores = self.scores(features)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return [
            {label: float(row[index]) for index, label in enumerate(self.labels)}
            for row in weights
        ]

    def predict(self, features: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.scores(features), axis=1)
        return np.asarray([self.labels[index] for index in indices])

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "heads": [head.to_dict() for head in self.heads],
            "evidence_note": "softmax-normalized discriminative scores; not calibrated probabilities",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BalancedRidgeMulticlass":
        return cls(
            labels=tuple(str(label) for label in value["labels"]),
            heads=tuple(
                BalancedRidgeBinary.from_dict(head) for head in value["heads"]
            ),
        )


def transition_feature_block(
    before: np.ndarray, after: np.ndarray, block: str
) -> np.ndarray:
    """Build an explicitly history-dependent feature without simulator state."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] < 1:
        raise ValueError("before and after features must have the same non-empty [N, D] shape")
    if block == "after":
        return after
    if block == "delta":
        return after - before
    if block == "history":
        return np.concatenate([before, after, after - before], axis=1)
    if block.startswith("spatial_"):
        if before.shape[1] <= 8192:
            raise ValueError("spatial blocks require global-plus-spatial v2 features")
        spatial_before = before[:, 8192:]
        spatial_after = after[:, 8192:]
        if block == "spatial_after":
            return spatial_after
        if block == "spatial_delta":
            return spatial_after - spatial_before
        if block == "spatial_history":
            return np.concatenate(
                [spatial_before, spatial_after, spatial_after - spatial_before],
                axis=1,
            )
        raise ValueError(f"unknown transition feature block {block!r}")
    if before.shape[1] != 8192:
        raise ValueError("legacy prompt blocks require 8192-D global features")
    prompt = slice(2048, 4096)
    if block == "after_prompt":
        return after[:, prompt]
    if block == "prompt_delta":
        return after[:, prompt] - before[:, prompt]
    if block == "prompt_history":
        return np.concatenate(
            [before[:, prompt], after[:, prompt], after[:, prompt] - before[:, prompt]],
            axis=1,
        )
    if block == "all_delta":
        return after - before
    if block == "all_history":
        return np.concatenate([before, after, after - before], axis=1)
    raise ValueError(f"unknown transition feature block {block!r}")


def temporal_history_feature_block(
    history: np.ndarray,
    robot_state_history: np.ndarray,
    block: str,
) -> np.ndarray:
    """Summarize the frozen six-point visual/proprioceptive option history."""

    visual = np.asarray(history, dtype=np.float64)
    robot = np.asarray(robot_state_history, dtype=np.float64)
    if visual.ndim != 3 or visual.shape[0] < 1 or visual.shape[1] != 6:
        raise ValueError("history must have shape [N, 6, D]")
    if robot.shape != (visual.shape[0], 6, 8):
        raise ValueError("robot_state_history must have shape [N, 6, 8]")
    if not np.all(np.isfinite(visual)) or not np.all(np.isfinite(robot)):
        raise ValueError("temporal history features must be finite")

    if block == "no_history":
        return visual[:, -1]
    if block == "visual_only_history":
        selected = visual
    elif block == "global_visual_history":
        if visual.shape[2] < 8192:
            raise ValueError("global visual history requires at least 8192 features")
        selected = visual[:, :, :8192]
    elif block == "temporal_history":
        selected = visual
    else:
        raise ValueError(f"unknown temporal feature block {block!r}")

    visual_summary = np.concatenate(
        [
            selected[:, 0],
            selected[:, -1],
            selected[:, -1] - selected[:, 0],
            np.mean(selected, axis=1),
            np.std(selected, axis=1),
            np.max(np.abs(np.diff(selected, axis=1)), axis=1),
        ],
        axis=1,
    )
    if block != "temporal_history":
        return visual_summary

    robot_summary = np.concatenate(
        [
            robot[:, 0],
            robot[:, -1],
            robot[:, -1] - robot[:, 0],
            np.mean(robot, axis=1),
            np.std(robot, axis=1),
            np.ptp(robot, axis=1),
            np.diff(robot, axis=1).reshape(robot.shape[0], -1),
        ],
        axis=1,
    )
    return np.concatenate([visual_summary, robot_summary], axis=1)


@dataclasses.dataclass(frozen=True)
class ActionOutcomePrediction:
    evidence: dict[str, float]
    prediction_set: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ActionOutcomePredictor:
    input_dimension: int
    block: str
    critic: BalancedRidgeMulticlass
    conformal: MondrianSemanticConformalCalibrator
    standardizer: FeatureStandardizer | None = None

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "ActionOutcomePredictor":
        return cls(
            input_dimension=int(artifact.get("input_feature_dimension", 8192)),
            block=str(artifact["model_selection"]["selected"]["block"]),
            critic=BalancedRidgeMulticlass.from_dict(artifact["critic"]),
            conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["conformal"]
            ),
            standardizer=(
                FeatureStandardizer.from_dict(artifact["feature_standardizer"])
                if artifact.get("feature_standardizer") is not None
                else None
            ),
        )

    def predict(
        self, before_prefix: np.ndarray, after_prefix: np.ndarray
    ) -> ActionOutcomePrediction:
        before = np.asarray(before_prefix, dtype=np.float64)
        after = np.asarray(after_prefix, dtype=np.float64)
        expected = (self.input_dimension,)
        if before.shape != expected or after.shape != expected:
            raise ValueError(
                f"before/after prefix features must both have shape {expected}"
            )
        values = transition_feature_block(before[None, :], after[None, :], self.block)
        if self.standardizer is not None:
            values = self.standardizer.transform(values)
        evidence = self.critic.evidence(values)[0]
        return ActionOutcomePrediction(
            evidence=evidence,
            prediction_set=self.conformal.predict(evidence),
        )

    def predict_history(
        self,
        history_features: np.ndarray,
        robot_state_history: np.ndarray,
    ) -> ActionOutcomePrediction:
        """Predict from the v5 policy-visible history and executed option."""

        history = np.asarray(history_features, dtype=np.float64)
        if history.shape != (6, self.input_dimension):
            raise ValueError(
                "history_features must have shape "
                f"{(6, self.input_dimension)}, got {history.shape}"
            )
        robot = np.asarray(robot_state_history, dtype=np.float64)
        if robot.shape != (6, 8):
            raise ValueError("robot_state_history must have shape (6, 8)")
        values = temporal_history_feature_block(
            history[None, :], robot[None, :], self.block
        )
        if self.standardizer is not None:
            values = self.standardizer.transform(values)
        evidence = self.critic.evidence(values)[0]
        return ActionOutcomePrediction(
            evidence=evidence,
            prediction_set=self.conformal.predict(evidence),
        )


@dataclasses.dataclass(frozen=True)
class HierarchicalActionOutcomePredictor:
    """First recognize physical completion, then prompt-relevant content."""

    input_dimension: int
    block: str
    standardizer: FeatureStandardizer
    effect_critic: BalancedRidgeMulticlass
    effect_conformal: MondrianSemanticConformalCalibrator
    content_critic: BalancedRidgeMulticlass
    content_conformal: MondrianSemanticConformalCalibrator

    @classmethod
    def from_artifact(
        cls, artifact: dict[str, Any]
    ) -> "HierarchicalActionOutcomePredictor":
        return cls(
            input_dimension=int(artifact["input_feature_dimension"]),
            block=str(artifact["model_selection"]["block"]),
            standardizer=FeatureStandardizer.from_dict(
                artifact["feature_standardizer"]
            ),
            effect_critic=BalancedRidgeMulticlass.from_dict(
                artifact["effect_head"]["critic"]
            ),
            effect_conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["effect_head"]["conformal"]
            ),
            content_critic=BalancedRidgeMulticlass.from_dict(
                artifact["content_head"]["critic"]
            ),
            content_conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["content_head"]["conformal"]
            ),
        )

    def predict_history(
        self,
        history_features: np.ndarray,
        robot_state_history: np.ndarray,
    ) -> ActionOutcomePrediction:
        history = np.asarray(history_features, dtype=np.float64)
        robot = np.asarray(robot_state_history, dtype=np.float64)
        if history.shape != (6, self.input_dimension):
            raise ValueError(
                f"history_features must have shape {(6, self.input_dimension)}"
            )
        if robot.shape != (6, 8):
            raise ValueError("robot_state_history must have shape (6, 8)")
        values = temporal_history_feature_block(
            history[None, :], robot[None, :], self.block
        )
        values = self.standardizer.transform(values)
        effect_evidence = self.effect_critic.evidence(values)[0]
        effect_set = self.effect_conformal.predict(effect_evidence)
        content_evidence = self.content_critic.evidence(values)[0]
        content_set = self.content_conformal.predict(content_evidence)
        outcomes = []
        if EffectOutcome.FAILED.value in effect_set:
            outcomes.append(EffectOutcome.FAILED.value)
        if "COMPLETED" in effect_set:
            outcomes.extend(
                label
                for label in content_set
                if label in {EffectOutcome.REVEALED.value, EffectOutcome.EMPTY.value}
            )
        ordered = tuple(
            label
            for label in (
                EffectOutcome.FAILED.value,
                EffectOutcome.REVEALED.value,
                EffectOutcome.EMPTY.value,
            )
            if label in outcomes
        )
        if not ordered:
            # A malformed or distribution-shifted pair of sets must defer.
            ordered = tuple(item.value for item in EffectOutcome)
        evidence = {
            f"effect/{key}": value for key, value in effect_evidence.items()
        }
        evidence.update(
            {f"content/{key}": value for key, value in content_evidence.items()}
        )
        return ActionOutcomePrediction(evidence=evidence, prediction_set=ordered)
