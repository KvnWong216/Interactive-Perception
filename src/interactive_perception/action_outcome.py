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


def label_temporal_information_outcome_v10(
    *,
    full_executor: bool,
    target_pixel_history: Sequence[Sequence[int]],
    minimum_target_pixels: int,
    searched_region_coverage_history: Sequence[bool],
) -> EffectOutcome:
    """Evaluator-only information label with temporal evidence persistence.

    Version 9 incorrectly required the drawer to remain open *and* the arm to
    reach one exact return pose before certifying ``EMPTY``.  Those are motor
    diagnostics, not information endpoints.  Version 10 applies the same
    persistence rule to positive and negative information: once a public
    history point reveals the target, or independently certifies coverage of
    the searched region, later re-occlusion cannot erase the observation.

    ``searched_region_coverage_history`` is constructed offline with a
    seed-matched same-camera-pose counterfactual.  It is never an online input.
    """

    history = [[int(value) for value in point] for point in target_pixel_history]
    coverage = [bool(value) for value in searched_region_coverage_history]
    if not history or any(not point for point in history):
        raise ValueError("target_pixel_history must contain non-empty view counts")
    if len(coverage) != len(history):
        raise ValueError("coverage history must align with target pixel history")
    if any(value < 0 for point in history for value in point):
        raise ValueError("target pixel counts must be non-negative")
    if minimum_target_pixels < 1:
        raise ValueError("minimum_target_pixels must be positive")
    if not full_executor:
        return EffectOutcome.FAILED
    if max(value for point in history for value in point) >= minimum_target_pixels:
        return EffectOutcome.REVEALED
    if any(coverage):
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
    if block in {"visual_only_history", "visual_extrema_history"}:
        selected = visual
    elif block == "global_visual_history":
        if visual.shape[2] < 8192:
            raise ValueError("global visual history requires at least 8192 features")
        selected = visual[:, :, :8192]
    elif block == "temporal_history":
        selected = visual
    else:
        raise ValueError(f"unknown temporal feature block {block!r}")

    summary_parts = [
        selected[:, 0],
        selected[:, -1],
        selected[:, -1] - selected[:, 0],
        np.mean(selected, axis=1),
        np.std(selected, axis=1),
        np.max(np.abs(np.diff(selected, axis=1)), axis=1),
    ]
    if block == "visual_extrema_history":
        summary_parts.extend(
            [np.max(selected, axis=1), np.min(selected, axis=1)]
        )
    visual_summary = np.concatenate(summary_parts, axis=1)
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


def temporal_target_score_summary(frame_scores: Sequence[float]) -> np.ndarray:
    """Preserve six frame scores and explicit temporal-OR statistics."""

    values = np.asarray(frame_scores, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("frame_scores must contain six finite values")
    ordered = np.sort(values)
    return np.concatenate(
        [
            values,
            np.asarray(
                [
                    ordered[-1],
                    ordered[-2],
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.ptp(values)),
                ]
            ),
        ]
    )


@dataclasses.dataclass(frozen=True)
class TemporalTargetEvidencePredictor:
    """Frame-level prompt evidence followed by an episode-level temporal OR."""

    input_dimension: int
    feature_start: int
    feature_end: int
    frame_standardizer: FeatureStandardizer
    frame_probe: BalancedRidgeBinary
    episode_standardizer: FeatureStandardizer
    episode_critic: BalancedRidgeMulticlass
    episode_conformal: MondrianSemanticConformalCalibrator

    def __post_init__(self) -> None:
        if not 0 <= self.feature_start < self.feature_end <= self.input_dimension:
            raise ValueError("target-evidence feature slice is invalid")

    @classmethod
    def from_artifact(
        cls, artifact: dict[str, Any]
    ) -> "TemporalTargetEvidencePredictor":
        feature_slice = artifact["frame_feature_slice"]
        return cls(
            input_dimension=int(artifact["input_feature_dimension"]),
            feature_start=int(feature_slice["start"]),
            feature_end=int(feature_slice["end"]),
            frame_standardizer=FeatureStandardizer.from_dict(
                artifact["frame_standardizer"]
            ),
            frame_probe=BalancedRidgeBinary.from_dict(artifact["frame_probe"]),
            episode_standardizer=FeatureStandardizer.from_dict(
                artifact["episode_standardizer"]
            ),
            episode_critic=BalancedRidgeMulticlass.from_dict(
                artifact["episode_critic"]
            ),
            episode_conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["episode_conformal"]
            ),
        )

    def predict_history(self, history_features: np.ndarray) -> ActionOutcomePrediction:
        history = np.asarray(history_features, dtype=np.float64)
        if history.shape != (6, self.input_dimension):
            raise ValueError(
                f"history_features must have shape {(6, self.input_dimension)}"
            )
        frame_values = history[:, self.feature_start : self.feature_end]
        frame_values = self.frame_standardizer.transform(frame_values)
        frame_scores = self.frame_probe.score(frame_values)
        episode_values = temporal_target_score_summary(frame_scores)[None, :]
        episode_values = self.episode_standardizer.transform(episode_values)
        evidence = self.episode_critic.evidence(episode_values)[0]
        return ActionOutcomePrediction(
            evidence={
                **evidence,
                "maximum_frame_score": float(np.max(frame_scores)),
            },
            prediction_set=self.episode_conformal.predict(evidence),
        )


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

    def predict_effect_history(
        self,
        history_features: np.ndarray,
        robot_state_history: np.ndarray,
    ) -> ActionOutcomePrediction:
        """Classify whether the option produced a certifiable observation.

        This exposes the first hierarchical head for the v10 composite critic.
        Target evidence is handled by an independent per-frame RGB detector;
        only histories without singleton target evidence reach this head.
        """

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
        evidence = self.effect_critic.evidence(values)[0]
        return ActionOutcomePrediction(
            evidence=evidence,
            prediction_set=self.effect_conformal.predict(evidence),
        )


def combine_v10_outcome_sets(
    target_evidence_set: Sequence[str],
    observation_effect_set: Sequence[str],
) -> tuple[str, ...]:
    """Combine target temporal-OR and observation-effect conformal sets.

    Ambiguity in either hierarchy level is preserved as a multi-label outcome,
    which the online controller maps to SAFE_STOP.  A singleton REVEALED result
    persists even if a later motor or return phase fails.
    """

    target = tuple(target_evidence_set)
    effect = tuple(observation_effect_set)
    valid_target = {"REVEALED", "NOT_REVEALED"}
    valid_effect = {"FAILED", "COMPLETED"}
    if not target or not set(target).issubset(valid_target):
        raise ValueError("invalid target-evidence prediction set")
    if not effect or not set(effect).issubset(valid_effect):
        raise ValueError("invalid observation-effect prediction set")
    if target == ("REVEALED",):
        return (EffectOutcome.REVEALED.value,)
    outcomes: set[str] = set()
    if "REVEALED" in target:
        outcomes.add(EffectOutcome.REVEALED.value)
    if "NOT_REVEALED" in target:
        if "FAILED" in effect:
            outcomes.add(EffectOutcome.FAILED.value)
        if "COMPLETED" in effect:
            outcomes.add(EffectOutcome.EMPTY.value)
    return tuple(
        label
        for label in (
            EffectOutcome.FAILED.value,
            EffectOutcome.REVEALED.value,
            EffectOutcome.EMPTY.value,
        )
        if label in outcomes
    )
