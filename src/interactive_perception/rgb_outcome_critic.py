"""All-public-RGB T01 OPEN_AND_OBSERVE outcome inference.

The critic consumes exactly six stock policy observations. Model checkpoints
are learned offline with evaluator-only labels, but inference has no access to
segmentation, drawer joints, object poses, or task predicates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .policy_client import ObservationPacket
from .semantic_conformal import MondrianSemanticConformalCalibrator


def build_rgb_evidence_cnn(torch: Any) -> Any:
    """Build the architecture used by the frozen v10/v11 RGB checkpoints."""

    nn = torch.nn

    class RGBEvidenceCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.SiLU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.SiLU(),
                nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.SiLU(),
                nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 1),
            )

        def forward(self, values: Any) -> Any:
            return self.classifier(self.features(values)).squeeze(1)

    return RGBEvidenceCNN()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary_evidence(
    logit: float, positive_label: str, negative_label: str
) -> dict[str, float]:
    if logit >= 0:
        probability = 1.0 / (1.0 + float(np.exp(-logit)))
    else:
        exponential = float(np.exp(logit))
        probability = exponential / (1.0 + exponential)
    return {positive_label: probability, negative_label: 1.0 - probability}


def resolve_v11_cascade(
    agentview_target_set: tuple[str, ...],
    wrist_target_set: tuple[str, ...],
    coverage_set: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Apply the preregistered v11 hierarchy without selecting from ambiguity."""

    if agentview_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "agentview"
    elif wrist_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "wrist_positive_rescue"
    elif agentview_target_set == ("NOT_REVEALED",):
        target_set = ("NOT_REVEALED",)
        target_source = "agentview_negative"
    else:
        target_set = ("REVEALED", "NOT_REVEALED")
        target_source = "ambiguous"

    if target_set == ("REVEALED",):
        return ("REVEALED",), target_source
    if target_set == ("NOT_REVEALED",):
        values: list[str] = []
        if "FAILED" in coverage_set:
            values.append("FAILED")
        if "COMPLETED" in coverage_set:
            values.append("EMPTY")
        return tuple(values), target_source
    return ("FAILED", "REVEALED", "EMPTY"), target_source


def resolve_v12_cascade(
    agentview_target_set: tuple[str, ...],
    wrist_target_set: tuple[str, ...],
    coverage_set: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Compose camera evidence without resolving cross-camera conflict by fiat.

    A singleton agentview reveal remains sufficient. A wrist reveal can rescue
    ambiguous agentview evidence, but it cannot overwrite a singleton
    agentview NOT_REVEALED decision. That disagreement is retained and mapped
    through the coverage set, so the controller SAFE_STOPs instead of emitting
    a false singleton REVEALED or EMPTY.
    """

    if agentview_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "agentview"
    elif agentview_target_set == ("NOT_REVEALED",):
        if wrist_target_set == ("NOT_REVEALED",):
            target_set = ("NOT_REVEALED",)
            target_source = "camera_agreement_negative"
        else:
            target_set = ("REVEALED", "NOT_REVEALED")
            target_source = "camera_conflict"
    elif wrist_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "wrist_resolves_agentview_ambiguity"
    else:
        target_set = ("REVEALED", "NOT_REVEALED")
        target_source = "target_ambiguous"

    compatible: list[str] = []
    if "REVEALED" in target_set:
        compatible.append("REVEALED")
    if "NOT_REVEALED" in target_set:
        if "FAILED" in coverage_set:
            compatible.append("FAILED")
        if "COMPLETED" in coverage_set:
            compatible.append("EMPTY")
    canonical = tuple(
        label
        for label in ("FAILED", "REVEALED", "EMPTY")
        if label in compatible
    )
    return canonical, target_source


def resolve_v12b_cascade(
    agentview_target_set: tuple[str, ...],
    wrist_target_set: tuple[str, ...],
    coverage_set: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Preserve only a *singleton* positive wrist/agentview disagreement.

    The wrist head is registered as a positive-only auxiliary sensor. Its
    multi-label output is abstention, not counterevidence against a singleton
    agentview negative. A singleton wrist positive, however, is retained as a
    real camera conflict and cannot be silently discarded.
    """

    if agentview_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "agentview"
    elif agentview_target_set == ("NOT_REVEALED",):
        if wrist_target_set == ("REVEALED",):
            target_set = ("REVEALED", "NOT_REVEALED")
            target_source = "singleton_camera_conflict"
        else:
            target_set = ("NOT_REVEALED",)
            target_source = "agentview_negative_no_singleton_wrist_counterevidence"
    elif wrist_target_set == ("REVEALED",):
        target_set = ("REVEALED",)
        target_source = "wrist_resolves_agentview_ambiguity"
    else:
        target_set = ("REVEALED", "NOT_REVEALED")
        target_source = "target_ambiguous"

    compatible: list[str] = []
    if "REVEALED" in target_set:
        compatible.append("REVEALED")
    if "NOT_REVEALED" in target_set:
        if "FAILED" in coverage_set:
            compatible.append("FAILED")
        if "COMPLETED" in coverage_set:
            compatible.append("EMPTY")
    return (
        tuple(
            label
            for label in ("FAILED", "REVEALED", "EMPTY")
            if label in compatible
        ),
        target_source,
    )


def resolve_v13_complementary_cascade(
    agentview_target_set: tuple[str, ...],
    wrist_target_set: tuple[str, ...],
    coverage_set: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Fuse cameras as complementary fields of view.

    ``NOT_REVEALED`` is local to a camera: it says that camera supplied no
    positive target evidence.  It is therefore not counterevidence against a
    singleton ``REVEALED`` from the other public camera.  A local EMPTY result
    is emitted only when *both* target heads are singleton-negative and the
    independent search-coverage head is singleton-completed.  Any unresolved
    target or coverage set remains unresolved for the controller.

    This is a development composition over the frozen per-camera heads.  It
    does not change or retroactively relabel the frozen v12b artifact.
    """

    agent_positive = agentview_target_set == ("REVEALED",)
    wrist_positive = wrist_target_set == ("REVEALED",)
    if agent_positive or wrist_positive:
        if agent_positive and wrist_positive:
            source = "complementary_camera_agreement_positive"
        elif agent_positive:
            source = "agentview_positive_evidence"
        else:
            source = "wrist_positive_evidence"
        return ("REVEALED",), source

    cameras_negative = (
        agentview_target_set == ("NOT_REVEALED",)
        and wrist_target_set == ("NOT_REVEALED",)
    )
    if cameras_negative:
        compatible: list[str] = []
        if "FAILED" in coverage_set:
            compatible.append("FAILED")
        if "COMPLETED" in coverage_set:
            compatible.append("EMPTY")
        return tuple(compatible), "complementary_camera_agreement_negative"

    # At least one target head abstained and neither supplied singleton
    # positive evidence.  Preserve every outcome compatible with the coverage
    # set instead of selecting the convenient branch.
    compatible = ["REVEALED"]
    if "FAILED" in coverage_set:
        compatible.append("FAILED")
    if "COMPLETED" in coverage_set:
        compatible.append("EMPTY")
    return (
        tuple(
            label
            for label in ("FAILED", "REVEALED", "EMPTY")
            if label in compatible
        ),
        "complementary_target_ambiguous",
    )


@dataclass(frozen=True)
class RGBOutcomePrediction:
    prediction_set: tuple[str, ...]
    target_source: str
    agentview_target_set: tuple[str, ...]
    wrist_target_set: tuple[str, ...]
    coverage_set: tuple[str, ...]
    maximum_logits: dict[str, float]
    frame_logits: dict[str, tuple[float, ...]]
    composite_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_set": list(self.prediction_set),
            "target_source": self.target_source,
            "agentview_target_set": list(self.agentview_target_set),
            "wrist_target_set": list(self.wrist_target_set),
            "coverage_set": list(self.coverage_set),
            "maximum_logits": dict(self.maximum_logits),
            "frame_logits": {
                key: list(values) for key, values in self.frame_logits.items()
            },
            "composite_sha256": self.composite_sha256,
            "online_oracle_inputs": [],
        }


class _RGBHead:
    def __init__(
        self,
        *,
        artifact_path: Path,
        checkpoint_path: Path,
        camera: str,
        positive_label: str,
        negative_label: str,
        torch: Any,
    ) -> None:
        artifact = json.loads(artifact_path.read_text())
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        self.model = build_rgb_evidence_cnn(torch)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.calibrator = MondrianSemanticConformalCalibrator.from_dict(
            artifact["conformal"]
        )
        self.image_size = int(artifact["model"]["image_size"])
        self.camera = camera
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.torch = torch

    def _array(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        value = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        width, height = value.size
        if self.camera == "agentview":
            value = value.crop(
                (
                    int(0.36 * width),
                    int(0.22 * height),
                    width,
                    int(0.88 * height),
                )
            )
        value = value.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(value, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return np.transpose(array, (2, 0, 1)).copy()

    def predict(
        self, images: Sequence[np.ndarray]
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if len(images) != 6:
            raise ValueError("the RGB outcome protocol requires exactly six frames")
        batch = np.stack([self._array(image) for image in images])
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(batch)).cpu().numpy()
        values = tuple(float(value) for value in logits)
        maximum = max(values)
        prediction = self.calibrator.predict(
            _binary_evidence(maximum, self.positive_label, self.negative_label)
        )
        return tuple(prediction), values


class V11PublicRGBOutcomeCritic:
    """Frozen hierarchical outcome critic for the T01 six-frame protocol."""

    def __init__(
        self, composite_path: str | Path, *, root: str | Path | None = None
    ) -> None:
        import torch

        self.composite_path = Path(composite_path).resolve()
        self.root = (
            Path(root).resolve()
            if root is not None
            else self.composite_path.parents[2]
        )
        composite = json.loads(self.composite_path.read_text())
        if composite.get("online_oracle_inputs"):
            raise ValueError("RGB outcome composite declares privileged online inputs")

        def resolve(name: str) -> Path:
            reference = composite["dependencies"][name]
            path = self.root / reference["path"]
            if _digest(path) != reference["sha256"]:
                raise ValueError(f"frozen RGB outcome dependency changed: {path}")
            return path

        self.composite_sha256 = _digest(self.composite_path)
        self.agentview_target = _RGBHead(
            artifact_path=resolve("agentview_target_artifact"),
            checkpoint_path=resolve("agentview_target_checkpoint"),
            camera="agentview",
            positive_label="REVEALED",
            negative_label="NOT_REVEALED",
            torch=torch,
        )
        self.wrist_target = _RGBHead(
            artifact_path=resolve("wrist_target_artifact"),
            checkpoint_path=resolve("wrist_target_checkpoint"),
            camera="wrist",
            positive_label="REVEALED",
            negative_label="NOT_REVEALED",
            torch=torch,
        )
        self.coverage = _RGBHead(
            artifact_path=resolve("coverage_artifact"),
            checkpoint_path=resolve("coverage_checkpoint"),
            camera="agentview",
            positive_label="COMPLETED",
            negative_label="FAILED",
            torch=torch,
        )

    def predict(
        self, history: Sequence[ObservationPacket]
    ) -> RGBOutcomePrediction:
        if len(history) != 6:
            raise ValueError("the RGB outcome protocol requires exactly six observations")
        agent_images = [packet.image for packet in history]
        wrist_images = [packet.wrist_image for packet in history]
        agent_set, agent_logits = self.agentview_target.predict(agent_images)
        wrist_set, wrist_logits = self.wrist_target.predict(wrist_images)
        coverage_set, coverage_logits = self.coverage.predict(agent_images)
        outcome, source = resolve_v11_cascade(
            agent_set, wrist_set, coverage_set
        )
        return RGBOutcomePrediction(
            prediction_set=outcome,
            target_source=source,
            agentview_target_set=agent_set,
            wrist_target_set=wrist_set,
            coverage_set=coverage_set,
            maximum_logits={
                "agentview_target": max(agent_logits),
                "wrist_target": max(wrist_logits),
                "agentview_coverage": max(coverage_logits),
            },
            frame_logits={
                "agentview_target": agent_logits,
                "wrist_target": wrist_logits,
                "agentview_coverage": coverage_logits,
            },
            composite_sha256=self.composite_sha256,
        )


class V12PublicRGBOutcomeCritic(V11PublicRGBOutcomeCritic):
    """Conflict-preserving v12 composition over the frozen public-RGB heads."""

    def predict(
        self, history: Sequence[ObservationPacket]
    ) -> RGBOutcomePrediction:
        if len(history) != 6:
            raise ValueError("the RGB outcome protocol requires exactly six observations")
        agent_images = [packet.image for packet in history]
        wrist_images = [packet.wrist_image for packet in history]
        agent_set, agent_logits = self.agentview_target.predict(agent_images)
        wrist_set, wrist_logits = self.wrist_target.predict(wrist_images)
        coverage_set, coverage_logits = self.coverage.predict(agent_images)
        outcome, source = resolve_v12_cascade(
            agent_set, wrist_set, coverage_set
        )
        return RGBOutcomePrediction(
            prediction_set=outcome,
            target_source=source,
            agentview_target_set=agent_set,
            wrist_target_set=wrist_set,
            coverage_set=coverage_set,
            maximum_logits={
                "agentview_target": max(agent_logits),
                "wrist_target": max(wrist_logits),
                "agentview_coverage": max(coverage_logits),
            },
            frame_logits={
                "agentview_target": agent_logits,
                "wrist_target": wrist_logits,
                "agentview_coverage": coverage_logits,
            },
            composite_sha256=self.composite_sha256,
        )


class V12bPublicRGBOutcomeCritic(V11PublicRGBOutcomeCritic):
    """V12b singleton-conflict composition over the frozen RGB heads."""

    def predict(
        self, history: Sequence[ObservationPacket]
    ) -> RGBOutcomePrediction:
        if len(history) != 6:
            raise ValueError("the RGB outcome protocol requires exactly six observations")
        agent_images = [packet.image for packet in history]
        wrist_images = [packet.wrist_image for packet in history]
        agent_set, agent_logits = self.agentview_target.predict(agent_images)
        wrist_set, wrist_logits = self.wrist_target.predict(wrist_images)
        coverage_set, coverage_logits = self.coverage.predict(agent_images)
        outcome, source = resolve_v12b_cascade(
            agent_set, wrist_set, coverage_set
        )
        return RGBOutcomePrediction(
            prediction_set=outcome,
            target_source=source,
            agentview_target_set=agent_set,
            wrist_target_set=wrist_set,
            coverage_set=coverage_set,
            maximum_logits={
                "agentview_target": max(agent_logits),
                "wrist_target": max(wrist_logits),
                "agentview_coverage": max(coverage_logits),
            },
            frame_logits={
                "agentview_target": agent_logits,
                "wrist_target": wrist_logits,
                "agentview_coverage": coverage_logits,
            },
            composite_sha256=self.composite_sha256,
        )


class V13ComplementaryPublicRGBOutcomeCritic(V11PublicRGBOutcomeCritic):
    """Development complementary-view composition over frozen RGB heads."""

    def predict(
        self, history: Sequence[ObservationPacket]
    ) -> RGBOutcomePrediction:
        if len(history) != 6:
            raise ValueError("the RGB outcome protocol requires exactly six observations")
        agent_images = [packet.image for packet in history]
        wrist_images = [packet.wrist_image for packet in history]
        agent_set, agent_logits = self.agentview_target.predict(agent_images)
        wrist_set, wrist_logits = self.wrist_target.predict(wrist_images)
        coverage_set, coverage_logits = self.coverage.predict(agent_images)
        outcome, source = resolve_v13_complementary_cascade(
            agent_set, wrist_set, coverage_set
        )
        return RGBOutcomePrediction(
            prediction_set=outcome,
            target_source=source,
            agentview_target_set=agent_set,
            wrist_target_set=wrist_set,
            coverage_set=coverage_set,
            maximum_logits={
                "agentview_target": max(agent_logits),
                "wrist_target": max(wrist_logits),
                "agentview_coverage": max(coverage_logits),
            },
            frame_logits={
                "agentview_target": agent_logits,
                "wrist_target": wrist_logits,
                "agentview_coverage": coverage_logits,
            },
            composite_sha256=self.composite_sha256,
        )
