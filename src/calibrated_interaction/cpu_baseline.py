"""Deterministic public-input features for a no-GPU benchmark baseline.

These features are deliberately not presented as a replacement for the frozen
VLM. They provide a reproducible low-capacity baseline when the experiment's
GPU-memory budget excludes loading the VLA/VLM checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FEATURE_WIDTH = 2048


def _pad(values: np.ndarray) -> np.ndarray:
    result = np.zeros(FEATURE_WIDTH, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(flat) > FEATURE_WIDTH:
        raise ValueError("feature exceeds fixed width")
    result[: len(flat)] = flat
    return result


def image_token(path: str | Path) -> np.ndarray:
    """Encode public RGB through fixed pixels, histograms, and edge summaries."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        small_rgb = np.asarray(rgb.resize((16, 16)), dtype=np.float32) / 255.0
        gray = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
    histograms = np.concatenate(
        [
            np.histogram(small_rgb[..., channel], bins=32, range=(0.0, 1.0))[0]
            for channel in range(3)
        ]
    ).astype(np.float32)
    histograms /= max(float(histograms.sum()), 1.0)
    moments = np.asarray(
        [
            statistic(small_rgb[..., channel])
            for channel in range(3)
            for statistic in (np.mean, np.std, np.min, np.max)
        ],
        dtype=np.float32,
    )
    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    vertical_edges = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return _pad(
        np.concatenate(
            (
                gray.reshape(-1),
                small_rgb.reshape(-1),
                histograms,
                moments,
                horizontal_edges,
                vertical_edges,
            )
        )
    )


def text_token(value: str) -> np.ndarray:
    """Signed feature hashing over normalized character n-grams."""

    normalized = " ".join(value.lower().split())
    result = np.zeros(FEATURE_WIDTH, dtype=np.float32)
    grams = [normalized]
    for width in (2, 3, 4, 5):
        grams.extend(
            normalized[index : index + width]
            for index in range(len(normalized) - width + 1)
        )
    for gram in grams:
        digest = hashlib.sha256(gram.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % FEATURE_WIDTH
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign
    norm = float(np.linalg.norm(result))
    if norm:
        result /= norm
    return result


def context_tokens(
    *, prompt: str, agentview: str | Path, wrist: str | Path
) -> np.ndarray:
    agent = image_token(agentview)
    hand = image_token(wrist)
    language = text_token(prompt)
    fused = 0.4 * agent + 0.4 * hand + 0.2 * language
    return np.stack((agent, hand, language, fused)).astype(np.float32)


def candidate_token(candidate: dict[str, Any]) -> np.ndarray:
    visible = {
        key: candidate.get(key)
        for key in ("candidate_id", "primitive", "target", "reference", "purpose")
    }
    return text_token(json.dumps(visible, sort_keys=True, separators=(",", ":")))
