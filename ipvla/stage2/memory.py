"""Non-Oracle Inspect-Keyframe Visual Memory Buffer for Stage-2 Active Perception.

Prevents reveal frames (e.g., hidden backside sticker color or expiry date observed
during an `inspect` / `check` / `rotate` skill) from being evicted by a short sliding
window while the robot executes subsequent restore/grasp/place maneuvers.

Strict public/non-oracle rule:
- Pinning depends ONLY on the autonomous planner's own previously committed subtask text
  (e.g. `inspect`, `check`, `rotate`, `scan`, `search`, `lift cloth`) and causal visual
  novelty r_t — NEVER on simulator oracle states or ground-truth reveal flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import numpy as np


INSPECT_VERB_RE = re.compile(
    r"\b(inspect|check|examine|look|scan|search|rotate|turn|flip|reveal|lift|uncover|move\s+.*aside)\b",
    re.IGNORECASE,
)


def is_information_gathering_subtask(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(INSPECT_VERB_RE.search(text))


def compute_visual_novelty(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fast causal perceptual novelty r_t in [0, 1] between two RGB images (H, W, 3).

    Downsamples via simple spatial striding to a 32x32 normalized RGB grid and combines
    mean L1 luminance/color change with center-weighted focus (where manipulated objects live).
    """
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    if a.shape != b.shape or a.ndim < 2:
        return 1.0
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0

    sh = max(a.shape[0] // 32, 1)
    sw = max(a.shape[1] // 32, 1)
    da = a[::sh, ::sw]
    db = b[::sh, ::sw]
    diff = np.abs(da - db)

    global_l1 = float(np.mean(diff))
    h, w = diff.shape[0], diff.shape[1]
    ch0, ch1 = h // 4, max(3 * h // 4, h // 4 + 1)
    cw0, cw1 = w // 4, max(3 * w // 4, w // 4 + 1)
    center_l1 = float(np.mean(diff[ch0:ch1, cw0:cw1]))

    score = 0.4 * global_l1 + 0.6 * center_l1
    # Calibrate typical manipulation pixel shift (~0.02-0.25) smoothly into [0, 1].
    return float(np.clip(score * 3.5, 0.0, 1.0))


@dataclass
class BufferedFrame:
    step: int
    image: np.ndarray
    subtask_at_capture: str = ""
    pinned: bool = False
    pin_reason: str = ""


@dataclass
class MemoryConfig:
    mode: str = "inspect_keyframe_pin"  # "inspect_keyframe_pin" | "sliding_window"
    num_frames: int = 4
    max_pinned_inspect_frames: int = 2
    sliding_stride_steps: int = 16
    novelty_pin_threshold: float = 0.32
    max_history: int = 256


class InspectKeyframeMemoryBuffer:
    """Maintains causal history and constructs the 4-frame VLM planner input."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.reset()

    def reset(self) -> None:
        self._frames: list[BufferedFrame] = []
        self._pinned: list[BufferedFrame] = []
        self._last_vlm_frame: Optional[BufferedFrame] = None

    @property
    def pinned_steps(self) -> list[int]:
        return [f.step for f in self._pinned]

    def push(self, step: int, image: np.ndarray, active_subtask: str = "") -> BufferedFrame:
        entry = BufferedFrame(
            step=int(step),
            image=np.asarray(image),
            subtask_at_capture=active_subtask or "",
        )
        self._frames.append(entry)
        if len(self._frames) > self.config.max_history:
            self._frames = self._frames[-self.config.max_history :]
        return entry

    def mark_vlm_call(self, frame: BufferedFrame) -> None:
        self._last_vlm_frame = frame

    def novelty_since_last_vlm_call(self, current_image: np.ndarray) -> float:
        """Compute novelty r_t anchored at the last VLM invocation frame t_v."""
        if self._last_vlm_frame is None:
            return 1.0
        return compute_visual_novelty(self._last_vlm_frame.image, current_image)

    def pin_current_if_inspect_end(
        self,
        previous_subtask: str,
        new_subtask: str,
        current_frame: BufferedFrame,
        novelty: float = 0.0,
    ) -> bool:
        """Pin the reveal/end frame when transitioning out of an inspect/search skill."""
        if self.config.mode != "inspect_keyframe_pin":
            return False

        reason = ""
        if is_information_gathering_subtask(previous_subtask) and not is_information_gathering_subtask(new_subtask):
            reason = "inspect_skill_completion"
        elif is_information_gathering_subtask(previous_subtask) and novelty >= self.config.novelty_pin_threshold:
            reason = "inspect_high_novelty"

        if not reason:
            return False

        # Avoid duplicate pinning at the exact same step.
        if any(f.step == current_frame.step for f in self._pinned):
            return False

        pinned_copy = BufferedFrame(
            step=current_frame.step,
            image=current_frame.image,
            subtask_at_capture=previous_subtask,
            pinned=True,
            pin_reason=reason,
        )
        self._pinned.append(pinned_copy)
        max_pin = max(int(self.config.max_pinned_inspect_frames), 0)
        if max_pin == 0:
            self._pinned.clear()
            return False
        if len(self._pinned) > max_pin:
            self._pinned = self._pinned[-max_pin:]
        return True

    def select_causal_frames(self) -> list[BufferedFrame]:
        """Return `num_frames` chronologically ordered frames ending at the current frame.

        In `inspect_keyframe_pin` mode:
        - Slot 0: earliest context frame (or stride history)
        - Slots 1..k: pinned inspect reveal frames (never evicted by sliding window)
        - Last slot: current cutoff frame
        """
        n = max(int(self.config.num_frames), 1)
        if not self._frames:
            return []
        current = self._frames[-1]
        if n == 1:
            return [current]

        if self.config.mode == "sliding_window" or not self._pinned:
            return self._select_sliding(n)

        selected: dict[int, BufferedFrame] = {current.step: current}
        for pf in self._pinned[-self.config.max_pinned_inspect_frames :]:
            if len(selected) < n:
                selected[pf.step] = pf

        # Fill remaining slots from causal sliding history (preferring earliest + evenly spaced).
        sliding = self._select_sliding(n)
        for sf in sliding:
            if len(selected) >= n:
                break
            if sf.step not in selected:
                selected[sf.step] = sf

        ordered = sorted(selected.values(), key=lambda f: f.step)
        # If episode just started (< n frames), left-pad with earliest frame (matching training contract).
        while len(ordered) < n:
            ordered.insert(0, ordered[0])
        return ordered[-n:]

    def _select_sliding(self, n: int) -> list[BufferedFrame]:
        if not self._frames:
            return []
        stride = max(int(self.config.sliding_stride_steps), 1)
        current = self._frames[-1]
        targets = [max(0, current.step - stride * k) for k in reversed(range(n))]
        chosen: list[BufferedFrame] = []
        for t in targets:
            best = min(self._frames, key=lambda f: abs(f.step - t))
            chosen.append(best)
        return chosen
