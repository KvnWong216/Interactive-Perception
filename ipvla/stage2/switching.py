"""Two-Stage Confidence-Gated Switching Controller (Opus 5.5 §2.6–§3.3).

Stage 1 (Wake Gate):
    WAKE(t) <=> (r_t > b + kappa * phi_t) AND (t - t_v >= T_lock)
    (with initial wake on step 0 and optional safety max_idle_steps heartbeat).

Stage 2 (Relative Confidence Commit Gate):
    COMMIT(u') <=> (not Eq(u', u_k))
                   AND (c(u') >= c_min)
                   AND (c(u') - c(u_k) >= delta_0 + kappa_c * phi_t)
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from ipvla.stage2.cerebellum import CerebellumConfig, ContinuousCerebellumInertia, InertiaSnapshot
from ipvla.stage2.memory import BufferedFrame, InspectKeyframeMemoryBuffer, MemoryConfig


_VERB_SYNONYMS = {
    "grasp": "pick",
    "grab": "pick",
    "take": "pick",
    "lift": "pick",
    "put": "place",
    "set": "place",
    "position": "place",
    "drop": "place",
    "seat": "place",
    "check": "inspect",
    "examine": "inspect",
    "look": "inspect",
    "rotate": "inspect",
    "turn": "inspect",
    "scan": "search",
    "find": "search",
    "locate": "search",
    "press": "click",
    "push": "click",
    "tap": "click",
}
_STOPWORDS = {"the", "a", "an", "to", "on", "onto", "in", "into", "at", "of", "for", "up", "it", "and"}


def normalize_subtask(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        out.append(_VERB_SYNONYMS.get(tok, tok))
    return " ".join(out)


def subtasks_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return normalize_subtask(a) == normalize_subtask(b)


@dataclass
class SwitchingConfig:
    wake_bias: float = 0.12
    wake_inertia_kappa: float = 0.35
    t_lock_steps: int = 16
    max_idle_steps: int = 96
    fixed_period_steps: int = 32
    c_min: float = 0.45
    commit_delta0: float = 0.05
    commit_kappa_c: float = 0.20


@dataclass(frozen=True)
class WakeDecision:
    should_wake: bool
    reason: str
    step: int
    novelty_r: float
    phi: float
    threshold: float
    steps_since_vlm: int


@dataclass(frozen=True)
class CommitDecision:
    committed: bool
    reason: str
    active_subtask: str
    active_confidence: float
    candidate_subtask: str
    candidate_confidence: float
    margin_required: float
    pinned_inspect_frame: bool


class TwoStageSwitchingController:
    """Coordinates continuous cerebellum inertia, inspect memory, wake gate, and commit gate."""

    def __init__(
        self,
        cerebellum_cfg: CerebellumConfig | None = None,
        memory_cfg: MemoryConfig | None = None,
        switching_cfg: SwitchingConfig | None = None,
    ) -> None:
        self.cerebellum_cfg = cerebellum_cfg or CerebellumConfig()
        self.memory_cfg = memory_cfg or MemoryConfig()
        self.switching_cfg = switching_cfg or SwitchingConfig()
        self.cerebellum = ContinuousCerebellumInertia(self.cerebellum_cfg)
        self.memory = InspectKeyframeMemoryBuffer(self.memory_cfg)
        self.reset()

    def reset(self) -> None:
        self.cerebellum.reset()
        self.memory.reset()
        self.active_subtask: str = ""
        self.active_confidence: float = 0.0
        self._last_vlm_step: Optional[int] = None
        self.total_vlm_calls: int = 0
        self.total_commits: int = 0

    def observe_and_check_wake(
        self,
        step: int,
        image,
        joint_state,
        commanded_action_chunk=None,
    ) -> tuple[WakeDecision, InertiaSnapshot, BufferedFrame]:
        """Push observation, update cerebellum inertia, and evaluate Stage-1 wake gate."""
        snap = self.cerebellum.update(step, joint_state, commanded_action_chunk)
        frame = self.memory.push(step, image, active_subtask=self.active_subtask)
        r_t = self.memory.novelty_since_last_vlm_call(frame.image)

        cfg = self.switching_cfg
        steps_since_vlm = (
            int(step) - self._last_vlm_step if self._last_vlm_step is not None else 10**9
        )

        # Initial subtask bootstrap on first observation.
        if not self.active_subtask:
            return (
                WakeDecision(
                    should_wake=True,
                    reason="initial_bootstrap",
                    step=int(step),
                    novelty_r=r_t,
                    phi=snap.phi,
                    threshold=0.0,
                    steps_since_vlm=steps_since_vlm,
                ),
                snap,
                frame,
            )

        # Ablation arm: fixed-period VLM polling.
        if self.cerebellum_cfg.mode == "fixed_period":
            due = steps_since_vlm >= max(int(cfg.fixed_period_steps), 1)
            return (
                WakeDecision(
                    should_wake=due,
                    reason="fixed_period_due" if due else "fixed_period_wait",
                    step=int(step),
                    novelty_r=r_t,
                    phi=snap.phi,
                    threshold=0.0,
                    steps_since_vlm=steps_since_vlm,
                ),
                snap,
                frame,
            )

        # Cooldown T_lock always enforced after a VLM call.
        threshold = cfg.wake_bias + cfg.wake_inertia_kappa * snap.phi
        if steps_since_vlm < cfg.t_lock_steps:
            return (
                WakeDecision(
                    should_wake=False,
                    reason="t_lock_cooldown",
                    step=int(step),
                    novelty_r=r_t,
                    phi=snap.phi,
                    threshold=threshold,
                    steps_since_vlm=steps_since_vlm,
                ),
                snap,
                frame,
            )

        if r_t > threshold:
            return (
                WakeDecision(
                    should_wake=True,
                    reason="novelty_exceeded_inertia_threshold",
                    step=int(step),
                    novelty_r=r_t,
                    phi=snap.phi,
                    threshold=threshold,
                    steps_since_vlm=steps_since_vlm,
                ),
                snap,
                frame,
            )

        if cfg.max_idle_steps > 0 and steps_since_vlm >= cfg.max_idle_steps:
            return (
                WakeDecision(
                    should_wake=True,
                    reason="max_idle_heartbeat",
                    step=int(step),
                    novelty_r=r_t,
                    phi=snap.phi,
                    threshold=threshold,
                    steps_since_vlm=steps_since_vlm,
                ),
                snap,
                frame,
            )

        return (
            WakeDecision(
                should_wake=False,
                reason="below_wake_threshold",
                step=int(step),
                novelty_r=r_t,
                phi=snap.phi,
                threshold=threshold,
                steps_since_vlm=steps_since_vlm,
            ),
            snap,
            frame,
        )

    def evaluate_candidate(
        self,
        step: int,
        candidate_subtask: str,
        candidate_confidence: float,
        frame: BufferedFrame,
        snap: InertiaSnapshot,
        novelty_r: float = 0.0,
    ) -> CommitDecision:
        """Record a VLM invocation and apply Stage-2 relative confidence commit gate."""
        self._last_vlm_step = int(step)
        self.memory.mark_vlm_call(frame)
        self.total_vlm_calls += 1

        cand_text = (candidate_subtask or "").strip()
        cand_conf = float(candidate_confidence)
        cfg = self.switching_cfg
        margin_req = cfg.commit_delta0 + cfg.commit_kappa_c * snap.phi

        # Bootstrap first subtask immediately.
        if not self.active_subtask:
            self.active_subtask = cand_text
            self.active_confidence = cand_conf
            self.cerebellum.mark_commit(step)
            self.total_commits += 1
            return CommitDecision(
                committed=True,
                reason="initial_commit",
                active_subtask=self.active_subtask,
                active_confidence=self.active_confidence,
                candidate_subtask=cand_text,
                candidate_confidence=cand_conf,
                margin_required=0.0,
                pinned_inspect_frame=False,
            )

        # Equivalent phrasing keeps current subtask and refreshes confidence estimate.
        if subtasks_equivalent(cand_text, self.active_subtask):
            self.active_confidence = cand_conf
            return CommitDecision(
                committed=False,
                reason="equivalent_subtask_refresh",
                active_subtask=self.active_subtask,
                active_confidence=self.active_confidence,
                candidate_subtask=cand_text,
                candidate_confidence=cand_conf,
                margin_required=margin_req,
                pinned_inspect_frame=False,
            )

        # In fixed_period ablation, commit any non-equivalent candidate above c_min without inertia margin.
        if self.cerebellum_cfg.mode == "fixed_period":
            if cand_conf >= cfg.c_min:
                pinned = self.memory.pin_current_if_inspect_end(
                    self.active_subtask, cand_text, frame, novelty=novelty_r
                )
                self.active_subtask = cand_text
                self.active_confidence = cand_conf
                self.cerebellum.mark_commit(step)
                self.total_commits += 1
                return CommitDecision(
                    committed=True,
                    reason="fixed_period_commit",
                    active_subtask=self.active_subtask,
                    active_confidence=self.active_confidence,
                    candidate_subtask=cand_text,
                    candidate_confidence=cand_conf,
                    margin_required=0.0,
                    pinned_inspect_frame=pinned,
                )
            return CommitDecision(
                committed=False,
                reason="below_c_min",
                active_subtask=self.active_subtask,
                active_confidence=self.active_confidence,
                candidate_subtask=cand_text,
                candidate_confidence=cand_conf,
                margin_required=0.0,
                pinned_inspect_frame=False,
            )

        # Full Stage-2 relative confidence commit gate.
        if cand_conf < cfg.c_min:
            return CommitDecision(
                committed=False,
                reason="below_c_min",
                active_subtask=self.active_subtask,
                active_confidence=self.active_confidence,
                candidate_subtask=cand_text,
                candidate_confidence=cand_conf,
                margin_required=margin_req,
                pinned_inspect_frame=False,
            )

        delta = cand_conf - self.active_confidence
        if delta < margin_req:
            return CommitDecision(
                committed=False,
                reason="insufficient_confidence_margin",
                active_subtask=self.active_subtask,
                active_confidence=self.active_confidence,
                candidate_subtask=cand_text,
                candidate_confidence=cand_conf,
                margin_required=margin_req,
                pinned_inspect_frame=False,
            )

        # Commit accepted! Pin reveal frame if transitioning out of an inspect/search skill.
        pinned = self.memory.pin_current_if_inspect_end(
            self.active_subtask, cand_text, frame, novelty=novelty_r
        )
        self.active_subtask = cand_text
        self.active_confidence = cand_conf
        self.cerebellum.mark_commit(step)
        self.total_commits += 1
        return CommitDecision(
            committed=True,
            reason="relative_confidence_commit",
            active_subtask=self.active_subtask,
            active_confidence=self.active_confidence,
            candidate_subtask=cand_text,
            candidate_confidence=cand_conf,
            margin_required=margin_req,
            pinned_inspect_frame=pinned,
        )
