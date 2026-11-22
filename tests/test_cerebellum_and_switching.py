"""Unit tests for Stage-2 cerebellum inertia, inspect-keyframe memory, switching & eval readiness."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ipvla.config import load_paths_config
from ipvla.stage1.confidence import RidgeTempConfidenceHead
from ipvla.stage2.cerebellum import CerebellumConfig, ContinuousCerebellumInertia
from ipvla.stage2.memory import InspectKeyframeMemoryBuffer, MemoryConfig, compute_visual_novelty
from ipvla.stage2.switching import (
    SwitchingConfig,
    TwoStageSwitchingController,
    subtasks_equivalent,
)
from eval.activearena.run_eval import check_readiness


def test_cerebellum_inertia_bounds_and_gripper_separation():
    cb = ContinuousCerebellumInertia(
        CerebellumConfig(
            mode="physical_cerebellum",
            left_gripper_idx=7,
            right_gripper_idx=15,
            horizon_steps=16,
            dwell_tau0=0.5,
        )
    )
    q0 = np.zeros(18, dtype=np.float64)
    cb.mark_commit(0)

    # Immediately at commit t = 0: dwell inertia phi_dw == 1.0 -> phi == 1.0 even at zero velocity
    s0 = cb.update(step=0, state=q0)
    assert abs(s0.phi_dw - 1.0) < 1e-6
    assert abs(s0.phi - 1.0) < 1e-6

    # After horizon expires (step >= 16) at zero motion: dwell drops to 0 and phi -> 0
    s20 = cb.update(step=20, state=q0)
    assert s20.phi_dw == 0.0
    assert s20.phi < 1e-6

    # Pure gripper closure (index 7 changes, 16 non-gripper joints stationary) activates ONLY phi_grip
    q_grip = q0.copy()
    q_grip[7] = 0.8
    s21 = cb.update(step=21, state=q_grip)
    assert s21.phi_kin == 0.0
    assert s21.phi_grip > 0.5
    assert abs(s21.phi - s21.phi_grip) < 1e-6

    # Ablation arm no_inertia_k0 forces phi == 0
    cb_k0 = ContinuousCerebellumInertia(CerebellumConfig(mode="no_inertia_k0"))
    cb_k0.mark_commit(0)
    assert cb_k0.update(step=0, state=q_grip).phi == 0.0


def test_inspect_keyframe_memory_pins_reveal_frame_against_eviction():
    mem = InspectKeyframeMemoryBuffer(
        MemoryConfig(mode="inspect_keyframe_pin", num_frames=4, max_pinned_inspect_frames=2, sliding_stride_steps=8)
    )
    # Push 60 steps; at step 12 the robot finishes "Inspect the backside of the gray block" -> "Place on red pad"
    reveal_frame = None
    for step in range(0, 64, 4):
        img = np.full((64, 64, 3), step * 3 % 255, dtype=np.uint8)
        bf = mem.push(step=step, image=img, active_subtask="Inspect the gray block" if step <= 12 else "Place on red pad")
        if step == 12:
            reveal_frame = bf
            pinned = mem.pin_current_if_inspect_end(
                previous_subtask="Inspect the backside of the gray block",
                new_subtask="Place the gray block on the red pad",
                current_frame=bf,
                novelty=0.4,
            )
            assert pinned is True

    assert reveal_frame is not None
    selected = mem.select_causal_frames()
    steps = [f.step for f in selected]
    assert len(steps) == 4
    assert steps == sorted(steps)
    assert 12 in steps, f"Reveal frame at step 12 was evicted! Got {steps}"
    assert steps[-1] == 60

    # Contrast with sliding_window ablation: step 12 IS evicted by step 60
    mem_sw = InspectKeyframeMemoryBuffer(
        MemoryConfig(mode="sliding_window", num_frames=4, sliding_stride_steps=8)
    )
    for step in range(0, 64, 4):
        img = np.full((64, 64, 3), step * 3 % 255, dtype=np.uint8)
        mem_sw.push(step=step, image=img)
    sw_steps = [f.step for f in mem_sw.select_causal_frames()]
    assert 12 not in sw_steps


def test_two_stage_wake_and_relative_confidence_commit_gate():
    ctrl = TwoStageSwitchingController(
        cerebellum_cfg=CerebellumConfig(mode="physical_cerebellum", horizon_steps=16),
        memory_cfg=MemoryConfig(mode="inspect_keyframe_pin"),
        switching_cfg=SwitchingConfig(
            wake_bias=0.10,
            wake_inertia_kappa=0.40,
            t_lock_steps=10,
            c_min=0.45,
            commit_delta0=0.05,
            commit_kappa_c=0.25,
        ),
    )
    q = np.zeros(18, dtype=np.float64)
    img0 = np.zeros((64, 64, 3), dtype=np.uint8)
    img_new = np.full((64, 64, 3), 220, dtype=np.uint8)

    # Step 0: bootstrap wake + initial commit
    w0, s0, f0 = ctrl.observe_and_check_wake(0, img0, q)
    assert w0.should_wake and w0.reason == "initial_bootstrap"
    c0 = ctrl.evaluate_candidate(0, "Inspect the gray block", 0.60, f0, s0, w0.novelty_r)
    assert c0.committed and ctrl.active_subtask == "Inspect the gray block"

    # Step 5 (< T_lock=10): blocked by T_lock even on huge visual novelty
    w5, s5, f5 = ctrl.observe_and_check_wake(5, img_new, q)
    assert not w5.should_wake and w5.reason == "t_lock_cooldown"

    # Step 20 (>= T_lock and dwell expired -> phi=0): wakes on high novelty
    w20, s20, f20 = ctrl.observe_and_check_wake(20, img_new, q)
    assert w20.should_wake and w20.novelty_r > 0.5

    # Small confidence gain (+0.02 < delta_0=0.05) -> rejected by commit gate
    c_rej = ctrl.evaluate_candidate(20, "Place the gray block on the red pad", 0.62, f20, s20, w20.novelty_r)
    assert not c_rej.committed and c_rej.reason == "insufficient_confidence_margin"

    # Strong confidence gain (0.80 - 0.60 = 0.20 >= 0.05) -> commits AND pins step 20 inspect reveal frame!
    c_ok = ctrl.evaluate_candidate(20, "Place the gray block on the red pad", 0.80, f20, s20, w20.novelty_r)
    assert c_ok.committed and c_ok.pinned_inspect_frame
    assert 20 in ctrl.memory.pinned_steps
    assert subtasks_equivalent("Put the gray block onto red pad", "Place the gray block on the red pad")


def test_confidence_head_loads_real_stage1_v2_checkpoint():
    paths = load_paths_config()
    ckpt_path = Path(paths["stage1_confidence_head_path"])
    if not ckpt_path.is_file():
        return
    head = RidgeTempConfidenceHead.from_checkpoint(ckpt_path)
    p = head.predict_proba(np.zeros(head.weight.shape[0], dtype=np.float64))
    assert 0.0 < p < 1.0


def test_activearena_readiness_check_runs():
    paths = load_paths_config()
    info = check_readiness(paths, "oracle_subtask_oft", ["check_block_color"], "info_gathering_randomized")
    assert "checks" in info and "ready" in info
    assert compute_visual_novelty(np.zeros((32, 32, 3)), np.zeros((32, 32, 3))) == 0.0
