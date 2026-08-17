#!/usr/bin/env python3
"""Diagnostic-only physical validation of the proprioceptive return controller.

The evaluator opens the drawer and applies a fixed OSC perturbation to create a
non-home arm pose.  The tested controller then receives only proprioception and
must recover the stock observation pose.  This validates controller geometry
and camera recovery, but it does not authorize ``OPEN_AND_OBSERVE`` because the
starting pose was not produced by the real pi0.5 drawer-opening trajectory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402
from interactive_perception.observation_option import (  # noqa: E402
    ObservationReturnConfig,
    ObservationReturnController,
)
from scan_t01_open_home_visibility import (  # noqa: E402
    DUMMY_ACTION,
    MIDDLE_JOINT,
    OPEN_POSITION,
    target_pixels,
)
from run_t01_outcome_closed_loop import capture_biview_demo  # noqa: E402


PERTURBATIONS = (
    np.asarray([0.45, 0.35, -0.20, 0.20, -0.15, 0.15, -1.0]),
    np.asarray([0.45, -0.35, -0.20, -0.20, 0.15, -0.15, -1.0]),
    np.asarray([0.35, 0.00, -0.35, 0.20, 0.00, -0.20, -1.0]),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(600, 630)))
    parser.add_argument("--opening-steps", type=int, default=300)
    parser.add_argument("--perturbation-steps", type=int, default=5)
    parser.add_argument("--video-seed", type=int, default=None)
    parser.add_argument(
        "--video-output",
        type=Path,
        default=ROOT / "assets/slides/demos/t01_return_to_observe_diagnostic_seed600.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/t01_return_to_observe_perturbation_v1.json",
    )
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if not args.video_output.is_absolute():
        args.video_output = ROOT / args.video_output

    from libero.libero.envs import SegmentationRenderEnv

    bddl = ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl"
    config = ObservationReturnConfig()
    benchmark = yaml.safe_load(
        (ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text()
    )
    demo = {
        "demo_camera": benchmark["backend"]["demo_camera"],
        "demo_camera_pose": benchmark["backend"]["demo_camera_pose"],
    }
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for offset, seed in enumerate(args.seeds):
            writer = None
            env.seed(seed)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            initial_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
            for position in np.linspace(initial_joint, OPEN_POSITION, args.opening_steps):
                env.env.sim.data.set_joint_qpos(MIDDLE_JOINT, float(position))
                env.env.sim.forward()
                env.env.sim.step()
            obs = env.env._get_observations(force_update=True)
            if seed == args.video_seed:
                import imageio.v2 as imageio

                args.video_output.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(args.video_output, fps=20)
                for _ in range(10):
                    writer.append_data(
                        capture_biview_demo(
                            env, obs, demo=demo, status="DRAWER OPEN / HOME VIEW"
                        )
                    )

            perturbation = PERTURBATIONS[offset % len(PERTURBATIONS)]
            for step in range(args.perturbation_steps):
                obs, _, _, _ = env.step(perturbation.tolist())
                if writer is not None:
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status=f"DIAGNOSTIC PERTURB {step + 1}/{args.perturbation_steps}",
                        )
                    )
            perturbed_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            perturbed_quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)

            controller = ObservationReturnController(config)
            while not controller.status().terminal:
                obs, _, _, _ = env.step(controller.act(obs).tolist())
                if writer is not None:
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status="RETURN_TO_OBSERVE",
                        )
                    )
            status = controller.status()
            if writer is not None:
                for _ in range(15):
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status="RETURN COMPLETE / TARGET VISIBLE",
                        )
                    )
                writer.close()
                writer = None
            pixels = {
                camera: target_pixels(env, obs, camera)
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            visible = max(pixels.values()) >= 5
            passed = status.succeeded and visible
            row = {
                "seed": seed,
                "perturbation": perturbation.tolist(),
                "perturbation_steps": args.perturbation_steps,
                "perturbed_eef_position": perturbed_position.tolist(),
                "perturbed_eef_quaternion_xyzw": perturbed_quaternion.tolist(),
                "return_status": {
                    **dataclasses.asdict(status),
                    "phase": status.phase.value,
                },
                "final_target_pixels": pixels,
                "passed": passed,
            }
            rows.append(row)
            print(
                f"seed={seed} phase={status.phase.value} "
                f"pos_err={status.position_error_metres:.5f} "
                f"ori_err={status.orientation_error_radians:.5f} "
                f"pixels={pixels} passed={passed}",
                flush=True,
            )
            if writer is not None:
                writer.close()
    finally:
        env.close()

    successes = sum(row["passed"] for row in rows)
    report = {
        "schema_version": "interactive-perception.return-to-observe-diagnostic.v1",
        "diagnostic_only": True,
        "authorizes_executor": False,
        "reason_not_authorizing": (
            "drawer opening and arm perturbation are evaluator constructed; "
            "the start state is not a pi0.5 post-open trajectory"
        ),
        "online_controller_inputs": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "online_oracle_inputs": [],
        "controller_config": dataclasses.asdict(config),
        "seeds": args.seeds,
        "successes": successes,
        "trials": len(rows),
        "empirical_rate": successes / len(rows),
        "one_sided_95_lower_bound": exact_binomial_lower_bound(
            successes, len(rows), 0.95
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "successes",
        "trials",
        "empirical_rate",
        "one_sided_95_lower_bound",
    )}, indent=2))


if __name__ == "__main__":
    main()
