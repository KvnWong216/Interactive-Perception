#!/usr/bin/env python3
"""Run the prompt ladder on the challenge scenes and record uncertainty traces.

Each scenario is run under the same policy and the same initial state with only
the prompt changed, which isolates the decision from the skill:

    implicit    the final goal alone -- the condition under test
    hinted      the final goal plus a content-free nudge
    explicit    the information step, then the same final goal
    capability  the information step alone

The comparison that matters is ``capability`` against ``implicit``. If the
policy performs the information step when told to and never performs it when
not told, the failure is a decision failure. If it cannot perform the step even
when told, the scenario is measuring a missing skill and cannot support any
claim about information seeking.

Requires a policy server. On the GPU host:

    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config pi05_libero --policy.dir <checkpoint_dir>

Use ``--policy stub`` to exercise the pipeline on a machine with no GPU. Stub
results are diagnostic only and must never be reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _bootstrap  # noqa: F401,E402

from interactive_perception.anchors import AnchorSpec  # noqa: E402
from interactive_perception.metrics import aggregate, paired_difference  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    ScriptedStubPolicy,
)
from interactive_perception.primitive_decoder import PrimitiveDecoderConfig  # noqa: E402
from interactive_perception.rollout import RolloutConfig, run_episode, write_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--spec",
        default=str(root / "benchmarks" / "interactive_manipulation_v0" / "benchmark.yaml"),
    )
    parser.add_argument("--output", default=str(root / "outputs" / "challenge_rollout"))
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["implicit", "explicit"],
        help="prompt ladder rungs to run",
    )
    parser.add_argument("--policy", choices=["pi05", "stub"], default="pi05")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--probe-samples", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="store policy-view frames as .npy for the demo renderer",
    )
    return parser.parse_args()


def build_policy(args: argparse.Namespace, task: dict[str, Any]) -> Any:
    if args.policy == "pi05":
        return OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    # The stub drives at the first declared anchor so traces have structure; it
    # models nothing about the real policy and exists only to test plumbing.
    return ScriptedStubPolicy(goal_position=(0.0, 0.0, 1.0), seed=abs(hash(task["id"])) % 2**31)


def rollout_config(spec: dict[str, Any], args: argparse.Namespace) -> RolloutConfig:
    defaults = spec.get("rollout", {}) or {}
    return RolloutConfig(
        max_steps=args.max_steps or int(defaults.get("max_steps", 400)),
        num_steps_wait=int(defaults.get("num_steps_wait", 10)),
        replan_steps=int(defaults.get("replan_steps", 5)),
        probe_samples=args.probe_samples or int(defaults.get("probe_samples", 16)),
        probe_every=args.probe_every or int(defaults.get("probe_every", 20)),
        camera=spec["backend"]["policy_camera"],
        resize_size=int(spec["backend"].get("policy_resize", 224)),
        commitment_probability=float(defaults.get("commitment_probability", 0.6)),
        decoder=PrimitiveDecoderConfig(),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with Path(args.spec).expanduser().resolve().open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)

    from libero.libero.envs import SegmentationRenderEnv

    config = rollout_config(spec, args)
    selected = set(args.task_ids or [task["id"] for task in spec["tasks"]])
    ladder = spec.get("prompt_ladder", {}).get("variants", ["implicit", "explicit"])
    for variant in args.variants:
        if variant not in ladder:
            raise SystemExit(f"unknown prompt variant {variant!r}; spec allows {ladder}")

    outcomes_by_variant: dict[str, list[Any]] = {variant: [] for variant in args.variants}

    for task in spec["tasks"]:
        if task["id"] not in selected:
            continue
        variants = task.get("prompt_variants") or {}
        anchor_specs = [
            AnchorSpec.from_dict(item) for item in (task.get("hypothesis_anchors") or [])
        ]
        if len(anchor_specs) < 2:
            print(f'[skip] {task["id"]}: needs >= 2 hypothesis anchors', flush=True)
            continue

        for variant in args.variants:
            prompt = variants.get(variant)
            if not prompt:
                print(f'[skip] {task["id"]}/{variant}: no prompt defined', flush=True)
                continue
            prompt = " ".join(str(prompt).split())

            for seed in args.seeds:
                print(f'[rollout] {task["id"]} {variant} seed={seed}', flush=True)
                env = SegmentationRenderEnv(
                    bddl_file_name=str(root / task["bddl"]),
                    camera_heights=args.height,
                    camera_widths=args.width,
                )
                frames: list[np.ndarray] | None = [] if args.save_frames else None
                try:
                    policy = build_policy(args, task)
                    outcome, records = run_episode(
                        env=env,
                        policy=policy,
                        task=task,
                        prompt=prompt,
                        prompt_variant=variant,
                        anchor_specs=anchor_specs,
                        seed=seed,
                        config=config,
                        frames=frames,
                    )
                finally:
                    env.close()

                case_dir = output_root / task["id"] / variant / f"seed_{seed:03d}"
                write_trace(
                    case_dir / "trace.jsonl",
                    outcome=outcome,
                    records=records,
                    metadata={
                        "prompt": prompt,
                        "policy": args.policy,
                        "expected_nbv_verdict": task.get("expected_nbv_verdict"),
                        "primary_condition": task.get("primary_condition"),
                        "skill_confound": task.get("skill_confound"),
                    },
                )
                if frames:
                    np.save(case_dir / "frames.npy", np.stack(frames))
                outcomes_by_variant[variant].append(outcome)
                print(
                    f'  success={outcome.task_success} '
                    f'endpoint={outcome.information_endpoint_reached} '
                    f'mean_vacuity={outcome.mean_vacuity:.3f}',
                    flush=True,
                )

    summary: dict[str, Any] = {
        "spec": str(args.spec),
        "policy": args.policy,
        "seeds": args.seeds,
        "variants": args.variants,
        "conditions": {},
    }
    for variant, outcomes in outcomes_by_variant.items():
        if outcomes:
            summary["conditions"][variant] = aggregate(
                outcomes, condition=variant
            ).to_dict()

    # The decision-versus-skill contrast, computed on episodes sharing a seed.
    if "capability" in outcomes_by_variant and "implicit" in outcomes_by_variant:
        summary["capability_minus_implicit"] = paired_difference(
            outcomes_by_variant["capability"],
            outcomes_by_variant["implicit"],
            attribute="information_endpoint_reached",
        )
    if "explicit" in outcomes_by_variant and "implicit" in outcomes_by_variant:
        summary["explicit_minus_implicit"] = paired_difference(
            outcomes_by_variant["explicit"],
            outcomes_by_variant["implicit"],
            attribute="information_endpoint_reached",
        )

    report_path = output_root / "rollout_summary.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    if args.policy == "stub":
        print("\nNOTE: stub policy results are plumbing checks, not findings.")


if __name__ == "__main__":
    main()
