#!/usr/bin/env python3
"""Reproduction gate: confirm the checkpoint and client are wired correctly.

This is a correctness check, not a re-verification of published results. The
failure it is designed to catch is a silent wiring bug -- a missing 180-degree
image rotation, a mis-ordered state vector, a stale checkpoint -- which would
depress success on the challenge scenes and be indistinguishable from the
information-seeking failure the benchmark is trying to measure.

One suite at reduced trial count is sufficient for that purpose and costs an
order of magnitude less than the full four-suite sweep. The gate passes when
success rate lands within ``tolerance`` of the reference in openpi's LIBERO
README.

Run the policy server first, on the GPU host:

    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config pi05_libero --policy.dir <checkpoint_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _bootstrap  # noqa: F401,E402

from interactive_perception.metrics import bootstrap_interval  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.rollout import LIBERO_DUMMY_ACTION  # noqa: E402

# Step budgets copied from openpi's examples/libero/main.py; they are tied to
# the longest training demonstration in each suite.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default=str(root / "benchmarks" / "interactive_manipulation_v0" / "benchmark.yaml"),
    )
    parser.add_argument("--output", default=str(root / "outputs" / "repro_gate"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--trials-per-task", type=int, default=None)
    parser.add_argument("--task-index", type=int, nargs="+", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_init_states(task_suite: Any, task_id: int) -> Any:
    """Read a task's initial states across the torch 2.6 pickle change.

    LIBERO stores initial states as pickled numpy arrays written long before
    torch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``, so
    loading them now fails on an unpickling guard. The guard exists to stop
    untrusted checkpoints executing code; these files come from the pinned
    LIBERO checkout in ``third_party/`` and are the same data the benchmark has
    always shipped, so relaxing it for this one call is safe.

    The patch is scoped to the call and reverted in a ``finally`` block rather
    than applied globally, so nothing else in the process silently loses the
    protection.
    """

    import torch

    original = torch.load

    def permissive(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = permissive
    try:
        return task_suite.get_task_init_states(task_id)
    finally:
        torch.load = original


def run_task(
    *,
    env: Any,
    policy: OpenPiWebsocketPolicy,
    description: str,
    initial_states: Any,
    trials: int,
    max_steps: int,
    args: argparse.Namespace,
) -> list[bool]:
    results: list[bool] = []
    for episode in range(trials):
        env.reset()
        obs = env.set_init_state(initial_states[episode])
        plan: collections.deque = collections.deque()
        done = False
        step = 0
        while step < max_steps + args.num_steps_wait:
            try:
                if step < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    step += 1
                    continue
                if not plan:
                    packet = build_observation(
                        obs, description, resize_size=args.resize_size
                    )
                    chunk = policy.sample_chunks(packet, 1)[0]
                    if len(chunk) < args.replan_steps:
                        raise ValueError(
                            f"policy returned {len(chunk)} steps, need {args.replan_steps}"
                        )
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                step += 1
                if done:
                    break
            except Exception as error:  # noqa: BLE001 - recorded per episode
                print(f"    episode {episode} aborted: {error}", flush=True)
                break
        results.append(bool(done))
    return results


def main() -> None:
    args = parse_args()
    with Path(args.spec).expanduser().resolve().open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    gate = spec.get("reproduction_gate", {}) or {}
    suite = args.suite or str(gate.get("suite", "libero_object"))
    trials = args.trials_per_task or int(gate.get("trials_per_task", 20))
    reference = float(gate.get("reference_success_rate", 0.982))
    tolerance = float(gate.get("tolerance", 0.05))

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    task_suite = libero_benchmark.get_benchmark_dict()[suite]()
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)

    indices = args.task_index or list(range(task_suite.n_tasks))
    per_task: dict[str, float] = {}
    all_results: list[float] = []

    for task_id in indices:
        task = task_suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        try:
            env.seed(args.seed)
            print(f"[repro] {suite}[{task_id}] {task.language}", flush=True)
            results = run_task(
                env=env,
                policy=policy,
                description=str(task.language),
                initial_states=load_init_states(task_suite, task_id),
                trials=trials,
                max_steps=MAX_STEPS[suite],
                args=args,
            )
        finally:
            env.close()
        rate = float(np.mean(results)) if results else 0.0
        per_task[str(task.language)] = rate
        all_results.extend(float(item) for item in results)
        print(f"  success rate: {rate:.3f}", flush=True)

    overall = float(np.mean(all_results)) if all_results else 0.0
    lower, upper = bootstrap_interval(all_results)
    passed = abs(overall - reference) <= tolerance

    summary = {
        "suite": suite,
        "trials_per_task": trials,
        "tasks": len(indices),
        "episodes": len(all_results),
        "success_rate": overall,
        "success_ci95": [lower, upper],
        "reference_success_rate": reference,
        "tolerance": tolerance,
        "server_metadata": policy.server_metadata,
        "per_task_success": per_task,
        "gate_passed": passed,
        "note": (
            "A failure here means the client/checkpoint wiring is wrong, and "
            "no challenge-scenario result should be trusted until it is fixed."
        ),
    }
    report_path = output_root / f"repro_{suite}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task_success"}, indent=2))
    print(f"Report: {report_path}")
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
