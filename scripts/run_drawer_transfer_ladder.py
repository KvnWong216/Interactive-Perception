#!/usr/bin/env python3
"""Controlled drawer-skill transfer ladder using ordinary seeded resets."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa: F401,E402

from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402
from interactive_perception.rollout import LIBERO_DUMMY_ACTION  # noqa: E402


PROMPT = "Open the middle layer of the drawer"


def make_variants(stock: str) -> dict[str, str]:
    explicit_close = stock.replace(
        "    (On wooden_cabinet_1 main_table_cabinet_region)",
        "    (On wooden_cabinet_1 main_table_cabinet_region)\n"
        "    (Close wooden_cabinet_1_top_region)\n"
        "    (Close wooden_cabinet_1_middle_region)\n"
        "    (Close wooden_cabinet_1_bottom_region)",
    )
    with_butter = explicit_close.replace(
        "    plate_1 - plate",
        "    plate_1 - plate\n    butter_1 - butter",
    ).replace(
        "    (On wine_bottle_1 main_table_wine_bottle_region)",
        "    (On wine_bottle_1 main_table_wine_bottle_region)\n"
        "    (In butter_1 wooden_cabinet_1_middle_region)",
    )
    with_basket = with_butter.replace(
        "      (plate_region",
        "      (basket_region\n"
        "          (:target main_table)\n"
        "          (:ranges ((0.28 0.10 0.30 0.12)))\n"
        "      )\n"
        "      (contain_region (:target basket_1))\n"
        "      (plate_region",
    ).replace(
        "    plate_1 - plate",
        "    plate_1 - plate\n    basket_1 - basket",
    ).replace(
        "    (On wine_bottle_1 main_table_wine_bottle_region)",
        "    (On wine_bottle_1 main_table_wine_bottle_region)\n"
        "    (On basket_1 main_table_basket_region)",
    )
    retrieval_goal = with_basket.replace(
        "(Open wooden_cabinet_1_middle_region)",
        "(In butter_1 basket_1_contain_region)",
    )
    return {
        "B_stock_seeded_reset": stock,
        "C_explicit_close": explicit_close,
        "D_hidden_butter": with_butter,
        "E_add_basket_open_goal": with_basket,
        "F_retrieval_goal": retrieval_goal,
    }


def middle_joint(env) -> float:
    return float(np.asarray(env.env.sim.data.get_joint_qpos("wooden_cabinet_1_middle_level")).reshape(-1)[0])


def run_episode(env, policy, *, seed: int, max_steps: int, replan_steps: int) -> dict:
    env.seed(seed)
    obs = env.reset()
    initial = middle_joint(env)
    minimum = initial
    plan: collections.deque[np.ndarray] = collections.deque()
    success = False
    for step in range(max_steps + 10):
        if step < 10:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        if not plan:
            chunk = policy.sample_chunks(build_observation(obs, PROMPT), 1)[0]
            plan.extend(chunk[:replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        minimum = min(minimum, middle_joint(env))
        if done:
            success = True
            break
    return {
        "seed": seed,
        "success": success,
        "initial_joint": initial,
        "minimum_joint": minimum,
        "final_joint": middle_joint(env),
        "opened_threshold": minimum < -0.14,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument(
        "--env-wrapper", choices=["offscreen", "segmentation"], default="offscreen"
    )
    parser.add_argument("--output", type=Path, default=root / "outputs/gates/drawer_transfer_ladder")
    args = parser.parse_args()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    stock_path = Path(get_libero_path("bddl_files")) / "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl"
    variants = make_variants(stock_path.read_text(encoding="utf-8"))
    if args.conditions is not None:
        unknown = set(args.conditions) - set(variants)
        if unknown:
            raise SystemExit(f"unknown conditions: {sorted(unknown)}")
        variants = {name: variants[name] for name in args.conditions}
    generated = args.output / "generated_bddl"
    generated.mkdir(parents=True, exist_ok=True)
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    env_class = OffScreenRenderEnv if args.env_wrapper == "offscreen" else SegmentationRenderEnv
    report = {
        "prompt": PROMPT,
        "max_steps": args.max_steps,
        "env_wrapper": args.env_wrapper,
        "conditions": {},
    }
    for name, source in variants.items():
        path = generated / f"{name}.bddl"
        path.write_text(source, encoding="utf-8")
        rows = []
        print(f"[ladder] {name}", flush=True)
        for seed in args.seeds:
            env = env_class(
                bddl_file_name=str(path), camera_heights=256, camera_widths=256
            )
            try:
                row = run_episode(
                    env, policy, seed=seed, max_steps=args.max_steps, replan_steps=args.replan_steps
                )
            finally:
                env.close()
            rows.append(row)
            print(
                f"  seed={seed} opened={row['opened_threshold']} "
                f"min_joint={row['minimum_joint']:.4f}",
                flush=True,
            )
        report["conditions"][name] = {
            "episodes": len(rows),
            "open_rate": float(np.mean([row["opened_threshold"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "rows": rows,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "drawer_transfer_ladder.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
