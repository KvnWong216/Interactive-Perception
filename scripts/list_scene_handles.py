#!/usr/bin/env python3
"""Print the object, site, body and joint names a scenario actually exposes.

Anchor references in ``benchmark.yaml`` are written by hand, and a wrong name
fails at rollout time after the environment has already been built. Run this
against a scene to confirm every ``hypothesis_anchors[].ref`` resolves, and to
find the correct name when one does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _bootstrap  # noqa: F401,E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--spec",
        default=str(root / "benchmarks" / "interactive_manipulation_v0" / "benchmark.yaml"),
    )
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filter", default="", help="substring filter for names")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args()


def describe(env: Any, needle: str) -> dict[str, list[str]]:
    model = env.env.sim.model

    def names(count: int, lookup: Any) -> list[str]:
        collected = []
        for index in range(count):
            try:
                name = lookup(index)
            except Exception:  # noqa: BLE001 - unnamed slots are expected
                continue
            if name and needle in name:
                collected.append(name)
        return sorted(collected)

    return {
        "objects": sorted(
            name for name in env.env.objects_dict if needle in name
        ),
        "instances": sorted(name for name in env.instance_to_id if needle in name),
        "sites": names(model.nsite, model.site_id2name),
        "bodies": names(model.nbody, model.body_id2name),
        "joints": names(model.njnt, model.joint_id2name),
    }


def check_anchors(task: dict[str, Any], handles: dict[str, list[str]]) -> list[str]:
    """Report anchor refs that will not resolve at rollout time."""

    pools = {
        "object": set(handles["objects"]),
        "site": set(handles["sites"]),
        "body": set(handles["bodies"]),
    }
    problems: list[str] = []
    for anchor in task.get("hypothesis_anchors", []) or []:
        kind = str(anchor.get("kind", "object"))
        ref = str(anchor["ref"])
        if kind not in pools:
            problems.append(f'{anchor["label"]}: unsupported kind {kind!r}')
        elif ref not in pools[kind]:
            problems.append(f'{anchor["label"]}: {kind} {ref!r} not found')
    joint = task.get("reveal_joint_name")
    if joint and joint not in set(handles["joints"]):
        problems.append(f"reveal_joint_name {joint!r} not found")
    return problems


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    with Path(args.spec).expanduser().resolve().open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)

    from libero.libero.envs import SegmentationRenderEnv

    selected = set(args.task_ids or [task["id"] for task in spec["tasks"]])
    results: dict[str, Any] = {}
    failures = 0

    for task in spec["tasks"]:
        if task["id"] not in selected:
            continue
        env = SegmentationRenderEnv(
            bddl_file_name=str(root / task["bddl"]),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            env.seed(args.seed)
            env.reset()
            # Validation always runs against the complete handle set; --filter
            # only narrows what gets printed. Checking against a filtered list
            # would report every anchor outside the filter as missing.
            handles = describe(env, "")
            problems = check_anchors(task, handles)
            failures += len(problems)
            shown = describe(env, args.filter) if args.filter else handles
            results[task["id"]] = {"handles": handles, "anchor_problems": problems}
            if not args.json:
                print(f'\n=== {task["id"]} ===')
                for key, values in shown.items():
                    print(f"  {key} ({len(values)}): {values}")
                if problems:
                    print("  UNRESOLVED ANCHORS:")
                    for problem in problems:
                        print(f"    - {problem}")
                else:
                    print("  all anchor refs resolve")
        finally:
            env.close()

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
