#!/usr/bin/env python3
"""Run each interactive-manipulation task/seed in a fresh MuJoCo process."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    default_project_root = Path(__file__).resolve().parents[1]
    default_starter = default_project_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(default_project_root))
    parser.add_argument(
        "--spec",
        default=str(
            default_starter
            / "benchmarks"
            / "interactive_manipulation_v0"
            / "benchmark.yaml"
        ),
    )
    parser.add_argument("--bddl-root", default=str(default_starter))
    parser.add_argument(
        "--output",
        default=str(
            default_starter / "outputs" / "interactive_manipulation_v0"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Number of isolated MuJoCo child processes to run concurrently.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def prompt_contract_errors(spec: dict[str, Any]) -> list[str]:
    forbidden = [
        phrase.casefold()
        for phrase in spec["prompt_contract"]["forbidden_exploration_phrases"]
    ]
    errors: list[str] = []
    for task in spec["tasks"]:
        leaks = [
            phrase
            for phrase in forbidden
            if phrase in task["prompt"].casefold()
        ]
        if leaks:
            errors.append(f'{task["id"]}: exploration leakage {leaks}')
    return errors


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    bddl_root = Path(args.bddl_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / ".shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)

    spec = load_yaml(spec_path)
    tasks_by_id = {task["id"]: task for task in spec["tasks"]}
    selected_task_ids = args.task_ids or list(tasks_by_id)
    unknown_task_ids = set(selected_task_ids) - set(tasks_by_id)
    if unknown_task_ids:
        raise KeyError(f"Unknown task IDs: {sorted(unknown_task_ids)}")

    validator = Path(__file__).resolve().with_name(
        "validate_interactive_manipulation_v0.py"
    )
    environment = os.environ.copy()
    environment.setdefault(
        "MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"
    )
    environment.setdefault(
        "NUMBA_CACHE_DIR", str(project_root / ".cache" / "numba")
    )
    environment.setdefault(
        "MPLCONFIGDIR", str(project_root / ".cache" / "matplotlib")
    )
    Path(environment["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    def run_case(
        task_id: str, seed: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        shard_output = shard_root / task_id / f"seed_{seed:03d}"
        command = [
            sys.executable,
            str(validator),
            "--project-root",
            str(project_root),
            "--spec",
            str(spec_path),
            "--bddl-root",
            str(bddl_root),
            "--output",
            str(shard_output),
            "--task-ids",
            task_id,
            "--seeds",
            str(seed),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--settle-steps",
            str(args.settle_steps),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess_record = {
            "task_id": task_id,
            "seed": seed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        report_path = shard_output / "validation_report.json"
        if not report_path.exists():
            return (
                {
                    "task_id": task_id,
                    "seed": seed,
                    "passed": False,
                    "error": (
                        "Child validation report missing; "
                        f"returncode={completed.returncode}"
                    ),
                },
                subprocess_record,
            )

        with report_path.open("r", encoding="utf-8") as file:
            child_report = json.load(file)
        child_rows = child_report.get("rows", [])
        if len(child_rows) != 1:
            return (
                {
                    "task_id": task_id,
                    "seed": seed,
                    "passed": False,
                    "error": (
                        "Expected exactly one child row, "
                        f"got {len(child_rows)}"
                    ),
                },
                subprocess_record,
            )
        row = child_rows[0]
        source_case_dir = shard_output / task_id / f"seed_{seed:03d}"
        final_case_dir = output_root / task_id / f"seed_{seed:03d}"
        if final_case_dir.exists():
            shutil.rmtree(final_case_dir)
        final_case_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_case_dir, final_case_dir)
        for snapshot_name in ("initial", "oracle_revealed"):
            snapshot = row.get(snapshot_name)
            if isinstance(snapshot, dict) and "image" in snapshot:
                snapshot["image"] = str(
                    final_case_dir / Path(snapshot["image"]).name
                )
        return row, subprocess_record

    cases = [
        (task_id, seed)
        for task_id in selected_task_ids
        for seed in args.seeds
    ]
    order = {case: index for index, case in enumerate(cases)}
    rows: list[dict[str, Any]] = []
    subprocesses: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.jobs, len(cases))
    ) as executor:
        future_to_case = {
            executor.submit(run_case, task_id, seed): (task_id, seed)
            for task_id, seed in cases
        }
        for future in concurrent.futures.as_completed(future_to_case):
            task_id, seed = future_to_case[future]
            print(f"[done] {task_id} seed={seed}", flush=True)
            try:
                row, subprocess_record = future.result()
            except Exception as error:
                row = {
                    "task_id": task_id,
                    "seed": seed,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                subprocess_record = {
                    "task_id": task_id,
                    "seed": seed,
                    "returncode": None,
                    "stdout": "",
                    "stderr": row["error"],
                }
            rows.append(row)
            subprocesses.append(subprocess_record)
            print(
                f'  {"PASS" if row.get("passed") else "FAIL"}',
                flush=True,
            )
    rows.sort(key=lambda row: order[(row["task_id"], row["seed"])])
    subprocesses.sort(
        key=lambda record: order[(record["task_id"], record["seed"])]
    )

    errors = prompt_contract_errors(spec)
    summary = {
        "benchmark": spec["name"],
        "version": spec["version"],
        "spec": str(spec_path),
        "execution_mode": "fresh_process_per_task_seed",
        "jobs": args.jobs,
        "policy_camera": spec["backend"]["policy_camera"],
        "active_viewpoint_actions": spec["backend"][
            "active_viewpoint_actions"
        ],
        "prompt_contract_errors": errors,
        "task_ids": selected_task_ids,
        "seeds": args.seeds,
        "expected_case_runs": len(selected_task_ids) * len(args.seeds),
        "case_runs": len(rows),
        "case_passes": sum(bool(row.get("passed")) for row in rows),
        "all_checks_passed": bool(
            not errors
            and len(rows) == len(selected_task_ids) * len(args.seeds)
            and all(row.get("passed") for row in rows)
        ),
        "rows": rows,
        "subprocesses": subprocesses,
    }
    report_path = output_root / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"rows", "subprocesses"}
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Report: {report_path}")
    if args.strict and not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
