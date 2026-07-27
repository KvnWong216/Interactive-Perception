#!/usr/bin/env python3
"""Run Information-Enrichment v0 cases in isolated MuJoCo processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    starter = project_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument(
        "--spec",
        default=str(
            starter
            / "benchmarks"
            / "information_enrichment_v0"
            / "benchmark.yaml"
        ),
    )
    parser.add_argument("--bddl-root", default=str(starter))
    parser.add_argument(
        "--output",
        default=str(starter / "outputs" / "information_enrichment_v0"),
    )
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    bddl_root = Path(args.bddl_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    all_task_ids = [task["id"] for task in spec["tasks"]]
    task_ids = args.task_ids or all_task_ids
    validator = Path(__file__).with_name(
        "validate_information_enrichment_v0.py"
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
    rows = []
    subprocesses = []
    for task_id in task_ids:
        for seed in args.seeds:
            case_report = (
                output_root
                / task_id
                / f"seed_{seed:03d}"
                / "case_report.json"
            )
            if case_report.exists():
                case_report.unlink()
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
                str(output_root),
                "--task-ids",
                task_id,
                "--seeds",
                str(seed),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ]
            print(f"[run] {task_id} seed={seed}", flush=True)
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if case_report.exists():
                with case_report.open("r", encoding="utf-8") as file:
                    row = json.load(file)
            else:
                row = {
                    "task_id": task_id,
                    "seed": seed,
                    "passed": False,
                    "error": "case_report.json missing",
                }
            rows.append(row)
            subprocesses.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            print(f"  {'PASS' if row.get('passed') else 'FAIL'}", flush=True)
    summary = {
        "benchmark": spec["name"],
        "version": spec["version"],
        "task_ids": task_ids,
        "seeds": args.seeds,
        "case_runs": len(rows),
        "case_passes": sum(bool(row.get("passed")) for row in rows),
        "all_checks_passed": all(row.get("passed") for row in rows),
        "rows": rows,
        "subprocesses": subprocesses,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report = output_root / "validation_report.json"
    with report.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"rows", "subprocesses"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report: {report}")
    if args.strict and not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
