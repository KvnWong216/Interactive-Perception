#!/usr/bin/env python3
"""Audit the privileged-input boundary of an object-level PIU mechanism run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import build_observation  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_subscript_keys(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    return {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/mechanism/piu_object_t01_seed1399_v1.json",
    )
    parser.add_argument(
        "--frontend-manifest",
        type=Path,
        default=ROOT
        / "outputs/piu_object_t01_seed1399_v1/object_scene.manifest.json",
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=ROOT
        / "results/assets/piu_object_t01_seed1399_v1/visualizations_v1/assets_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/audits/piu_object_t01_seed1399_v1_privileged_input_audit.json",
    )
    args = parser.parse_args()
    for name in ("report", "frontend_manifest", "asset_manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable audit exists: {args.output}")
    report = json.loads(args.report.read_text())
    frontend = json.loads(args.frontend_manifest.read_text())
    assets = json.loads(args.asset_manifest.read_text())
    runner = ROOT / "scripts/run_piu_object_mechanism.py"
    runner_tree = ast.parse(runner.read_text())
    main_node = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_names = {node.id for node in ast.walk(main_node) if isinstance(node, ast.Name)}
    allowed_policy_keys = {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    observed_keys = literal_subscript_keys(build_observation)
    model_path = ROOT / report["model"]
    critic_path = ROOT / report["outcome_critic"]["path"]
    rows_clean = all(not row.get("online_oracle_inputs") for row in report["rows"])
    traces_clean = all(
        not event.get("object_runtime", {}).get("online_oracle_inputs")
        for row in report["rows"]
        for event in row["trace"]
    )
    asset_contract = assets["demo_contract"]
    runtime_tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_object_runtime.py",
            "tests/test_action_options.py::test_full_composite_never_reads_privileged_observation_keys",
            "tests/test_observation_option.py::test_controller_reads_only_declared_policy_visible_fields",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    checks = {
        "report_online_oracle_inputs_zero": report.get("online_oracle_inputs") == [],
        "row_online_oracle_inputs_zero": rows_clean,
        "runtime_prediction_oracle_inputs_zero": traces_clean,
        "frontend_online_oracle_inputs_zero": frontend.get("online_oracle_inputs") == [],
        "frontend_offline_labels_consumed_zero": frontend.get("offline_labels_consumed") == [],
        "model_hash_verified": digest(model_path) == report["model_sha256"],
        "outcome_composite_hash_verified": digest(critic_path)
        == report["outcome_critic"]["sha256"],
        "policy_observation_keys_public": observed_keys <= allowed_policy_keys,
        "controller_uses_offscreen_environment": "OffScreenRenderEnv" in main_names,
        "controller_does_not_construct_segmentation_environment": "SegmentationRenderEnv"
        not in main_names,
        "visual_assets_match_report": assets["source_report_sha256"] == digest(args.report),
        "global_demo_view_reads_zero": int(asset_contract["controller_global_view_reads"]) == 0,
        "rendered_assets_not_model_inputs": asset_contract[
            "policy_or_outcome_model_uses_rendered_assets"
        ]
        is False,
        "runtime_guard_tests_passed": runtime_tests.returncode == 0,
        "evaluator_scored_after_controller_terminal": all(
            (row["evaluator_only"].get("final_task") or {}).get(
                "scored_after_controller_terminal", True
            )
            for row in report["rows"]
        ),
    }
    source_paths = (
        runner,
        ROOT / "scripts/build_piu_object_scene_packets.py",
        ROOT / "src/interaction_uncertainty/object_runtime.py",
        ROOT / "src/interaction_uncertainty/object_sidecar.py",
        ROOT / "src/interactive_perception/policy_client.py",
        ROOT / "src/interactive_perception/action_options.py",
        ROOT / "src/interactive_perception/observation_option.py",
        ROOT / "src/interactive_perception/rgb_outcome_critic.py",
        ROOT / "scripts/render_piu_v0_assets.py",
    )
    audit = {
        "schema_version": "interaction-uncertainty.piu-object-mechanism-privileged-audit.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "report": str(args.report.relative_to(ROOT)),
        "report_sha256": digest(args.report),
        "frontend_manifest": str(args.frontend_manifest.relative_to(ROOT)),
        "frontend_manifest_sha256": digest(args.frontend_manifest),
        "asset_manifest": str(args.asset_manifest.relative_to(ROOT)),
        "asset_manifest_sha256": digest(args.asset_manifest),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_paths
        },
        "allowed_policy_observation_keys": sorted(allowed_policy_keys),
        "observed_policy_literal_keys": sorted(observed_keys),
        "controller_environment": "libero.envs.OffScreenRenderEnv",
        "evaluator_environment": "separate replay after controller terminal",
        "checks": checks,
        "runtime_guard_tests": {
            "returncode": runtime_tests.returncode,
            "stdout": runtime_tests.stdout.strip(),
            "stderr": runtime_tests.stderr.strip(),
        },
        "online_oracle_inputs": [],
        "passed": all(checks.values()),
        "limitations": [
            "The run uses a disposable mechanism seed and is not held-out evidence.",
            "The action registry contains one T01 middle-drawer option; exact drawer-layer node localization is not validated.",
            "Information acquisition succeeded once, while final manipulation failed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
