#!/usr/bin/env python3
"""Machine-readable oracle-boundary audit for the PIU V0 embodied smoke."""

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
from interaction_uncertainty.contracts import ScenePacket  # noqa: E402
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
        "--smoke",
        type=Path,
        default=ROOT / "results/smoke/piu_v0_end_to_end_v3_seed1399.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results/models/piu_v0_3_sidecar.pt",
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/audits/piu_v0_privileged_input_audit.json",
    )
    args = parser.parse_args()
    for name in ("smoke", "model", "asset_manifest", "output"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)
    smoke_path = args.smoke
    model_path = args.model
    output = args.output
    if output.exists():
        raise FileExistsError(f"immutable audit exists: {output}")
    smoke = json.loads(smoke_path.read_text())
    runner = ROOT / "scripts/run_piu_v0_smoke.py"
    tree = ast.parse(runner.read_text())
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_names = {node.id for node in ast.walk(main_node) if isinstance(node, ast.Name)}
    controller_uses_public_env = "OffScreenRenderEnv" in main_names
    controller_constructs_segmentation_env = "SegmentationRenderEnv" in main_names
    evaluator_helpers = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"static_evaluator", "replay_final_task_evaluator"}
    }
    allowed_policy_keys = {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    observed_keys = literal_subscript_keys(build_observation)
    row_oracles_empty = all(not row["online_oracle_inputs"] for row in smoke["rows"])
    critic_contract = smoke.get("outcome_critic", {})
    critic_oracles_empty = critic_contract.get("online_oracle_inputs") == []
    critic_path = (
        ROOT / critic_contract["path"] if critic_contract.get("path") else None
    )
    critic_artifact_intact = bool(
        critic_path is not None
        and critic_path.exists()
        and digest(critic_path) == critic_contract.get("sha256")
    )
    trace_terminals_present = all(bool(row["terminal"]) for row in smoke["rows"])
    scene_firewall = False
    try:
        ScenePacket(
            frame_id="audit",
            prompt="Place the butter in the basket",
            objects=(),
            unknown_regions=(),
            public_robot_state=(),
            online_oracle_inputs=("segmentation",),
        )
    except ValueError:
        scene_firewall = True
    runtime_test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_piu_v0.py::test_scene_packet_rejects_oracle_inputs",
            "tests/test_action_options.py::test_full_composite_never_reads_privileged_observation_keys",
            "tests/test_observation_option.py::test_controller_reads_only_declared_policy_visible_fields",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    source_paths = (
        ROOT / "src/interaction_uncertainty/contracts.py",
        ROOT / "src/interaction_uncertainty/candidates.py",
        ROOT / "src/interaction_uncertainty/optimizer.py",
        ROOT / "src/interaction_uncertainty/sidecar.py",
        ROOT / "src/interaction_uncertainty/scene_memory.py",
        ROOT / "src/interactive_perception/policy_client.py",
        ROOT / "src/interactive_perception/action_options.py",
        ROOT / "src/interactive_perception/observation_option.py",
        ROOT / "src/interactive_perception/rgb_outcome_critic.py",
        runner,
    )
    asset_manifest = None
    asset_contract_passed = True
    if args.asset_manifest is not None:
        asset_manifest = json.loads(args.asset_manifest.read_text())
        contract = asset_manifest["demo_contract"]
        asset_contract_passed = all(
            (
                asset_manifest["source_report_sha256"] == digest(smoke_path),
                int(contract["controller_global_view_reads"]) == 0,
                contract["policy_or_outcome_model_uses_rendered_assets"] is False,
            )
        )
        source_paths = source_paths + (ROOT / "scripts/render_piu_v0_assets.py",)
    passed = all(
        (
            smoke["online_oracle_inputs"] == [],
            row_oracles_empty,
            critic_oracles_empty,
            critic_artifact_intact,
            trace_terminals_present,
            observed_keys <= allowed_policy_keys,
            controller_uses_public_env,
            not controller_constructs_segmentation_env,
            evaluator_helpers == {"static_evaluator", "replay_final_task_evaluator"},
            scene_firewall,
            runtime_test.returncode == 0,
            asset_contract_passed,
        )
    )
    act_limitation = (
        "DIRECT_ACT was physically executed on one disposable behavior seed and failed; this is diagnostic, not an ACT reliability estimate."
        if smoke.get("physical_final_act_executed")
        else "DIRECT_ACT is a semantic handoff and was not physically executed."
    )
    report = {
        "schema_version": "interaction-uncertainty.piu-privileged-input-audit.v0",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "smoke": str(smoke_path.relative_to(ROOT)),
        "smoke_sha256": digest(smoke_path),
        "model": str(model_path.relative_to(ROOT)),
        "model_sha256": digest(model_path),
        "asset_manifest": (
            str(args.asset_manifest.relative_to(ROOT))
            if args.asset_manifest is not None
            else None
        ),
        "asset_manifest_sha256": (
            digest(args.asset_manifest) if args.asset_manifest is not None else None
        ),
        "asset_contract_passed": asset_contract_passed,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_paths
        },
        "allowed_policy_observation_keys": sorted(allowed_policy_keys),
        "observed_policy_literal_keys": sorted(observed_keys),
        "controller_environment": "libero.envs.OffScreenRenderEnv",
        "controller_constructs_segmentation_environment": controller_constructs_segmentation_env,
        "evaluator_environment": "separate SegmentationRenderEnv/replay after terminal",
        "actual_trace_rows_have_zero_oracle_inputs": row_oracles_empty,
        "rgb_outcome_critic_has_zero_oracle_inputs": critic_oracles_empty,
        "rgb_outcome_composite_hash_verified": critic_artifact_intact,
        "scene_packet_runtime_firewall": scene_firewall,
        "runtime_guard_tests": {
            "returncode": runtime_test.returncode,
            "stdout": runtime_test.stdout.strip(),
            "stderr": runtime_test.stderr.strip(),
        },
        "online_oracle_inputs": [],
        "passed": passed,
        "limitations": [
            "Static/runtime isolation does not validate learned calibration.",
            "Initial V0 belief is a prefix proxy, not Grounding DINO/SAM object grounding.",
            "The v12b outcome critic passed T01 clean and sealed gates but remains T01-specific; it does not establish scene-disjoint or cross-object generalization.",
            act_limitation,
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
