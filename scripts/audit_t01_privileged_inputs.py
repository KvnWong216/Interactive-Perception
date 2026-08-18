#!/usr/bin/env python3
"""Machine-readable static audit of the OPEN_AND_OBSERVE controller boundary."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402

from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.observation_option import (  # noqa: E402
    ObservationReturnController,
)
from interactive_perception.policy_client import build_observation  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_subscript_keys(function) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


def main() -> None:
    output = ROOT / "results/t01_open_and_observe_privileged_input_audit_v1.json"
    if output.exists():
        raise FileExistsError(f"audit result is immutable: {output}")
    policy_keys = literal_subscript_keys(build_observation)
    return_keys = literal_subscript_keys(ObservationReturnController.act)
    allowed_policy = {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    forbidden_fragments = (
        "segmentation",
        "drawer_joint",
        "hidden_target_pose",
        "task_success_predicate",
        "bev",
    )
    controller_source = "\n".join(
        (
            inspect.getsource(build_observation),
            inspect.getsource(ObservationReturnController.act),
            inspect.getsource(execute_open_and_observe),
        )
    ).lower()
    source_forbidden = [
        fragment for fragment in forbidden_fragments if fragment in controller_source
    ]
    collector = ROOT / "scripts/collect_t01_open_and_observe_effect.py"
    collector_tree = ast.parse(collector.read_text())
    main_node = next(
        node
        for node in collector_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_names = {
        node.id for node in ast.walk(main_node) if isinstance(node, ast.Name)
    }
    collector_uses_public_env = "OffScreenRenderEnv" in main_names
    collector_online_segmentation_env = "SegmentationRenderEnv" in main_names
    closed_loop = ROOT / "scripts/run_t01_closed_loop_v5.py"
    closed_loop_source = closed_loop.read_text()
    closed_loop_public_controller = "env = OffScreenRenderEnv(" in closed_loop_source
    closed_loop_posthoc_replay = "replay_evaluator_trace(" in closed_loop_source
    closed_loop_scores_controller_env = "evaluator_pixels(env," in closed_loop_source
    closed_loop_uses_probabilistic_q = "TemporalBelief" in closed_loop_source
    runtime_test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            (
                "tests/test_action_options.py::"
                "test_full_composite_never_reads_privileged_observation_keys"
            ),
            (
                "tests/test_observation_option.py::"
                "test_controller_reads_only_declared_policy_visible_fields"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    passed = (
        policy_keys <= allowed_policy
        and return_keys <= allowed_policy
        and not source_forbidden
        and collector_uses_public_env
        and not collector_online_segmentation_env
        and closed_loop_public_controller
        and closed_loop_posthoc_replay
        and not closed_loop_scores_controller_env
        and not closed_loop_uses_probabilistic_q
        and runtime_test.returncode == 0
    )
    report = {
        "schema_version": "interactive-perception.privileged-input-audit.v1",
        "scope": "OPEN_AND_OBSERVE controller, v4 collector, and sealed-gated v5 runner",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                ROOT / "src/interactive_perception/policy_client.py",
                ROOT / "src/interactive_perception/observation_option.py",
                ROOT / "src/interactive_perception/action_options.py",
                ROOT / "src/interactive_perception/minimal_pipeline.py",
                collector,
                closed_loop,
            )
        },
        "allowed_online_observation_keys": sorted(allowed_policy),
        "build_observation_literal_keys": sorted(policy_keys),
        "return_controller_literal_keys": sorted(return_keys),
        "forbidden_controller_source_fragments": source_forbidden,
        "controller_environment": "libero.envs.OffScreenRenderEnv",
        "evaluator_environment": "libero.envs.SegmentationRenderEnv",
        "evaluator_timing": "recorded-action replay after controller termination",
        "collector_main_mentions_segmentation_environment": collector_online_segmentation_env,
        "closed_loop_controller_environment": "libero.envs.OffScreenRenderEnv",
        "closed_loop_uses_post_termination_evaluator_replay": closed_loop_posthoc_replay,
        "closed_loop_scores_controller_environment": closed_loop_scores_controller_env,
        "closed_loop_uses_probabilistic_temporal_progress": closed_loop_uses_probabilistic_q,
        "runtime_guard_tests": {
            "returncode": runtime_test.returncode,
            "stdout": runtime_test.stdout.strip(),
            "stderr": runtime_test.stderr.strip(),
        },
        "online_oracle_inputs": [],
        "passed": passed,
        "limitation": (
            "Passing source/input isolation does not authorize a rollout. The v5 runner "
            "remains sealed-audit gated, and its global demo panel is evaluator-only."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
