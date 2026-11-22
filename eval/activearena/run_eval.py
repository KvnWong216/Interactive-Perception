#!/usr/bin/env python3
"""Unified ActiveArena Closed-Loop Evaluation Launcher for IP-VLA.

Supports all Stage-2 evaluation arms on identical paired ActiveArena seeds:
  1. `instruction_oft`    : Official `oft_instruction_action_12_ws` end-to-end baseline
  2. `oracle_subtask_oft` : Official `oft_subtask_action_12_ws` + Privileged Env Oracle (upper bound)
  3. `ipvla_planner`      : Autonomous IP-VLA Stage-2 VLM Planner + Continuous Cerebellum Inertia
                            + Inspect-Keyframe Memory -> Frozen `oft_subtask_action_12_ws` (0 Oracle!)

Examples:
  # Dry-run preflight check (zero GPU usage)
  python -m eval.activearena.run_eval --mode oracle_subtask_oft --tasks check_block_color --episodes 5 --dry-run

  # Official Oracle upper bound on pilot IA tasks
  python -m eval.activearena.run_eval --mode oracle_subtask_oft --setting ood --tasks check_block_color check_cola_date --episodes 10

  # Official End-to-End Instruction baseline
  python -m eval.activearena.run_eval --mode instruction_oft --setting ood --tasks check_block_color check_cola_date --episodes 10

  # Autonomous IP-VLA Stage-2 (with cerebellum ablation flags)
  python -m eval.activearena.run_eval --mode ipvla_planner --setting ood \\
      --tasks check_block_color check_cola_date --episodes 10 \\
      --cerebellum-mode physical_cerebellum --memory-mode inspect_keyframe_pin
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import requests

from ipvla.config import REPO_ROOT, load_paths_config, load_stage2_config


SETTING_MAP = {
    "id": "info_gathering_demo",
    "ood": "info_gathering_randomized",
    "info_gathering_demo": "info_gathering_demo",
    "info_gathering_randomized": "info_gathering_randomized",
}

MODE_TO_MODEL = {
    "instruction_oft": "oft_instruction_action_12_ws",
    "oracle_subtask_oft": "oft_subtask_action_12_ws",
    "ipvla_planner": "oft_subtask_action_12_ws",
}


def check_readiness(paths: dict[str, Any], mode: str, tasks: list[str], setting_name: str) -> dict[str, Any]:
    aa = paths["activearena"]
    sim_repo = Path(aa["sim_repo"])
    vla_repo = Path(aa["vla_repo"])
    weights_root = Path(aa["weights_root"])
    model_name = MODE_TO_MODEL[mode]
    ckpt_path = weights_root / model_name / "checkpoints" / "steps_100000_pytorch_model.pt"

    checks = {
        "sim_repo_exists": sim_repo.is_dir(),
        "vla_repo_exists": vla_repo.is_dir(),
        "assets_embodiments_exists": (sim_repo / "assets" / "embodiments" / "astribot_descriptions_texture").is_dir(),
        "assets_objects_exists": (sim_repo / "assets" / "objects" / "005_button").is_dir(),
        "assets_bg_exists": (sim_repo / "assets" / "background_texture").is_dir(),
        "base_vlm_exists": Path(paths["base_vlm_dir"]).is_dir(),
        "action_ckpt_exists": ckpt_path.is_file() and ckpt_path.stat().st_size > 1_000_000_000,
        "action_ckpt_path": str(ckpt_path),
        "action_ckpt_bytes": ckpt_path.stat().st_size if ckpt_path.is_file() else 0,
    }
    if mode == "ipvla_planner":
        lora = Path(paths["stage1_lora_path"])
        conf = Path(paths["stage1_confidence_head_path"])
        checks["stage1_lora_exists"] = lora.is_file()
        checks["stage1_conf_head_exists"] = conf.is_file()

    seed_dir = sim_repo / "eval_seed_lists" / setting_name
    missing_tasks = [t for t in tasks if not (seed_dir / f"{t}.json").is_file()]
    checks["seed_dir_exists"] = seed_dir.is_dir()
    checks["missing_task_seed_lists"] = missing_tasks

    ready = all(
        v for k, v in checks.items() if k.endswith("_exists")
    ) and not missing_tasks
    return {"ready": ready, "mode": mode, "model_name": model_name, "setting": setting_name, "checks": checks}


def wait_for_http(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    sess = requests.Session()
    sess.trust_env = False
    while time.monotonic() < deadline:
        try:
            r = sess.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Service did not become ready at {url} within {timeout_s:.0f}s")


def wait_for_tcp(host: str, port: int, timeout_s: float = 180.0) -> None:
    import socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return
        except OSError:
            time.sleep(1.5)
    raise TimeoutError(f"Action server did not open {host}:{port} within {timeout_s:.0f}s")


def main() -> None:
    paths = load_paths_config()
    stage2_cfg = load_stage2_config()
    eval_defaults = stage2_cfg.get("eval", {})

    parser = argparse.ArgumentParser(description="IP-VLA Unified ActiveArena Evaluator")
    parser.add_argument(
        "--mode",
        choices=("instruction_oft", "oracle_subtask_oft", "ipvla_planner"),
        required=True,
    )
    parser.add_argument("--setting", default=eval_defaults.get("default_setting", "ood"))
    parser.add_argument("--tasks", nargs="+", default=eval_defaults.get("pilot_tasks", ["check_block_color"]))
    parser.add_argument("--episodes", type=int, default=int(eval_defaults.get("default_episodes", 10)))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--action-port", type=int, default=int(eval_defaults.get("action_server_base_port", 19000)))
    parser.add_argument("--planner-port", type=int, default=int(eval_defaults.get("planner_server_port", 19100)))
    parser.add_argument(
        "--cerebellum-mode",
        choices=("physical_cerebellum", "no_inertia_k0", "fixed_period"),
        default=stage2_cfg.get("cerebellum", {}).get("mode", "physical_cerebellum"),
    )
    parser.add_argument(
        "--memory-mode",
        choices=("inspect_keyframe_pin", "sliding_window"),
        default=stage2_cfg.get("memory", {}).get("mode", "inspect_keyframe_pin"),
    )
    parser.add_argument("--no-lora", action="store_true", help="Ablation: use Base Qwen3-VL-2B as planner")
    parser.add_argument("--dry-run", action="store_true", help="Verify paths, assets, checkpoints, and exit")
    args = parser.parse_args()

    setting_name = SETTING_MAP.get(args.setting, args.setting)
    readiness = check_readiness(paths, args.mode, args.tasks, setting_name)
    if args.dry_run:
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        sys.exit(0 if readiness["ready"] else 2)

    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit("ActiveArena preflight check failed; run `bash eval/activearena/setup_activearena.sh` first.")

    aa = paths["activearena"]
    sim_repo = Path(aa["sim_repo"])
    vla_repo = Path(aa["vla_repo"])
    weights_root = Path(aa["weights_root"])
    model_name = MODE_TO_MODEL[args.mode]
    ckpt_path = weights_root / model_name / "checkpoints" / "steps_100000_pytorch_model.pt"

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{args.mode}_{args.cerebellum_mode}_{args.memory_mode}"
    run_dir = Path(aa["output_root"]) / setting_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bg_procs: list[subprocess.Popen] = []

    def _cleanup() -> None:
        for p in reversed(bg_procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    try:
        # 1. Start frozen ActiveArena-VLA Action Policy Server
        action_env = os.environ.copy()
        action_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        action_env["HF_HUB_OFFLINE"] = "1"
        action_env["ACTIVEARENA_VLA_BASE_VLM"] = paths["base_vlm_dir"]
        action_env["PYTHONPATH"] = f"{vla_repo}:{action_env.get('PYTHONPATH', '')}"
        action_log = (run_dir / "action_server.log").open("w")
        action_cmd = [
            aa["vla_python"],
            str(vla_repo / "deployment" / "model_server" / "server_policy.py"),
            "--ckpt_path",
            str(ckpt_path),
            "--port",
            str(args.action_port),
            "--use_bf16",
        ]
        bg_procs.append(subprocess.Popen(action_cmd, cwd=str(vla_repo), env=action_env, stdout=action_log, stderr=subprocess.STDOUT))
        print(f"[run_eval] Waiting for ActiveArena-VLA action server on 127.0.0.1:{args.action_port}...", flush=True)
        wait_for_tcp("127.0.0.1", args.action_port, timeout_s=240.0)

        # 2. If running autonomous IP-VLA planner mode, start Planner HTTP Server
        if args.mode == "ipvla_planner":
            planner_env = os.environ.copy()
            planner_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            planner_env["HF_HUB_OFFLINE"] = "1"
            planner_env["PYTHONPATH"] = f"{REPO_ROOT}:{planner_env.get('PYTHONPATH', '')}"
            planner_log = (run_dir / "planner_server.log").open("w")
            planner_cmd = [
                aa["planner_python"],
                "-m",
                "ipvla.stage2.planner_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.planner_port),
                "--base-vlm",
                paths["base_vlm_dir"],
                "--lora",
                paths["stage1_lora_path"],
                "--confidence-head",
                paths["stage1_confidence_head_path"],
                "--tag",
                "base_qwen3_vl_2b" if args.no_lora else "stage1_v2_ep3",
            ]
            if args.no_lora:
                planner_cmd.append("--no-lora")
            bg_procs.append(subprocess.Popen(planner_cmd, cwd=str(REPO_ROOT), env=planner_env, stdout=planner_log, stderr=subprocess.STDOUT))
            print(f"[run_eval] Waiting for IP-VLA planner server on 127.0.0.1:{args.planner_port}...", flush=True)
            wait_for_http(f"http://127.0.0.1:{args.planner_port}/health", timeout_s=180.0)

        # 3. Run ActiveArena official `script/eval_policy.py` per task on identical paired seeds
        results: list[dict[str, Any]] = []
        policy_module = (
            "eval.activearena.autonomous_policy"
            if args.mode == "ipvla_planner"
            else f"policy.{model_name}.deploy_policy"
        )

        for task_name in args.tasks:
            task_out = run_dir / "tasks" / task_name
            task_out.mkdir(parents=True, exist_ok=True)
            sim_env = os.environ.copy()
            sim_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            sim_env["PYTHONWARNINGS"] = "ignore::UserWarning"
            sim_env["PYTHONPATH"] = f"{REPO_ROOT}:{sim_repo}:{sim_env.get('PYTHONPATH', '')}"
            sim_env["ACTIVEARENA_EVAL_SEED_LIST"] = str(sim_repo / "eval_seed_lists" / setting_name / f"{task_name}.json")
            sim_env["IPVLA_PLANNER_URL"] = f"http://127.0.0.1:{args.planner_port}/plan"
            sim_env["IPVLA_CEREBELLUM_MODE"] = args.cerebellum_mode
            sim_env["IPVLA_MEMORY_MODE"] = args.memory_mode
            sim_env["IPVLA_TRACE_LOG_PATH"] = str(task_out / "ipvla_switching_trace.jsonl")

            eval_cmd = [
                aa["sim_python"],
                str(sim_repo / "script" / "eval_policy.py"),
                "--config",
                str(sim_repo / "policy" / model_name / "deploy_policy.yml"),
                "--overrides",
                "--task_name",
                task_name,
                "--task_config",
                setting_name,
                "--ckpt_setting",
                f"{args.mode}_{args.cerebellum_mode}",
                "--seed",
                "0",
                "--policy_name",
                policy_module,
                "--port",
                str(args.action_port),
                "--test_num",
                str(args.episodes),
            ]
            t0 = time.monotonic()
            proc = subprocess.run(
                eval_cmd,
                cwd=str(sim_repo),
                env=sim_env,
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - t0
            (task_out / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (task_out / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            results.append(
                {
                    "task": task_name,
                    "returncode": proc.returncode,
                    "elapsed_sec": round(elapsed, 2),
                    "trace_path": str(task_out / "ipvla_switching_trace.jsonl"),
                }
            )
            print(f"[run_eval] Task={task_name} rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)

        summary = {
            "run_id": run_id,
            "mode": args.mode,
            "setting": setting_name,
            "episodes": args.episodes,
            "cerebellum_mode": args.cerebellum_mode,
            "memory_mode": args.memory_mode,
            "results": results,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
