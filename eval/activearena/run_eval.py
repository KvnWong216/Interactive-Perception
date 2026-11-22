#!/usr/bin/env python3
"""Unified ActiveArena Closed-Loop Evaluation Launcher for IP-VLA, ASA, & ZS-IP.

Supports all evaluation arms on identical paired ActiveArena seeds:
  1. `asa_baseline`       : Upstream ASA (`EgoActiveFlowModelv2`) aligned to ActiveArena's `Astribot S1`
                            whole-body kinematics (head camera + pan/tilt + waist yaw + dual 7-DoF arms + grippers)
  2. `zsip_baseline`      : Upstream ZS-IP (`arXiv:2602.18374`, Sripada et al. 2026) — 100% frozen
                            `Qwen3-VL-2B-Instruct` (0 training) + Enhanced Observation + Memory Lookup + Primitives
  3. `instruction_oft`    : Official `oft_instruction_action_12_ws` end-to-end baseline
  4. `oracle_subtask_oft` : Official `oft_subtask_action_12_ws` + Privileged Env Oracle (upper bound)
  5. `ipvla_planner`      : Autonomous IP-VLA Stage-2 VLM Planner + Continuous Cerebellum Inertia
                            + Inspect-Keyframe Memory -> Frozen `oft_subtask_action_12_ws` (0 Oracle!)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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
    "asa_baseline": "asa_human_stage2",
    "zsip_baseline": "zsip_qwen3_vl_2b_frozen",
    "aawr_baseline": "aawr_astribot_asym",
    "instruction_oft": "oft_instruction_action_12_ws",
    "oracle_subtask_oft": "oft_subtask_action_12_ws",
    "ipvla_planner": "oft_subtask_action_12_ws",
}

EXPECTED_OFT_CKPT_BYTES = 4_967_033_387
BASELINE_STUDY_ROOT = Path("/home/ubuntu/ygwang/IP-VLA/baseline-study")
DEFAULT_ASA_REPO = Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration")
DEFAULT_ASA_PYTHON = DEFAULT_ASA_REPO / ".venv" / "bin" / "python"
DEFAULT_ASA_CKPT = DEFAULT_ASA_REPO / "checkpoints" / "asa-human-stage2"
DEFAULT_AAWR_CKPT = REPO_ROOT / "runs" / "aawr_astribot" / "ckpt_v1" / "best"
FFMPEG_BIN_DIR = Path("/home/ubuntu/miniforge3/envs/ygbench/bin")


def check_readiness(
    paths: dict[str, Any],
    mode: str,
    tasks: list[str],
    setting_name: str,
    asa_ckpt: Path | None = None,
    aawr_ckpt: Path | None = None,
) -> dict[str, Any]:
    aa = paths["activearena"]
    sim_repo = Path(aa["sim_repo"]).resolve()
    vla_repo = Path(aa["vla_repo"]).resolve()
    weights_root = Path(aa["weights_root"]).resolve()
    model_name = MODE_TO_MODEL[mode]

    needs_bg_textures = setting_name != "info_gathering_demo"
    bg_present = (sim_repo / "assets" / "background_texture").is_dir()

    checks: dict[str, Any] = {
        "sim_repo_exists": sim_repo.is_dir(),
        "assets_embodiments_exists": (sim_repo / "assets" / "embodiments" / "astribot_descriptions_texture").is_dir(),
        "assets_objects_exists": (sim_repo / "assets" / "objects" / "005_button").is_dir(),
        "assets_bg_present": bg_present,
        "assets_bg_required": needs_bg_textures,
        "assets_bg_exists": bg_present if needs_bg_textures else True,
        "base_vlm_exists": Path(paths["base_vlm_dir"]).is_dir(),
    }

    if mode == "asa_baseline":
        ckpt_dir = (asa_ckpt or DEFAULT_ASA_CKPT).resolve()
        checks["asa_python_exists"] = DEFAULT_ASA_PYTHON.is_file()
        checks["asa_ckpt_exists"] = (ckpt_dir / "config.json").is_file() and (
            (ckpt_dir / "model.safetensors.index.json").is_file() or (ckpt_dir / "model.safetensors").is_file()
        )
        checks["asa_ckpt_path"] = str(ckpt_dir)
    elif mode == "zsip_baseline":
        # ZS-IP requires ZERO finetuning checkpoints — only the frozen pretrained base VLM!
        checks["zsip_python_exists"] = DEFAULT_ASA_PYTHON.is_file()
        checks["zsip_training_steps"] = 0
    elif mode == "aawr_baseline":
        ckpt_dir = (aawr_ckpt or DEFAULT_AAWR_CKPT).resolve()
        checks["aawr_ckpt_exists"] = (ckpt_dir / "ckpt.pt").is_file() and (ckpt_dir / "aawr_norm_stats.json").is_file()
        checks["aawr_ckpt_path"] = str(ckpt_dir)
    else:
        ckpt_path = weights_root / model_name / "checkpoints" / "steps_100000_pytorch_model.pt"
        ckpt_bytes = ckpt_path.stat().st_size if ckpt_path.is_file() else 0
        checks["vla_repo_exists"] = vla_repo.is_dir()
        checks["action_ckpt_exists"] = ckpt_bytes == EXPECTED_OFT_CKPT_BYTES
        checks["action_ckpt_path"] = str(ckpt_path)
        checks["action_ckpt_bytes"] = ckpt_bytes
        checks["action_ckpt_expected_bytes"] = EXPECTED_OFT_CKPT_BYTES

    if mode == "ipvla_planner":
        lora = Path(paths["stage1_lora_path"])
        conf = Path(paths["stage1_confidence_head_path"])
        checks["stage1_lora_exists"] = lora.is_file()
        checks["stage1_conf_head_exists"] = conf.is_file()

    seed_dir = sim_repo / "eval_seed_lists" / setting_name
    missing_tasks = [t for t in tasks if not (seed_dir / f"{t}.json").is_file()]
    checks["seed_dir_exists"] = seed_dir.is_dir()
    checks["missing_task_seed_lists"] = missing_tasks

    ready = all(v for k, v in checks.items() if k.endswith("_exists")) and not missing_tasks
    return {"ready": ready, "mode": mode, "model_name": model_name, "setting": setting_name, "checks": checks}


def _apply_local_toolchain_env(
    env: dict[str, str], sim_repo: Path, vla_repo: Path, weights_root: Path, base_vlm: str
) -> None:
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    env["ACTIVEARENA_ROOT"] = str(sim_repo)
    env["ACTIVEARENA_VLA_ROOT"] = str(vla_repo)
    env["ACTIVEARENA_VLA_WEIGHTS_ROOT"] = str(weights_root)
    env["ACTIVEARENA_VLA_BASE_VLM"] = str(base_vlm)
    if FFMPEG_BIN_DIR.is_dir():
        env["PATH"] = f"{FFMPEG_BIN_DIR}:{env.get('PATH', '')}"
    if BASELINE_STUDY_ROOT.is_dir():
        gcc = BASELINE_STUDY_ROOT / ".venvs" / "build-tools" / "bin" / "x86_64-conda-linux-gnu-gcc"
        gxx = BASELINE_STUDY_ROOT / ".venvs" / "build-tools" / "bin" / "x86_64-conda-linux-gnu-g++"
        cuda_dir = BASELINE_STUDY_ROOT / "env" / "cuda"
        cxx_rt = BASELINE_STUDY_ROOT / "env" / "cxx-runtime"
        if gcc.is_file():
            env["CC"] = str(gcc)
        if gxx.is_file():
            env["CXX"] = str(gxx)
        if cuda_dir.is_dir():
            env["CUDA_HOME"] = str(cuda_dir)
            env["ACTIVEARENA_CUDA_HOME"] = str(cuda_dir)
            env["ROBOTWIN_CUDA_HOME"] = str(cuda_dir)
        if cxx_rt.is_dir():
            prev_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{cxx_rt}:{prev_ld}" if prev_ld else str(cxx_rt)


def wait_for_http(url: str, proc: subprocess.Popen | None = None, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    sess = requests.Session()
    sess.trust_env = False
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server at {url} exited prematurely with code {proc.returncode}")
        try:
            r = sess.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Service did not become ready at {url} within {timeout_s:.0f}s")


def wait_for_tcp(host: str, port: int, proc: subprocess.Popen | None = None, timeout_s: float = 240.0) -> None:
    import socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Action server exited prematurely with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return
        except OSError:
            time.sleep(1.5)
    raise TimeoutError(f"Action server did not open {host}:{port} within {timeout_s:.0f}s")


def _parse_success_rate(stdout_text: str, result_dir: Path | None = None) -> dict[str, Any]:
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout_text)
    matches = re.findall(r"Success rate:\s*(\d+)\s*/\s*(\d+)", clean)
    if matches:
        succ, total = int(matches[-1][0]), int(matches[-1][1])
        return {
            "successes": succ,
            "completed": total,
            "success_rate": round(succ / total, 4) if total > 0 else None,
        }
    if result_dir is not None:
        res_files = sorted(result_dir.rglob("_result.txt"))
        if res_files:
            try:
                val = float(res_files[-1].read_text(encoding="utf-8").strip().splitlines()[-1])
                return {"successes": None, "completed": None, "success_rate": round(val, 4)}
            except Exception:
                pass
    return {"successes": None, "completed": None, "success_rate": None}


def main() -> None:
    paths = load_paths_config()
    stage2_cfg = load_stage2_config()
    eval_defaults = stage2_cfg.get("eval", {})

    parser = argparse.ArgumentParser(description="IP-VLA, ASA, ZS-IP, & AAWR Unified ActiveArena Evaluator")
    parser.add_argument(
        "--mode",
        choices=("asa_baseline", "zsip_baseline", "aawr_baseline", "instruction_oft", "oracle_subtask_oft", "ipvla_planner"),
        required=True,
    )
    parser.add_argument("--setting", default=eval_defaults.get("default_setting", "id"))
    parser.add_argument("--tasks", nargs="+", default=eval_defaults.get("pilot_tasks", ["check_block_color"]))
    parser.add_argument("--episodes", type=int, default=int(eval_defaults.get("default_episodes", 10)))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--action-port", type=int, default=int(eval_defaults.get("action_server_base_port", 19000)))
    parser.add_argument("--planner-port", type=int, default=int(eval_defaults.get("planner_server_port", 19100)))
    parser.add_argument("--asa-port", type=int, default=19200)
    parser.add_argument("--zsip-port", type=int, default=19300)
    parser.add_argument("--zsip-max-turns", type=int, default=12)
    parser.add_argument("--aawr-port", type=int, default=19400)
    parser.add_argument("--aawr-ckpt", type=Path, default=DEFAULT_AAWR_CKPT)
    parser.add_argument("--aawr-exec-horizon", type=int, default=4)
    parser.add_argument("--asa-ckpt", type=Path, default=DEFAULT_ASA_CKPT)
    parser.add_argument("--asa-control-mode", choices=("ik_qpos", "curobo_ee"), default="ik_qpos")
    parser.add_argument("--asa-subtask-mode", choices=("instruction", "oracle_subtask"), default="instruction")
    parser.add_argument(
        "--asa-norm-profile",
        choices=("astribot_human_aligned", "holoassist", "monte02_step"),
        default="astribot_human_aligned",
    )
    parser.add_argument("--asa-exec-horizon", type=int, default=6)
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
    readiness = check_readiness(
        paths, args.mode, args.tasks, setting_name, asa_ckpt=args.asa_ckpt, aawr_ckpt=args.aawr_ckpt
    )
    if args.dry_run:
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        sys.exit(0 if readiness["ready"] else 2)

    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit("ActiveArena preflight check failed; see readiness report above.")

    aa = paths["activearena"]
    sim_repo = Path(aa["sim_repo"]).resolve()
    vla_repo = Path(aa["vla_repo"]).resolve()
    weights_root = Path(aa["weights_root"]).resolve()
    model_name = MODE_TO_MODEL[args.mode]

    if args.mode == "asa_baseline":
        tag = f"asa_{args.asa_control_mode}_{args.asa_subtask_mode}"
    elif args.mode == "zsip_baseline":
        tag = "zsip_frozen_qwen3vl2b"
    elif args.mode == "aawr_baseline":
        tag = "aawr_asym_unpriv"
    else:
        tag = f"{args.mode}_{args.cerebellum_mode}_{args.memory_mode}"
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    run_dir = (REPO_ROOT / aa["output_root"] / setting_name / run_id).resolve()
    eval_result_root = run_dir / "eval_result"
    warp_cache_path = run_dir / "warp_cache"
    torch_ext_dir = sim_repo / "logs" / "torch_extensions"
    for d in (run_dir, eval_result_root, warp_cache_path, torch_ext_dir):
        d.mkdir(parents=True, exist_ok=True)

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
        if args.mode == "asa_baseline":
            asa_env = os.environ.copy()
            asa_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            asa_env["HF_HUB_OFFLINE"] = "1"
            asa_env["TOKENIZERS_PARALLELISM"] = "false"
            asa_env["PYTHONPATH"] = f"{REPO_ROOT}:{DEFAULT_ASA_REPO}:{asa_env.get('PYTHONPATH', '')}"
            asa_log = (run_dir / "asa_server.log").open("w")
            asa_cmd = [
                str(DEFAULT_ASA_PYTHON),
                "-m",
                "eval.activearena.asa_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.asa_port),
                "--asa-repo",
                str(DEFAULT_ASA_REPO),
                "--asa-ckpt",
                str(args.asa_ckpt.resolve()),
                "--base-vlm",
                paths["base_vlm_dir"],
            ]
            asa_proc = subprocess.Popen(asa_cmd, cwd=str(REPO_ROOT), env=asa_env, stdout=asa_log, stderr=subprocess.STDOUT)
            bg_procs.append(asa_proc)
            print(f"[run_eval] Waiting for ASA Flow Server on 127.0.0.1:{args.asa_port}...", flush=True)
            wait_for_http(f"http://127.0.0.1:{args.asa_port}/health", proc=asa_proc, timeout_s=240.0)
        elif args.mode == "zsip_baseline":
            zsip_env = os.environ.copy()
            zsip_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            zsip_env["HF_HUB_OFFLINE"] = "1"
            zsip_env["TOKENIZERS_PARALLELISM"] = "false"
            zsip_env["PYTHONPATH"] = f"{REPO_ROOT}:{zsip_env.get('PYTHONPATH', '')}"
            zsip_log = (run_dir / "zsip_server.log").open("w")
            zsip_cmd = [
                str(DEFAULT_ASA_PYTHON),
                "-m",
                "eval.activearena.zsip_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.zsip_port),
                "--base-vlm",
                paths["base_vlm_dir"],
            ]
            zsip_proc = subprocess.Popen(
                zsip_cmd, cwd=str(REPO_ROOT), env=zsip_env, stdout=zsip_log, stderr=subprocess.STDOUT
            )
            bg_procs.append(zsip_proc)
            print(f"[run_eval] Waiting for ZS-IP Frozen VLM Server on 127.0.0.1:{args.zsip_port}...", flush=True)
            wait_for_http(f"http://127.0.0.1:{args.zsip_port}/health", proc=zsip_proc, timeout_s=240.0)
        elif args.mode == "aawr_baseline":
            aawr_env = os.environ.copy()
            aawr_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            aawr_env["PYTHONPATH"] = f"{REPO_ROOT}:{aawr_env.get('PYTHONPATH', '')}"
            aawr_log = (run_dir / "aawr_server.log").open("w")
            aawr_cmd = [
                aa["planner_python"],
                "-m",
                "eval.activearena.aawr_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.aawr_port),
                "--aawr-ckpt",
                str(args.aawr_ckpt.resolve()),
            ]
            aawr_proc = subprocess.Popen(
                aawr_cmd, cwd=str(REPO_ROOT), env=aawr_env, stdout=aawr_log, stderr=subprocess.STDOUT
            )
            bg_procs.append(aawr_proc)
            print(f"[run_eval] Waiting for AAWR Unprivileged Policy Server on 127.0.0.1:{args.aawr_port}...", flush=True)
            wait_for_http(f"http://127.0.0.1:{args.aawr_port}/health", proc=aawr_proc, timeout_s=120.0)
        else:
            ckpt_path = weights_root / model_name / "checkpoints" / "steps_100000_pytorch_model.pt"
            action_env = os.environ.copy()
            _apply_local_toolchain_env(action_env, sim_repo, vla_repo, weights_root, paths["base_vlm_dir"])
            action_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            action_env["HF_HUB_OFFLINE"] = "1"
            action_env["RUN_OUTPUT"] = str(weights_root / model_name)
            action_env["PYTHONPATH"] = f"{vla_repo}:{action_env.get('PYTHONPATH', '')}"
            action_log = (run_dir / "action_server.log").open("w")
            action_cmd = [
                aa["vla_python"],
                str(vla_repo / "deployment" / "model_server" / "server_policy.py"),
                "--ckpt_path",
                str(ckpt_path),
                "--port",
                str(args.action_port),
                "--idle_timeout",
                "-1",
                "--use_bf16",
            ]
            action_proc = subprocess.Popen(
                action_cmd, cwd=str(vla_repo), env=action_env, stdout=action_log, stderr=subprocess.STDOUT
            )
            bg_procs.append(action_proc)
            print(f"[run_eval] Waiting for ActiveArena-VLA action server on 127.0.0.1:{args.action_port}...", flush=True)
            wait_for_tcp("127.0.0.1", args.action_port, proc=action_proc, timeout_s=300.0)

        if args.mode == "ipvla_planner":
            planner_env = os.environ.copy()
            planner_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            planner_env["HF_HUB_OFFLINE"] = "1"
            planner_env["TOKENIZERS_PARALLELISM"] = "false"
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
            p_proc = subprocess.Popen(
                planner_cmd, cwd=str(REPO_ROOT), env=planner_env, stdout=planner_log, stderr=subprocess.STDOUT
            )
            bg_procs.append(p_proc)
            print(f"[run_eval] Waiting for IP-VLA planner server on 127.0.0.1:{args.planner_port}...", flush=True)
            wait_for_http(f"http://127.0.0.1:{args.planner_port}/health", proc=p_proc, timeout_s=180.0)

        results: list[dict[str, Any]] = []
        if args.mode == "asa_baseline":
            policy_module = "eval.activearena.asa_policy"
            deploy_yml = sim_repo / "policy" / "oft_instruction_action_12_ws" / "deploy_policy.yml"
        elif args.mode == "zsip_baseline":
            policy_module = "eval.activearena.zsip_policy"
            deploy_yml = sim_repo / "policy" / "oft_instruction_action_12_ws" / "deploy_policy.yml"
        elif args.mode == "aawr_baseline":
            policy_module = "eval.activearena.aawr_policy"
            deploy_yml = sim_repo / "policy" / "oft_instruction_action_12_ws" / "deploy_policy.yml"
        elif args.mode == "ipvla_planner":
            policy_module = "eval.activearena.autonomous_policy"
            deploy_yml = sim_repo / "policy" / model_name / "deploy_policy.yml"
        else:
            policy_module = f"policy.{model_name}.deploy_policy"
            deploy_yml = sim_repo / "policy" / model_name / "deploy_policy.yml"

        curobo_src = sim_repo / "envs" / "curobo" / "src"

        for task_name in args.tasks:
            task_out = run_dir / "tasks" / task_name
            task_out.mkdir(parents=True, exist_ok=True)
            sim_env = os.environ.copy()
            _apply_local_toolchain_env(sim_env, sim_repo, vla_repo, weights_root, paths["base_vlm_dir"])
            sim_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            sim_env["PYTHONWARNINGS"] = "ignore::UserWarning"
            sim_env["PYTHONUNBUFFERED"] = "1"
            sim_env["ACTIVEARENA_CUROBO_SRC"] = str(curobo_src)
            sim_env["ROBOTWIN_CUROBO_SRC"] = str(curobo_src)
            sim_env["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)
            sim_env["WARP_CACHE_PATH"] = str(warp_cache_path)
            sim_env["PYTHONPATH"] = f"{REPO_ROOT}:{vla_repo}:{curobo_src}:{sim_repo}:{sim_env.get('PYTHONPATH', '')}"

            seed_root = str(sim_repo / "eval_seed_lists")
            sim_env["ACTIVEARENA_USE_EVAL_SEED_LIST"] = "1"
            sim_env["ROBOTWIN_USE_EVAL_SEED_LIST"] = "1"
            sim_env["ACTIVEARENA_EVAL_SEED_LIST_PATH"] = seed_root
            sim_env["ROBOTWIN_EVAL_SEED_LIST_PATH"] = seed_root
            sim_env["ACTIVEARENA_EVAL_TEST_NUM"] = str(args.episodes)
            sim_env["ROBOTWIN_EVAL_TEST_NUM"] = str(args.episodes)
            sim_env["ACTIVEARENA_EVAL_RESULT_ROOT"] = str(eval_result_root)
            sim_env["ROBOTWIN_EVAL_RESULT_ROOT"] = str(eval_result_root)
            sim_env["ACTIVEARENA_REQUEST_LOG_PATH"] = str(task_out / "requests.jsonl")
            sim_env["ROBOTWIN_REQUEST_LOG_PATH"] = str(task_out / "requests.jsonl")
            sim_env["ACTIVEARENA_REQUEST_IMAGE_DIR"] = ""
            sim_env["ROBOTWIN_REQUEST_IMAGE_DIR"] = ""
            sim_env["ACTIVEARENA_POLICY_HOST"] = "127.0.0.1"
            sim_env["ACTIVEARENA_POLICY_PORT"] = str(args.action_port)
            sim_env["STARVLA_HOST"] = "127.0.0.1"
            sim_env["STARVLA_PORT"] = str(args.action_port)

            sim_env["IPVLA_PLANNER_URL"] = f"http://127.0.0.1:{args.planner_port}/plan"
            sim_env["IPVLA_CEREBELLUM_MODE"] = args.cerebellum_mode
            sim_env["IPVLA_MEMORY_MODE"] = args.memory_mode
            sim_env["IPVLA_TRACE_LOG_PATH"] = str(task_out / "ipvla_switching_trace.jsonl")

            sim_env["ASA_SERVER_URL"] = f"http://127.0.0.1:{args.asa_port}/act"
            sim_env["ASA_CKPT_DIR"] = str(args.asa_ckpt.resolve())
            sim_env["ASA_REPO_ROOT"] = str(DEFAULT_ASA_REPO)
            sim_env["ASA_CONTROL_MODE"] = args.asa_control_mode
            sim_env["ASA_SUBTASK_MODE"] = args.asa_subtask_mode
            sim_env["ASA_NORM_PROFILE"] = args.asa_norm_profile
            sim_env["ASA_EXEC_HORIZON"] = str(args.asa_exec_horizon)
            sim_env["ASA_TRACE_LOG_PATH"] = str(task_out / "asa_embodiment_trace.jsonl")

            sim_env["ZSIP_SERVER_URL"] = f"http://127.0.0.1:{args.zsip_port}/decide"
            sim_env["ZSIP_MAX_TURNS"] = str(args.zsip_max_turns)
            sim_env["ZSIP_TRACE_LOG_PATH"] = str(task_out / "zsip_trace.jsonl")

            sim_env["AAWR_SERVER_URL"] = f"http://127.0.0.1:{args.aawr_port}/act"
            sim_env["AAWR_EXEC_HORIZON"] = str(args.aawr_exec_horizon)
            sim_env["AAWR_TRACE_LOG_PATH"] = str(task_out / "aawr_trace.jsonl")

            if args.mode == "asa_baseline":
                active_port = args.asa_port
                trace_rel = "asa_embodiment_trace.jsonl"
            elif args.mode == "zsip_baseline":
                active_port = args.zsip_port
                trace_rel = "zsip_trace.jsonl"
            elif args.mode == "aawr_baseline":
                active_port = args.aawr_port
                trace_rel = "aawr_trace.jsonl"
            else:
                active_port = args.action_port
                trace_rel = "ipvla_switching_trace.jsonl"

            eval_cmd = [
                aa["sim_python"],
                str(sim_repo / "script" / "eval_policy.py"),
                "--config",
                str(deploy_yml),
                "--overrides",
                "--task_name",
                task_name,
                "--task_config",
                setting_name,
                "--ckpt_setting",
                tag,
                "--seed",
                "0",
                "--policy_name",
                policy_module,
                "--port",
                str(active_port),
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
            metrics = _parse_success_rate(proc.stdout, eval_result_root / task_name)
            results.append(
                {
                    "task": task_name,
                    "returncode": proc.returncode,
                    "elapsed_sec": round(elapsed, 2),
                    **metrics,
                    "trace_path": str(task_out / trace_rel),
                }
            )
            print(
                f"[run_eval] Task={task_name} rc={proc.returncode} "
                f"success_rate={metrics['success_rate']} elapsed={elapsed:.1f}s",
                flush=True,
            )

        summary = {
            "run_id": run_id,
            "mode": args.mode,
            "model_name": model_name,
            "setting": setting_name,
            "episodes": args.episodes,
            "results": results,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
