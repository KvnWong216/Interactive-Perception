#!/usr/bin/env python3
"""GPU-Safe Sequential Master Orchestrator for IP-VLA Baselines, Stage-1 v3 & Stage-2.

Executes strictly ONE GPU-intensive job at a time on GPU 0 (waiting cleanly for any
active job such as `asa-stage3-formal.service` to finish first):

1. Wait for `asa-stage3-formal.service` (ASA baseline reproduction) to finish & GPU 0 free >= 19 GiB.
2. As soon as `oft_subtask_action_12_ws` (4.63 GB) is ready:
   - Run Official `oracle_subtask_oft` upper-bound baseline on ActiveArena ID (`check_block_color`,
     `check_cola_date`, `check_cola_color`, 10 episodes/task).
   - Run Autonomous `ipvla_planner` (Stage-1 v2, zero-oracle) on the exact same ActiveArena ID seeds.
3. If `simplememvla_rmbench` checkpoint finished downloading via ModelScope:
   - Run `SimpleMemVLA` closed-loop baseline reproduction on RMBench (`observe_and_pickup`,
     `put_back_block`, `cover_blocks`, 10 seeds/task).
4. Wait until Teacher B finishes all remaining open-video causal checks (`queued == 0` and
   `inflight == 0`), rebuild the Full-17k `stage1_formal_suite.jsonl` in `rss2027-local-pilot`,
   and run **Stage-1 v3 Full-Scale Rebalanced 6-Epoch Training** (`orx exp run 33ccb80d-...`).
5. Once `oft_instruction_action_12_ws` & `background_texture` are ready, run Official
   `instruction_oft` end-to-end baseline + OOD closed-loop evaluation (with upgraded Stage-1 v3).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


IPVLA_ROOT = Path("/home/ubuntu/ygwang/IP-VLA/ip-vla")
BS_ROOT = Path("/home/ubuntu/ygwang/IP-VLA/baseline-study")
PILOT_WORKTREE = Path("/home/ubuntu/ygwang/IP-VLA/rss2027-local-pilot")
AUTORESEARCH = Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/reports/rss2027_autoresearch")
ROLE_B_PROGRESS = AUTORESEARCH / "data/teacher_chunks/full_sol_v1/role_b_progress.json"
V2_RESULTS_DIR = AUTORESEARCH / "results/stage1_v2_rebalanced_6ep_r32_448x336"
STAGE1_V2_ORX_EXP_ID = "33ccb80d-7688-4118-bdcc-812a594b4115"

YGB_PY = "/home/ubuntu/miniforge3/envs/ygbench/bin/python"
RMB_PY = str(BS_ROOT / ".venvs/simplememvla-rmbench/bin/python")
STATUS_PATH = IPVLA_ROOT / "runs/orchestrator_status.json"


def write_status(phase: str, detail: dict[str, Any] | None = None) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S JST"),
        "phase": phase,
        "detail": detail or {},
    }
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(STATUS_PATH)
    print(f"[orchestrator] {phase} | {json.dumps(detail or {}, ensure_ascii=False)}", flush=True)


def gpu0_free_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return 0


def unit_is_active(unit_name: str) -> bool:
    res = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit_name],
        capture_output=True,
    )
    return res.returncode == 0


def wait_for_gpu0_idle(reason: str, min_free_mib: int = 19000) -> None:
    while True:
        asa_active = unit_is_active("asa-stage3-formal.service")
        free_mib = gpu0_free_mib()
        if not asa_active and free_mib >= min_free_mib:
            return
        write_status(
            f"waiting_for_gpu0 ({reason})",
            {"asa_stage3_active": asa_active, "gpu0_free_mib": free_mib, "need_mib": min_free_mib},
        )
        time.sleep(30)


def ckpt_ready(model_name: str) -> bool:
    p = BS_ROOT / "checkpoints/ActiveArena-VLA" / model_name / "checkpoints/steps_100000_pytorch_model.pt"
    return p.is_file() and p.stat().st_size >= 4_960_000_000


def run_activearena_eval(mode: str, setting: str, tasks: list[str], episodes: int = 10, extra: list[str] | None = None) -> int:
    wait_for_gpu0_idle(f"activearena_{mode}_{setting}")
    write_status(f"running_activearena_{mode}_{setting}", {"tasks": tasks, "episodes": episodes})
    cmd = [
        YGB_PY, "-u", "-m", "eval.activearena.run_eval",
        "--mode", mode,
        "--setting", setting,
        "--tasks", *tasks,
        "--episodes", str(episodes),
        "--gpu", "0",
        *(extra or []),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{IPVLA_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=str(IPVLA_ROOT), env=env)
    write_status(f"finished_activearena_{mode}_{setting}", {"returncode": proc.returncode})
    return proc.returncode


def run_simplememvla_rmbench_pilot() -> None:
    ckpt_dir = BS_ROOT / "checkpoints/simplememvla_rmbench"
    if not (ckpt_dir / "model.safetensors").is_file() and not (ckpt_dir / "model.safetensors.index.json").is_file():
        write_status("skipping_simplememvla_rmbench_for_now", {"reason": "checkpoint_still_downloading"})
        return
    wait_for_gpu0_idle("simplememvla_rmbench")
    sm_repo = BS_ROOT / "external/simplememvla"
    write_status("running_simplememvla_rmbench_baseline", {"tasks": ["observe_and_pickup", "put_back_block", "cover_blocks"]})
    env = os.environ.copy()
    env.update({
        "CHECKPOINT": str(ckpt_dir),
        "TASKS": "observe_and_pickup put_back_block cover_blocks",
        "SEEDS_PER_TASK": "10",
        "GROUP_SIZE": "2",
        "NUM_GPUS": "1",
        "GPUS": "0",
        "HF_HUB_OFFLINE": "1",
    })
    proc = subprocess.run(["bash", "scripts/eval_rmbench.sh"], cwd=str(sm_repo), env=env)
    write_status("finished_simplememvla_rmbench_baseline", {"returncode": proc.returncode})


def wait_for_teacher_b_and_train_stage1_v3() -> Path | None:
    while True:
        try:
            prog = json.loads(ROLE_B_PROGRESS.read_text())["by_status"]
        except Exception:
            prog = {"queued": -1, "inflight": -1}
        if prog.get("queued", 1) == 0 and prog.get("inflight", 1) == 0:
            break
        write_status("waiting_for_teacher_b_completion", prog)
        time.sleep(45)

    wait_for_gpu0_idle("stage1_v3_full_17k_training")
    write_status("rebuilding_stage1_v3_formal_suite", {})
    build_script = PILOT_WORKTREE / "research/rss2027_local/build_stage1_formal_suite.py"
    subprocess.run([YGB_PY, str(build_script)], cwd=str(PILOT_WORKTREE), check=True)
    subprocess.run(
        ["git", "add", "research/rss2027_local/stage1_formal_suite.jsonl", "research/rss2027_local/stage1_formal_summary.json"],
        cwd=str(PILOT_WORKTREE), check=True,
    )
    diff = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(PILOT_WORKTREE), capture_output=True, text=True, check=True
    ).stdout.strip()
    if diff:
        subprocess.run(
            ["git", "-c", "user.name=IP-VLA Orchestrator", "-c", "user.email=ipvla@local",
             "commit", "-m", "Build full-17k Stage-1 v3 formal suite after Teacher B completion"],
            cwd=str(PILOT_WORKTREE), check=True,
        )

    existing_runs = {d.name for d in V2_RESULTS_DIR.iterdir() if d.is_dir()} if V2_RESULTS_DIR.is_dir() else set()
    write_status("launching_stage1_v3_orx_run", {"exp_id": STAGE1_V2_ORX_EXP_ID})
    subprocess.run(["/home/ubuntu/.local/bin/orx", "exp", "run", STAGE1_V2_ORX_EXP_ID, "--backend", "local"], cwd=str(PILOT_WORKTREE), check=True)

    # Monitor newly created run directory until receipt status == completed / failed
    new_run_dir: Path | None = None
    while True:
        time.sleep(30)
        if V2_RESULTS_DIR.is_dir():
            for d in sorted(V2_RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime):
                if d.is_dir() and d.name not in existing_runs:
                    new_run_dir = d
        if new_run_dir and (new_run_dir / "receipt.json").is_file():
            rec = json.loads((new_run_dir / "receipt.json").read_text())
            st = rec.get("status", "running")
            write_status("monitoring_stage1_v3_training", {"run_dir": str(new_run_dir), "status": st, "steps": rec.get("train_steps")})
            if st in ("completed", "failed", "stopped_by_budget"):
                return new_run_dir if st == "completed" else None


def main() -> None:
    pilot_tasks = ["check_block_color", "check_cola_date", "check_cola_color"]

    # Step 1: Wait for ASA Stage-3 baseline pipeline to finish cleanly on GPU 0
    wait_for_gpu0_idle("initial_handshake")

    # Step 2: Wait for oft_subtask_action_12_ws (almost 100% done) and run ID baselines + Stage-1 v2 autonomous!
    while not ckpt_ready("oft_subtask_action_12_ws"):
        write_status("waiting_for_oft_subtask_checkpoint", {})
        time.sleep(30)

    run_activearena_eval("oracle_subtask_oft", "id", pilot_tasks, episodes=10)
    run_activearena_eval("ipvla_planner", "id", pilot_tasks, episodes=10)

    # Step 3: Run SimpleMemVLA RMBench baseline if ModelScope download completed
    run_simplememvla_rmbench_pilot()

    # Step 4: Wait for Teacher B 100% completion -> Full-17k Stage-1 v3 6-epoch training
    v3_dir = wait_for_teacher_b_and_train_stage1_v3()

    # Step 5: Once oft_instruction_action_12_ws & background_texture are ready, run Instruction baseline + OOD eval
    bg_dir = BS_ROOT / "external/activearena/assets/background_texture"
    while not ckpt_ready("oft_instruction_action_12_ws"):
        write_status("waiting_for_oft_instruction_checkpoint", {})
        time.sleep(45)
    run_activearena_eval("instruction_oft", "id", pilot_tasks, episodes=10)

    if bg_dir.is_dir():
        run_activearena_eval("instruction_oft", "ood", pilot_tasks, episodes=10)
        run_activearena_eval("oracle_subtask_oft", "ood", pilot_tasks, episodes=10)
        if v3_dir:
            os.environ["IPVLA_STAGE1_LORA"] = str(v3_dir / "stage1_lora_r32_448x336_best.safetensors")
            os.environ["IPVLA_STAGE1_CONF_HEAD"] = str(v3_dir / "stage1_v2_confidence_head_ridge_temp.pt")
        run_activearena_eval("ipvla_planner", "ood", pilot_tasks, episodes=10)

    write_status("all_orchestrated_stages_complete", {"v3_dir": str(v3_dir) if v3_dir else None})


if __name__ == "__main__":
    main()
