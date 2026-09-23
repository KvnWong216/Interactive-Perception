"""Launch/resume one independent, budgeted native/best worker per authorized GPU."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage1_expanded_s3217"
DESIGN = ROOT / "experiments/diagnostic_cases/stage1_expanded_s3217"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(path)


def worker(gpu, execution_gpu=None):
    execution_gpu = gpu if execution_gpu is None else execution_gpu
    directory = RUN / f"gpu{gpu}"
    directory.mkdir(parents=True, exist_ok=True)
    status = {"physical_gpu": gpu, "execution_gpu": execution_gpu, "pid": os.getpid(), "manual_seed": 3217,
              "models": ["native", "best"], "complete": False, "phases": {}}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(execution_gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", HF_HOME=str(ROOT / ".cache/huggingface"),
               HF_MODULES_CACHE=str(ROOT / ".cache/huggingface/modules"))

    def save():
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write(directory / "status.json", status)

    def call(command, log):
        with log.open("a") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def wait_for_gpu():
        """A user authorization is not an exclusive reservation on a shared host."""
        idle_checks = 0
        while idle_checks < 3:
            free = int(subprocess.check_output(["nvidia-smi","-i",str(execution_gpu),
                "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
            pids = subprocess.check_output(["nvidia-smi","-i",str(execution_gpu),
                "--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
            available = free >= 18432 and not pids
            idle_checks = idle_checks+1 if available else 0
            status["phases"][status["current_kind"]] = "waiting_for_idle_gpu"
            status["gpu_availability"] = {"free_mib":free,"minimum_free_mib":18432,
                "compute_processes_present":bool(pids),"consecutive_idle_checks":idle_checks}
            save()
            if idle_checks < 3:
                time.sleep(10 if available else 30)

    with (RUN / f"gpu{gpu}.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            for kind in ("H", "P", "S"):
                phase = directory / kind
                phase.mkdir(exist_ok=True)
                manifest = DESIGN / f"gpu{gpu}_{kind}/manifest.json"
                evaluation = phase / "evaluation"
                previous = json.loads((evaluation / "report.json").read_text()) if (evaluation / "report.json").exists() else None
                status["current_kind"] = kind
                status["phases"][kind] = "auditing"
                save()
                if not (phase / "audit/report.json").exists():
                    call([str(ROOT / ".venv-sim/bin/python"), "-u", str(ROOT / "scripts/audit_expanded_scenes.py"),
                          "--manifest", str(manifest), "--output", str(phase / "audit"),
                          "--physical-gpu", str(execution_gpu)], phase / "audit.console.log")
                audit = json.loads((phase / "audit/report.json").read_text())
                if not audit["passed"]:
                    status["phases"][kind] = "not_evaluated_no_valid_paired_layout"
                    save()
                    continue
                status["phases"][kind] = "evaluating"
                status["eligible_cases"] = audit["cases"]
                save()
                if not previous or not previous["complete"]:
                    command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/evaluate_diagnostics.py"),
                        "--manifest", str(phase / "audit/eligible/manifest.json"),
                        "--scene-audit", str(phase / "audit/report.json"), "--output", str(evaluation),
                        "--physical-gpu", str(execution_gpu), "--modes", "native", "best", "--kinds", kind,
                        "--storage-root", str(directory), "--storage-limit-gb", "19"]
                    if previous:
                        command.append("--resume")
                        if execution_gpu != gpu:
                            command.append("--allow-device-migration")
                    # A different user can claim a card after our audit. Wait
                    # without terminating their work; retain and retry an OOM
                    # rollout only after resources become idle again.
                    while True:
                        wait_for_gpu()
                        status["phases"][kind] = "evaluating"
                        save()
                        try:
                            call(command, phase / "evaluation.console.log")
                            break
                        except subprocess.CalledProcessError:
                            with (phase / "evaluation.console.log").open("rb") as stream:
                                stream.seek(0,2)
                                stream.seek(max(0,stream.tell()-12000))
                                tail = stream.read().decode(errors="replace")
                            if "torch.OutOfMemoryError" not in tail:
                                raise
                            status.setdefault("resource_retries",[]).append({
                                "kind":kind,"reason":"CUDA out of memory; wait for an idle GPU",
                                "timestamp_utc":datetime.now(timezone.utc).isoformat()})
                            if "--resume" not in command:
                                command.append("--resume")
                            save()
                status["phases"][kind] = "complete"
                save()
            status["complete"] = True
            save()
            # The summarizer owns a lock, so concurrent final workers cannot interleave writes.
            call([str(ROOT / ".venv-sim/bin/python"), str(ROOT / "scripts/summarize_generalization_expansion.py")],
                 directory / "summary.console.log")
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError, RuntimeError) as error:
            status["error"] = str(error)
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-gpu", type=int, choices=[0,4,5,6,7])
    parser.add_argument("--execution-gpu", type=int, choices=[0,4,5,6,7])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-gpus", type=int, nargs="+", choices=[0,4,5,6,7])
    args = parser.parse_args()
    if args.worker_gpu is not None:
        worker(args.worker_gpu, args.execution_gpu)
        return
    if args.execution_gpu is not None and (args.only_gpus is None or len(args.only_gpus) != 1):
        parser.error("device migration requires exactly one logical shard in --only-gpus")
    protocol = json.loads((DESIGN / "protocol.json").read_text())
    used = int(subprocess.check_output(["du","-sb",str(ROOT)], text=True).split()[0])
    if used + protocol["budget"]["new_output_limit_decimal_bytes"] > protocol["budget"]["repository_limit_decimal_bytes"]:
        raise RuntimeError("repository plus reserved output would exceed 1 TB")
    RUN.mkdir(parents=True, exist_ok=True)
    launch = RUN / "launch.json"
    if launch.exists() and not args.resume:
        raise ValueError("launch already exists; inspect workers before explicitly resuming")
    processes = []
    for gpu in protocol["gpus"]:
        if args.only_gpus is not None and gpu not in args.only_gpus:
            continue
        directory = RUN / f"gpu{gpu}"
        directory.mkdir(exist_ok=True)
        status_file = directory / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status["complete"]:
                continue
            # Lock ownership, not PID reuse, decides whether a worker is still live.
            with (RUN / f"gpu{gpu}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
        with (directory / "worker.console.log").open("a") as log:
            command = [str(ROOT / ".venv-sim/bin/python"), "-u", str(Path(__file__).resolve()),
                       "--worker-gpu", str(gpu)]
            if args.execution_gpu is not None:
                command += ["--execution-gpu", str(args.execution_gpu)]
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        processes.append({"logical_shard":gpu,"physical_gpu":args.execution_gpu if args.execution_gpu is not None else gpu,"pid":process.pid})
    record = {"started_utc":datetime.now(timezone.utc).isoformat(), "manual_seed":3217,
              "repository_bytes_before_launch":used, "new_output_limit_bytes":100_000_000_000,
              "models":["native","best"], "workers":processes}
    with (RUN / "launch_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record)+"\n")
    write(launch, record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
