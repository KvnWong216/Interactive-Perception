"""Refresh paired results while independent evaluation workers progress."""

import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage1_expanded_s3217"


def main():
    with (RUN / "monitor.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(1440):  # At most 12 hours; no GPU work and no downloads.
            result = subprocess.run([str(ROOT / ".venv-sim/bin/python"), str(ROOT / "scripts/summarize_generalization_expansion.py")], cwd=ROOT, check=False)
            statuses = [json.loads((RUN / f"gpu{gpu}/status.json").read_text()) for gpu in (0,4,5,6,7)]
            retryable = []
            for status in statuses:
                if not status.get("error"):
                    continue
                log = RUN / f"gpu{status['physical_gpu']}" / status["current_kind"] / "evaluation.console.log"
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(0,2);stream.seek(max(0,stream.tell()-12000))
                        tail = stream.read().decode(errors="replace")
                    if "torch.OutOfMemoryError" in tail:
                        retryable.append(status["physical_gpu"])
            if retryable:
                for shard in retryable:
                    status = next(s for s in statuses if s["physical_gpu"] == shard)
                    subprocess.run([str(ROOT / ".venv-sim/bin/python"), str(ROOT / "scripts/run_generalization_expansion.py"),
                                    "--resume", "--only-gpus", str(shard), "--execution-gpu",
                                    str(status.get("execution_gpu",shard))],cwd=ROOT,check=True)
            stopped = all(s["complete"] or s.get("error") for s in statuses)
            payload = {"updated_utc":datetime.now(timezone.utc).isoformat(),"summary_returncode":result.returncode,
                       "all_workers_stopped":stopped,"resource_retry_gpus":retryable,
                       "worker_errors":{s["physical_gpu"]:s["error"] for s in statuses if s.get("error")}}
            (RUN / "monitor.json").write_text(json.dumps(payload,indent=2)+"\n")
            if stopped and not retryable:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
