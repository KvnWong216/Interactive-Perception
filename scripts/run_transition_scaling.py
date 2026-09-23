"""Bounded two-GPU data scaling experiment with locked held-out families."""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

import experiment_runner as runner

ROOT = runner.ROOT
SOURCE = "runs/transition_scaling_s10227"
FEATURES = SOURCE + "/features_stage1"
OUT = ROOT / SOURCE / "experiment"
GPUS = [0, 5]


def summarize():
    rows = []
    for p in sorted((OUT / "heads").glob("*/report.json")):
        report = json.loads(p.read_text())
        groups = {}
        for row in report["result"]["rows"]:
            key = (row["case"]["split"], row["case"]["family"])
            groups.setdefault(key, []).append(row)
        for (split, family), values in groups.items():

            def mean(key, values=values):
                return sum(v[key] for v in values) / len(values)

            rows.append(
                {
                    "condition": p.parent.name,
                    "manual_seed": report["manual_seed"],
                    "train_cases": report["training_cases"],
                    "steps": report["steps"],
                    "blind": report["blind"],
                    "split": split,
                    "family": family,
                    "source_episodes": len(values),
                    "loss": mean("loss"),
                    "copy_gain": 1 - mean("loss") / mean("copy_loss"),
                    "action_variance_explained": 1
                    - mean("action_mse") / mean("target_action_mse"),
                }
            )
    runner.write("family_results.json", rows)


def main():
    global OUT, SOURCE, FEATURES, GPUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-attempt", type=int, default=0)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--physical-gpus", type=int, nargs=2, default=GPUS)
    parser.add_argument("--collect-workers", type=int, default=12)
    args = parser.parse_args()
    collect_workers = args.collect_workers
    if collect_workers < 2 or collect_workers % 2:
        raise ValueError("collect-workers must be positive and even")
    source_path = (ROOT / args.source).resolve()
    if not source_path.is_relative_to(ROOT):
        raise ValueError("source must stay inside repository")
    SOURCE = str(source_path.relative_to(ROOT))
    FEATURES = SOURCE + "/features_stage1"
    OUT = source_path / "experiment"
    GPUS = args.physical_gpus
    if len(set(GPUS)) != 2:
        raise ValueError("two distinct physical GPUs required")
    if args.resume_attempt:
        OUT = ROOT / SOURCE / f"experiment_resume_{args.resume_attempt}"
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used = {int(x.split(",")[0]): int(x.split(",")[1]) for x in usage.splitlines()}
    if any(used[g] > 200 for g in GPUS):
        raise RuntimeError("selected GPUs are not idle")
    size = int(subprocess.check_output(["du", "-sb", str(ROOT)], text=True).split()[0])
    reserve = 60_000_000_000
    if size + reserve > 900_000_000_000 or shutil.disk_usage(ROOT).free < reserve:
        raise RuntimeError(
            "insufficient reserved experiment storage within project budget"
        )
    OUT.mkdir(exist_ok=False)
    runner.OUT = OUT
    runner.STARTED = time.monotonic()
    state = {
        "supervisor_pid": os.getpid(),
        "physical_gpus": GPUS,
        "collection_workers": collect_workers,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "phase": "collection_and_encoding",
        "repository_initial_bytes": size,
        "additional_storage_reserve_bytes": reserve,
        "joint_training_started": False,
    }
    try:
        jobs = []
        for i, gpu in enumerate(GPUS):
            encode_args = [
                "--shard",
                i,
                "--physical-gpu",
                gpu,
                "--source",
                SOURCE,
                "--features-directory",
                FEATURES,
                "--checkpoint",
                "runs/stage1_s17_gpu7/best.pt",
            ]
            if args.resume_attempt:
                encode_args.append("--resume")
            jobs.append(
                runner.launch(
                    f"encode_{i}", gpu, "encode_predictor_forks.py", encode_args
                )
            )
        for i in range(collect_workers):
            jobs.append(
                runner.launch(
                    f"collect_{i}",
                    GPUS[i % 2],
                    "collect_predictor_forks.py",
                    [
                        "collect",
                        "--output",
                        SOURCE,
                        "--workers",
                        collect_workers,
                        "--shard",
                        i,
                        "--physical-gpu",
                        GPUS[i % 2],
                        "--replay-retries",
                        2,
                    ],
                    simulator=True,
                )
            )
        runner.wait_phase(state, jobs)
        plan = json.loads((ROOT / SOURCE / "plan.json").read_text())
        reports = [
            json.loads(
                (ROOT / SOURCE / f"case_{c['case_id']:03d}/report.json").read_text()
            )
            for c in plan["cases"]
        ]
        if not all(r["complete"] and r["initial_arrays_identical"] for r in reports):
            raise RuntimeError("incomplete collection audit")
        for n, steps in [(24, 4000), (250, 400), (250, 4000)]:
            state["phase"] = f"frozen_heads_n{n}_u{steps}"
            jobs = []
            for seed in [17, 29, 43]:
                for blind in [False, True]:
                    name = f"n{n}_u{steps}_{'blind' if blind else 'live'}_s{seed}"
                    args = [
                        "--source",
                        SOURCE,
                        "--features-directory",
                        FEATURES,
                        "--output",
                        OUT / "heads" / name,
                        "--manual-seed",
                        seed,
                        "--steps",
                        steps,
                        "--training-cases",
                        n,
                        "--include-id-development",
                    ]
                    if blind:
                        args.append("--blind")
                    jobs.append(
                        runner.launch(
                            name,
                            GPUS[int(blind)],
                            "train_transition_predictor.py",
                            args,
                        )
                    )
            runner.wait_phase(state, jobs)
            summarize()
        state["phase"] = "locked_confirmation"
        args = [
            "--source",
            SOURCE,
            "--features-directory",
            FEATURES,
            "--output",
            OUT / "qualification.json",
            "--live",
            *[
                OUT / "heads" / f"n250_u4000_live_s{s}" / "head.pt"
                for s in [17, 29, 43]
            ],
            "--blind",
            *[
                OUT / "heads" / f"n250_u4000_blind_s{s}" / "head.pt"
                for s in [17, 29, 43]
            ],
        ]
        runner.wait_phase(
            state,
            [
                runner.launch(
                    "qualification", GPUS[0], "qualify_transition_predictor.py", args
                )
            ],
        )
        report = json.loads((OUT / "qualification.json").read_text())
        state.update(phase="complete", complete=True, qualified=report["passed"])
        runner.write("status.json", state)
        with (ROOT / "docs/experiment_log.md").open("a") as stream:
            stream.write(
                f"\n扩容实验完成：250训练初态、18组冻结头对照；新确认准入passed={report['passed']}。详见`{SOURCE}/experiment/qualification.json`及`family_results.json`。未启动联合训练。\n"
            )
    except BaseException as error:
        state.update(phase="failed", error=repr(error))
        runner.write("status.json", state)
        raise


if __name__ == "__main__":
    main()
