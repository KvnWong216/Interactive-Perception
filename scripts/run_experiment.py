"""Execute an explicit JSON experiment plan without shell interpolation.

Phases run sequentially; jobs inside a phase run concurrently. Existing output
is never overwritten. Only selected jobs are terminated on failure.
"""

import argparse
import json
import os
import time
from pathlib import Path

import experiment_runner as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.plan).resolve()
    output = Path(args.output).resolve()
    if any(not p.is_relative_to(runner.ROOT) for p in [source, output]):
        raise ValueError("plan and outputs must be inside repository")
    plan = json.loads(source.read_text())
    if not plan.get("phases"):
        raise ValueError("experiment has no phases")
    names = []
    for phase in plan["phases"]:
        for job in phase["jobs"]:
            script = runner.ROOT / "scripts" / job["script"]
            if script.parent != runner.ROOT / "scripts" or not script.is_file():
                raise ValueError("job must name an existing repository script")
            names.append(job["name"])
            if Path(job["name"]).name != job["name"]:
                raise ValueError("job name must be a filename")
    if len(names) != len(set(names)):
        raise ValueError("job names must be globally unique")
    output.mkdir(parents=True, exist_ok=False)
    runner.OUT, runner.STARTED = output, time.monotonic()
    runner.write("plan.json", plan)
    state = {"supervisor_pid": os.getpid(), "complete": False}
    jobs = []
    try:
        for phase in plan["phases"]:
            state["phase"] = phase["name"]
            jobs = []
            for job in phase["jobs"]:
                jobs.append(
                    runner.launch(
                        job["name"],
                        job["gpu"],
                        job["script"],
                        job["args"],
                        simulator=job.get("simulator", False),
                    )
                )
            runner.wait_phase(state, jobs)
        state.update(phase="complete", complete=True)
        runner.write("status.json", state)
    except BaseException as error:
        for _, child, _ in jobs:
            if child.poll() is None:
                child.terminate()
        for _, child, _ in jobs:
            child.wait()
        state.update(phase="failed", error=repr(error))
        runner.write("status.json", state)
        raise


if __name__ == "__main__":
    main()
