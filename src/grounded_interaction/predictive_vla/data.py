"""Real trajectory loading with episode/family split and causal window checks."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime import parse_response
from .types import AppliedAction, CameraFrame, Observation, PolicyContext

DATA_SCHEMA = "predictive-vla-trajectories-v1"


@dataclass(frozen=True)
class TrainingExample:
    context: PolicyContext
    actual_future_actions: np.ndarray
    future_observation: Observation | None
    response: str | None

    def __post_init__(self):
        actions = np.asarray(self.actual_future_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("future actions must be a finite T x 7 executed prefix")
        if len(actions) > self.context.remaining_steps:
            raise ValueError("supervised actions exceed the original budget")
        if self.future_observation is not None:
            if not len(
                actions
            ) or self.future_observation.step != self.context.current.step + len(
                actions
            ):
                raise ValueError(
                    "future observation must follow exactly the recorded action segment"
                )
        elif len(actions):
            raise ValueError("executed actions require an actual boundary observation")
        if self.response is not None:
            response = parse_response(self.response)
            if response.done and len(actions):
                raise ValueError(
                    "a completed example cannot also supervise further actions"
                )
        if not len(actions) and self.response is None:
            raise ValueError("example has no real supervision")
        object.__setattr__(self, "actual_future_actions", actions)


class Trajectory:
    def __init__(self, path, entry, *, total_steps):
        self.entry = entry
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        required = {"agent_rgb", "wrist_rgb", "states", "actions"}
        geometry = {
            f"{view}_{field}"
            for view in ("agent", "wrist")
            for field in ("depth", "K", "T_world")
        }
        if not required.issubset(self.arrays) or set(self.arrays) - required - geometry:
            raise ValueError("trajectory arrays are missing or unknown")
        a = self.arrays
        self.length = len(a["actions"])
        if (
            self.length > total_steps
            or a["actions"].shape != (self.length, 7)
            or not np.isfinite(a["actions"]).all()
        ):
            raise ValueError(
                "actions must be a finite T x 7 sequence within the episode budget"
            )
        if (
            a["states"].shape != (self.length + 1, 8)
            or not np.isfinite(a["states"]).all()
        ):
            raise ValueError("trajectory needs T+1 real 8-D states for T actions")
        for view in ("agent", "wrist"):
            rgb = a[f"{view}_rgb"]
            if (
                rgb.ndim != 4
                or rgb.shape[0] != self.length + 1
                or rgb.shape[-1] != 3
                or rgb.dtype != np.uint8
            ):
                raise ValueError("trajectory needs T+1 aligned RGB frames per camera")
            geo = {f"{view}_{field}" for field in ("depth", "K", "T_world")}
            if geo & a.keys() and not geo.issubset(a):
                raise ValueError(
                    "each camera needs depth, intrinsics and extrinsics together"
                )
            if geo.issubset(a) and (
                a[f"{view}_depth"].shape != rgb.shape[:-1]
                or a[f"{view}_K"].shape not in {(3, 3), (self.length + 1, 3, 3)}
                or a[f"{view}_T_world"].shape != (self.length + 1, 4, 4)
            ):
                raise ValueError(
                    "geometry must be aligned at every real observation time"
                )
        # Validate all calibrations before an optimizer can be constructed.
        for step in range(self.length + 1):
            self.observation(step)

    def observation(self, step):
        a = self.arrays
        cameras = []
        for view in ("agent", "wrist"):
            if f"{view}_depth" in a:
                k = a[f"{view}_K"]
                cameras.append(
                    CameraFrame(
                        a[f"{view}_rgb"][step],
                        a[f"{view}_depth"][step],
                        k if k.ndim == 2 else k[step],
                        a[f"{view}_T_world"][step],
                    )
                )
            else:
                cameras.append(CameraFrame(a[f"{view}_rgb"][step]))
        return Observation(step, tuple(cameras), a["states"][step])

    def context(self, step, config):
        # Includes the current frame once, at the same cadence as deployment.
        times = [*range(0, step, config.execute_steps), step][-config.history_frames :]
        actions = tuple(
            AppliedAction(i, self.arrays["actions"][i]) for i in range(times[0], step)
        )
        return PolicyContext(
            self.entry["task"],
            tuple(self.observation(i) for i in times),
            actions,
            config.total_steps - step,
        )

    def examples(self, config):
        for step in range(0, self.length, config.execute_steps):
            end = min(step + config.prediction_steps, self.length)
            yield TrainingExample(
                self.context(step, config),
                self.arrays["actions"][step:end],
                self.observation(end),
                None,  # Unannotated demonstrations do not prove a stop decision.
            )
        response = self.entry.get("final_response")
        if response is not None:
            yield TrainingExample(
                self.context(self.length, config),
                np.empty((0, 7), dtype=np.float32),
                None,
                response,
            )


class TrajectoryDataset:
    def __init__(self, manifest, config):
        self.path = Path(manifest).expanduser().resolve()
        self.config = config
        value = json.loads(self.path.read_text())
        if (
            set(value) != {"schema", "episodes"}
            or value["schema"] != DATA_SCHEMA
            or not value["episodes"]
        ):
            raise ValueError("unsupported or empty real trajectory manifest")
        self.entries = value["episodes"]
        identifiers, families, files = set(), {}, {}
        for row in self.entries:
            if set(row) - {
                "episode_id",
                "reset_family",
                "split",
                "task",
                "path",
                "final_response",
            } or not {
                "episode_id",
                "reset_family",
                "split",
                "task",
                "path",
            }.issubset(row):
                raise ValueError("invalid trajectory manifest fields")
            if any(
                not isinstance(row[k], str) or not row[k].strip()
                for k in ("episode_id", "reset_family", "task", "path")
            ):
                raise ValueError(
                    "episode identity, task and path must be nonempty strings"
                )
            if row["episode_id"] in identifiers:
                raise ValueError("duplicate episode identity")
            identifiers.add(row["episode_id"])
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError("unsupported trajectory split")
            if families.setdefault(row["reset_family"], row["split"]) != row["split"]:
                raise ValueError("a reset family crosses data splits")
            path = self.resolve(row)
            if path in files:
                raise ValueError("duplicated trajectory path, possibly across splits")
            files[path] = row["split"]
            if not path.is_file():
                raise ValueError("trajectory file is missing")
            if (
                row.get("final_response") is not None
                and not parse_response(row["final_response"]).done
            ):
                raise ValueError("final_response must be an annotated DONE answer")

    def resolve(self, row):
        return (self.path.parent / row["path"]).resolve()

    def read_validation_report(self, path):
        """Consume the explicitly supplied completed audit from this data directory."""
        path = Path(path).resolve()
        report = json.loads(path.read_text())
        splits = {}
        for row in self.entries:
            splits[row["split"]] = splits.get(row["split"], 0) + 1
        summary = report.get("summary", {})
        if (
            path.parent != self.path.parent
            or report.get("passed") is not True
            or report.get("manual_seed") != self.config.manual_seed
            or summary.get("episodes") != len(self.entries)
            or summary.get("splits") != splits
        ):
            raise ValueError(
                "data audit must pass for this directory, manual_seed and split counts"
            )
        return summary

    def validate(self, *, workers=16, progress=None):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("validation workers must be between 1 and 32")
        summary = {
            "episodes": len(self.entries),
            "actions": 0,
            "rgbd_episodes": 0,
            "answer_annotations": 0,
            "splits": {},
        }

        def check(row):
            trajectory = Trajectory(
                self.resolve(row), row, total_steps=self.config.total_steps
            )
            rgbd = int(
                all(f"{view}_depth" in trajectory.arrays for view in ("agent", "wrist"))
            )
            return (
                trajectory.length,
                rgbd,
                row.get("final_response") is not None,
                row["split"],
            )

        # Decode and inspect independent files concurrently, returning only small
        # summaries so decoded image arrays are bounded by the worker count.
        with ThreadPoolExecutor(max_workers=min(workers, len(self.entries))) as pool:
            futures = [pool.submit(check, row) for row in self.entries]
            for completed, future in enumerate(as_completed(futures), 1):
                actions, rgbd, annotated, split = future.result()
                summary["actions"] += actions
                summary["rgbd_episodes"] += rgbd
                summary["answer_annotations"] += int(annotated)
                summary["splits"][split] = summary["splits"].get(split, 0) + 1
                if progress is not None:
                    progress(completed, len(self.entries))
        return summary

    def examples(self, split, *, seed=0):
        import random

        rows = [r for r in self.entries if r["split"] == split]
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = iter(rows)
        pools = []

        def load_next():
            row = next(rows, None)
            if row is None:
                return False
            examples = list(
                Trajectory(
                    self.resolve(row), row, total_steps=self.config.total_steps
                ).examples(self.config)
            )
            rng.shuffle(examples)
            if examples:
                pools.append(examples)
            return True

        # Interleave trajectories so gradient accumulation is not dominated by
        # consecutive windows of one demonstration. Eight decoded episodes
        # bound RAM; all shuffling is local and deterministic for exact resume.
        for _ in range(8):
            if not load_next():
                break
        while pools:
            index = rng.randrange(len(pools))
            yield pools[index].pop()
            if not pools[index]:
                pools.pop(index)
                load_next()
