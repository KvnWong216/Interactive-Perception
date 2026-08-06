#!/usr/bin/env python3
"""Render uncertainty-versus-time figures and demo videos from rollout traces.

Two outputs.

``--figures`` draws vacuity, dissonance and target visibility against time for
every condition. A curve from a hidden-target scene is always drawn together
with the reference scene declared ``role: uncertainty_reference`` in the
benchmark spec, because a flat trace is only interpretable against a scene where
the evidence genuinely was available. Refusing to plot the challenge curve alone
is deliberate.

``--video`` composites the policy view with a live uncertainty curve inset in
the top-right corner, so behaviour and belief can be read off the same frame.
Probes are sparser than frames, so the inset holds the last measured value
between probes rather than interpolating -- an interpolated curve would imply
measurements that were never taken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

METRICS = {
    "vacuity": "Vacuity  (lack of evidence)",
    "dissonance": "Dissonance  (conflicting evidence)",
    "normalized_predictive_entropy": "Normalized predictive entropy",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=str(root / "outputs" / "challenge_rollout"))
    parser.add_argument(
        "--spec",
        default=str(root / "benchmarks" / "interactive_manipulation_v0" / "benchmark.yaml"),
    )
    parser.add_argument("--output", default=str(root / "outputs" / "figures"))
    parser.add_argument("--metric", default="vacuity", choices=sorted(METRICS))
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--all-videos",
        action="store_true",
        help="render a video for every episode that stored frames",
    )
    parser.add_argument("--task-id", default=None, help="task to render as video")
    parser.add_argument("--variant", default="implicit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            if payload.get("kind") == "episode":
                header = payload
            elif payload.get("kind") == "step":
                steps.append(payload)
    return header, steps


def discover(traces_root: Path) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    return [load_trace(path) for path in sorted(traces_root.rglob("trace.jsonl"))]


def reference_task_id(spec: dict[str, Any]) -> str | None:
    for task in spec.get("tasks", []):
        if task.get("role") == "uncertainty_reference":
            return str(task["id"])
    return None


def draw_figures(
    loaded: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    spec: dict[str, Any],
    metric: str,
    output: Path,
) -> list[Path]:
    reference = reference_task_id(spec)
    grouped: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for header, steps in loaded:
        if not steps:
            continue
        grouped.setdefault((header["task_id"], header["prompt_variant"]), []).append(steps)

    written: list[Path] = []
    task_ids = sorted({key[0] for key in grouped})
    challenge_ids = [task for task in task_ids if task != reference]

    for task_id in challenge_ids:
        variants = sorted(key[1] for key in grouped if key[0] == task_id)
        if not variants:
            continue
        figure, axes = plt.subplots(
            2, 1, figsize=(9.0, 6.4), sharex=True, height_ratios=[3, 2]
        )
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for index, variant in enumerate(variants):
            for run_index, steps in enumerate(grouped[(task_id, variant)]):
                x = [item["step"] for item in steps]
                y = [item[metric] for item in steps]
                axes[0].plot(
                    x,
                    y,
                    color=colors[index % 10],
                    alpha=0.85 if run_index == 0 else 0.35,
                    label=f"{task_id} / {variant}" if run_index == 0 else None,
                )
                axes[1].plot(
                    x,
                    [item["target_visible_pixels"] or 0 for item in steps],
                    color=colors[index % 10],
                    alpha=0.85 if run_index == 0 else 0.35,
                )

        # The reference condition is mandatory context, not decoration.
        if reference:
            for key in [k for k in grouped if k[0] == reference]:
                for run_index, steps in enumerate(grouped[key]):
                    axes[0].plot(
                        [item["step"] for item in steps],
                        [item[metric] for item in steps],
                        color="black",
                        linestyle="--",
                        alpha=0.9 if run_index == 0 else 0.3,
                        label=f"reference: {reference} / {key[1]}"
                        if run_index == 0
                        else None,
                    )

        axes[0].set_ylabel(METRICS[metric])
        axes[0].set_title(
            f"{task_id}: does the action distribution register missing evidence?"
        )
        axes[0].legend(fontsize=8, loc="best")
        axes[0].grid(alpha=0.25)
        axes[1].set_ylabel("target pixels\n(evaluator-only)")
        axes[1].set_xlabel("environment step")
        axes[1].grid(alpha=0.25)

        figure.tight_layout()
        path = output / f"{task_id}_{metric}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
    return written


def draw_primitive_figure(
    loaded: list[tuple[dict[str, Any], list[dict[str, Any]]]], *, output: Path
) -> Path | None:
    """Mean evidence per coarse primitive, per condition.

    The bar for ``NOT_FOUND`` is expected to be exactly zero everywhere: a
    continuous-action policy has no channel that expresses abstention. Plotting
    the empty bar is the point.
    """

    totals: dict[str, dict[str, list[float]]] = {}
    for header, steps in loaded:
        if not steps:
            continue
        condition = f'{header["task_id"]}/{header["prompt_variant"]}'
        bucket = totals.setdefault(condition, {})
        for step in steps:
            for name, value in step["primitive_evidence"].items():
                bucket.setdefault(name, []).append(float(value))
    if not totals:
        return None

    conditions = sorted(totals)
    primitives = sorted({name for bucket in totals.values() for name in bucket})
    figure, axis = plt.subplots(figsize=(max(7.5, 1.6 + 1.1 * len(conditions)), 4.6))
    width = 0.8 / max(1, len(primitives))
    positions = np.arange(len(conditions))

    for index, primitive in enumerate(primitives):
        means = [
            float(np.mean(totals[condition].get(primitive, [0.0])))
            for condition in conditions
        ]
        axis.bar(positions + index * width, means, width=width, label=primitive)

    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("mean evidence per probe")
    axis.set_title("Decoded action-primitive evidence (NOT_FOUND is structurally zero)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    path = output / "primitive_evidence.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _curve_inset(
    steps: list[dict[str, Any]], *, upto: int, metric: str, size: int
) -> np.ndarray:
    figure = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    axis = figure.add_subplot(111)
    x = [item["step"] for item in steps if item["step"] <= upto]
    y = [item[metric] for item in steps if item["step"] <= upto]
    all_x = [item["step"] for item in steps]

    if x:
        # Hold the last measurement rather than interpolating between probes.
        axis.step(x, y, where="post", color="#d62728", linewidth=2.0)
        axis.scatter(x[-1:], y[-1:], color="#d62728", s=18, zorder=3)
    axis.set_xlim(min(all_x, default=0), max(all_x, default=1))
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(metric, fontsize=8)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.3)
    figure.tight_layout(pad=0.2)

    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return buffer


def render_video(
    *,
    case_dir: Path,
    metric: str,
    output: Path,
    fps: int,
) -> Path:
    import imageio.v2 as imageio

    header, steps = load_trace(case_dir / "trace.jsonl")
    frames_path = case_dir / "frames.npy"
    if not frames_path.exists():
        raise SystemExit(
            f"{frames_path} missing; re-run the rollout with --save-frames"
        )
    frames = np.load(frames_path)
    if not steps:
        raise SystemExit(f"no probe records in {case_dir}")

    scale = 3
    height, width = frames.shape[1] * scale, frames.shape[2] * scale
    margin = 4
    # The inset must fit inside the frame with margins on both sides. Matplotlib
    # also rounds figure dimensions, so the rendered panel is cropped to the
    # destination box rather than assumed to match it.
    inset = min(max(96, min(height, width) // 3), min(height, width) - 2 * margin)
    if inset < 32:
        raise SystemExit(
            f"frames are {frames.shape[2]}x{frames.shape[1]}px, too small for an "
            f"uncertainty inset; re-render the scenario at a larger resolution"
        )
    composed: list[np.ndarray] = []

    for index, frame in enumerate(frames):
        canvas = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1).astype(np.uint8)
        overlay = _curve_inset(steps, upto=index, metric=metric, size=inset)
        panel_h = min(overlay.shape[0], height - 2 * margin)
        panel_w = min(overlay.shape[1], width - 2 * margin)
        canvas[margin : margin + panel_h, width - panel_w - margin : width - margin] = (
            overlay[:panel_h, :panel_w]
        )
        composed.append(canvas)

    output.mkdir(parents=True, exist_ok=True)
    path = output / (
        f'{header["task_id"]}_{header["prompt_variant"]}_seed{header["seed"]:03d}.mp4'
    )
    imageio.mimwrite(path, composed, fps=fps)
    return path


def main() -> None:
    args = parse_args()
    traces_root = Path(args.traces).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with Path(args.spec).expanduser().resolve().open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)

    if not args.figures and not args.video and not args.all_videos:
        args.figures = True

    if args.figures:
        loaded = discover(traces_root)
        if not loaded:
            raise SystemExit(f"no trace.jsonl found under {traces_root}")
        written = draw_figures(loaded, spec=spec, metric=args.metric, output=output)
        primitive = draw_primitive_figure(loaded, output=output)
        for path in written + ([primitive] if primitive else []):
            print(f"wrote {path}")

    if args.all_videos:
        cases = sorted(path.parent for path in traces_root.rglob("frames.npy"))
        if not cases:
            raise SystemExit(
                f"no frames.npy under {traces_root}; re-run the rollout with --save-frames"
            )
        for case_dir in cases:
            try:
                path = render_video(
                    case_dir=case_dir, metric=args.metric, output=output, fps=args.fps
                )
            except SystemExit as error:
                # One unusable episode must not abandon the remaining demos.
                print(f"skipped {case_dir}: {error}")
                continue
            print(f"wrote {path}")

    if args.video:
        if not args.task_id:
            raise SystemExit("--video requires --task-id")
        case_dir = traces_root / args.task_id / args.variant / f"seed_{args.seed:03d}"
        print(f"wrote {render_video(case_dir=case_dir, metric=args.metric, output=output, fps=args.fps)}")


if __name__ == "__main__":
    main()
