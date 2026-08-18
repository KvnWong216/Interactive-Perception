#!/usr/bin/env python3
"""Render PIU V0 demos and interpretable assets from a completed smoke trace.

The controller has already stopped when this script runs. Public wrist RGB and
the recorded 7D action history come from the smoke artifact. The right-hand
global panel is regenerated in a separate evaluator replay and is never
available to routing, outcome prediction, belief update, or replanning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import DUMMY_ACTION  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOCATION_LABELS = (
    "visible_workspace",
    "middle_drawer",
    "other_unsearched_region",
    "absent",
)
LOCATION_DISPLAY = (
    "Visible\nworkspace",
    "Middle\ndrawer",
    "Other\nunsearched",
    "Absent",
)
VIDEO_WIDTH = 848
VIDEO_HEIGHT = 256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def belief_event(row: dict, event: str) -> dict | None:
    return next(
        (item for item in row["trace"] if item["event"] == event),
        None,
    )


def location_probabilities(event: dict) -> dict[str, float]:
    fact = next(
        item for item in event["belief"]["facts"] if item["name"] == "target_location"
    )
    return {key: float(fact["probabilities"][key]) for key in LOCATION_LABELS}


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def render_uncertainty_map(row: dict, event: dict, path: Path, phase: str) -> None:
    probabilities = location_probabilities(event)
    values = np.asarray(
        [probabilities[label] for label in LOCATION_LABELS], dtype=np.float64
    ).reshape(2, 2)
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="magma")
    for index, (label, value) in enumerate(
        zip(LOCATION_DISPLAY, values.ravel(), strict=True)
    ):
        y, x = divmod(index, 2)
        color = "white" if value > 0.42 else "black"
        axis.text(x, y, f"{label}\n{value:.3f}", ha="center", va="center", color=color, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"{row['case']} · {phase} structured belief map\n"
        "V0 node map — not pixel grounding",
        fontsize=12,
    )
    figure.colorbar(image, ax=axis, label="Prompt-conditioned probability / uncertainty mass")
    save_figure(path)


def render_belief_update(row: dict, path: Path) -> None:
    before_event = belief_event(row, "INITIAL_DECISION")
    after_event = belief_event(row, "POST_OUTCOME_REPLAN")
    before = location_probabilities(before_event)
    phases = [("Before interaction", before)]
    if after_event is not None:
        phases.append(("After public RGB update", location_probabilities(after_event)))
    figure, axes = plt.subplots(
        1, len(phases), figsize=(6.4 * len(phases), 4.2), squeeze=False
    )
    colors = ("#2D8CFF", "#FF9F1C", "#8E6CFF", "#64748B")
    for axis, (title, probabilities) in zip(axes[0], phases, strict=True):
        values = [probabilities[label] for label in LOCATION_LABELS]
        axis.barh(LOCATION_DISPLAY, values, color=colors)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Probability")
        axis.set_title(title)
        for index, value in enumerate(values):
            axis.text(min(value + 0.015, 0.94), index, f"{value:.3f}", va="center")
    figure.suptitle(f"Prompt-conditioned target-location belief · {row['case']}")
    save_figure(path)


def render_action_utility(row: dict, path: Path) -> None:
    decision = belief_event(row, "INITIAL_DECISION")["decision"]
    identifiers = list(decision["utilities"])
    utilities = [decision["utilities"][key]["utility"] for key in identifiers]
    information = [
        decision["utilities"][key]["expected_information_gain"] for key in identifiers
    ]
    progress = [
        decision["utilities"][key]["expected_task_progress"] for key in identifiers
    ]
    x = np.arange(len(identifiers))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = axes[0].bar(x, utilities, color="#00A896")
    selected = decision["selected"]["candidate_id"]
    for index, identifier in enumerate(identifiers):
        if identifier == selected:
            bars[index].set_color("#E63946")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, identifiers, rotation=18, ha="right")
    axes[0].set_ylabel("Explicit utility")
    axes[0].set_title(f"Selected: {selected}")
    width = 0.36
    axes[1].bar(x - width / 2, information, width, label="Expected information gain")
    axes[1].bar(x + width / 2, progress, width, label="Expected task progress")
    axes[1].set_xticks(x, identifiers, rotation=18, ha="right")
    axes[1].set_ylim(bottom=min(0.0, min(information, default=0.0)))
    axes[1].legend()
    axes[1].set_title("Learned terms before cost/risk")
    figure.suptitle(f"Candidate-action comparison · {row['case']}")
    save_figure(path)


def render_effect_forecast(row: dict, path: Path) -> None:
    event = belief_event(row, "INITIAL_DECISION")
    selected = event["decision"]["selected"]["candidate_id"]
    forecast = event["effect_forecasts"].get(selected)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    if not forecast:
        axis.axis("off")
        axis.text(
            0.5,
            0.58,
            "No physical effect forecast executed",
            ha="center",
            va="center",
            fontsize=16,
        )
        axis.text(
            0.5,
            0.42,
            "DIRECT_ACT is a semantic handoff only",
            ha="center",
            va="center",
            fontsize=12,
            color="#B91C1C",
        )
    else:
        labels = list(forecast["outcomes"])
        values = [forecast["outcomes"][label] for label in labels]
        colors = [
            {"FAILED": "#D62828", "REVEALED": "#2A9D8F", "EMPTY": "#F4A261"}.get(
                label, "#64748B"
            )
            for label in labels
        ]
        axis.bar(labels, values, color=colors)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Predicted probability")
        axis.set_title(
            f"Pre-action effect forecast\nexpected future uncertainty="
            f"{forecast['expected_future_uncertainty']:.3f}"
        )
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
    figure.suptitle(f"{row['case']} · {selected}")
    save_figure(path)


def render_visibility(row: dict, path: Path) -> None:
    points = row["evaluator_only"]["visibility_history"]
    labels = [point["name"] for point in points]
    agent = [point["target_pixels"]["agentview"] for point in points]
    wrist = [point["target_pixels"]["wrist"] for point in points]
    x = np.arange(len(points))
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    axis.plot(x, agent, marker="o", label="agentview")
    axis.plot(x, wrist, marker="o", label="wrist")
    axis.axhline(256, linestyle="--", color="#D62828", label="256-pixel resolvability")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Evaluator-only target pixels")
    axis.set_title("Six-point visibility scoring — never a controller input")
    axis.legend()
    save_figure(path)


def render_storyboard(row: dict, path: Path) -> None:
    keyframes = row["assets"]["public_keyframes"]
    columns = len(keyframes)
    figure, axes = plt.subplots(2, columns, figsize=(3.0 * columns, 6.0), squeeze=False)
    for column, point in enumerate(keyframes):
        for row_index, view in enumerate(("agentview", "wrist")):
            axes[row_index, column].imshow(imageio.imread(resolve(point["image_paths"][view])))
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(
                f"{point['name']}\n{view}", fontsize=9
            )
    figure.suptitle(
        f"Policy-visible history · {row['case']}\n"
        f"outcome={row['outcome_prediction_set']} · terminal={row['terminal']}"
    )
    save_figure(path)


def info_panel(row: dict, *, phase: str, step: int, total: int) -> np.ndarray:
    panel = Image.new("RGB", (336, VIDEO_HEIGHT), (12, 18, 28))
    draw = ImageDraw.Draw(panel)
    initial = belief_event(row, "INITIAL_DECISION")
    probabilities = location_probabilities(initial)
    selected = row["initial_selected"]
    resolved = phase == row["terminal"]
    outcome_display = row["outcome_prediction_set"] if resolved else "pending"
    terminal_display = row["terminal"] if resolved else "pending"
    lines = [
        ("PIU V0 / NON-CLAIM", (255, 211, 105)),
        ("overlay: post-hoc trace replay", (156, 163, 175)),
        (f"case: {row['case']}", (255, 255, 255)),
        (f"prompt: {row['prompt']}", (190, 205, 225)),
        (f"route: {selected}", (112, 224, 197)),
        (f"phase: {phase}  {step}/{total}", (147, 197, 253)),
        (f"P(visible): {probabilities['visible_workspace']:.3f}", (255, 255, 255)),
        (f"P(middle drawer): {probabilities['middle_drawer']:.3f}", (255, 255, 255)),
        (f"outcome set: {outcome_display}", (251, 191, 36)),
        (f"terminal: {terminal_display}", (248, 113, 113)),
        ("Global view = post-terminal evaluator replay", (156, 163, 175)),
    ]
    y = 10
    for text, color in lines:
        wrapped = textwrap.wrap(text, width=48) or [""]
        for line in wrapped:
            draw.text((10, y), line, fill=color)
            y += 15
        y += 2
    return np.asarray(panel)


def capture_replay_frame(env, observation: dict, *, demo: dict, status: str) -> np.ndarray:
    left = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    camera = str(demo["demo_camera"])
    pose = demo["demo_camera_pose"]
    sim = env.env.sim
    camera_id = sim.model.camera_name2id(camera)
    original_pos = np.asarray(sim.model.cam_pos[camera_id], dtype=float).copy()
    original_quat = np.asarray(sim.model.cam_quat[camera_id], dtype=float).copy()
    try:
        sim.model.cam_pos[camera_id] = np.asarray(pose["position"], dtype=float)
        sim.model.cam_quat[camera_id] = np.asarray(
            pose["quaternion_wxyz"], dtype=float
        )
        sim.forward()
        global_observation = env.regenerate_obs_from_state(env.get_sim_state())
        right = np.ascontiguousarray(
            global_observation[f"{camera}_image"][::-1, ::-1]
        )
    finally:
        sim.model.cam_pos[camera_id] = original_pos
        sim.model.cam_quat[camera_id] = original_quat
        sim.forward()
    panels = []
    for frame, label in (
        (left, "WRIST / POLICY INPUT"),
        (right, "GLOBAL / EVALUATOR-ONLY REPLAY"),
    ):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 38), fill=(0, 0, 0))
        draw.text((7, 4), label, fill=(255, 255, 255))
        draw.text((7, 21), status, fill=(255, 226, 94))
        panels.append(np.asarray(image))
    return np.concatenate(panels, axis=1)


def title_card(row: dict) -> np.ndarray:
    card = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (9, 14, 24))
    draw = ImageDraw.Draw(card)
    draw.text((24, 24), "PROMPT-CONDITIONED INTERACTION UNCERTAINTY", fill=(255, 211, 105))
    draw.text((24, 54), f"CASE: {row['case']}", fill=(255, 255, 255))
    for index, line in enumerate(textwrap.wrap(row["prompt"], width=70)):
        draw.text((24, 84 + 18 * index), line, fill=(190, 205, 225))
    draw.text((24, 148), f"ROUTE: {row['initial_selected']}", fill=(112, 224, 197))
    draw.text((24, 172), f"OUTCOME: {row['outcome_prediction_set']}", fill=(251, 191, 36))
    draw.text((24, 196), f"TERMINAL: {row['terminal']}", fill=(248, 113, 113))
    draw.text((24, 228), "DIRECT_ACT is a semantic handoff; no grasp success is claimed.", fill=(156, 163, 175))
    return np.asarray(card)


def compose_video_frame(row: dict, replay: np.ndarray, *, phase: str, step: int, total: int) -> np.ndarray:
    return np.concatenate(
        (replay, info_panel(row, phase=phase, step=step, total=total)), axis=1
    )


def render_demo(row: dict, output: Path, combined_writer, *, wait_steps: int, stride: int, demo: dict) -> None:
    from libero.libero.envs import SegmentationRenderEnv

    actions_asset = row["assets"]["public_action_history"]
    actions = json.loads(resolve(actions_asset["path"]).read_text())
    if len(actions) != int(actions_asset["steps"]):
        raise ValueError(f"action asset length mismatch for {row['case']}")
    env = SegmentationRenderEnv(
        bddl_file_name=str(resolve(row["scenario"])),
        camera_heights=256,
        camera_widths=256,
    )
    writer = imageio.get_writer(output, fps=20, codec="libx264", quality=8)
    try:
        env.seed(int(row["seed"]))
        observation = env.reset()
        for _ in range(wait_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        card = title_card(row)
        for _ in range(18):
            writer.append_data(card)
            combined_writer.append_data(card)
        total = len(actions)
        initial = compose_video_frame(
            row,
            capture_replay_frame(env, observation, demo=demo, status="BEFORE"),
            phase="BEFORE",
            step=0,
            total=total,
        )
        for _ in range(14):
            writer.append_data(initial)
            combined_writer.append_data(initial)
        open_steps = int(row["executor"]["open_steps"]) if row["executor"] else 0
        for action_index, action in enumerate(actions, start=1):
            observation, _, done, _ = env.step(action)
            if action_index % stride != 0 and action_index != total:
                continue
            phase = "OPEN_TO_INSPECT" if action_index <= open_steps else "RETURN_TO_OBSERVE"
            replay = capture_replay_frame(
                env,
                observation,
                demo=demo,
                status=f"{phase} {action_index}/{total}",
            )
            frame = compose_video_frame(
                row, replay, phase=phase, step=action_index, total=total
            )
            writer.append_data(frame)
            combined_writer.append_data(frame)
            if done:
                break
        final = compose_video_frame(
            row,
            capture_replay_frame(env, observation, demo=demo, status="CONTROLLER TERMINAL"),
            phase=row["terminal"],
            step=total,
            total=total,
        )
        for _ in range(24):
            writer.append_data(final)
            combined_writer.append_data(final)
    finally:
        writer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=3)
    args = parser.parse_args()
    args.report = resolve(args.report)
    args.output_dir = resolve(args.output_dir)
    manifest_path = args.output_dir / "assets_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable asset manifest exists: {manifest_path}")
    report = json.loads(args.report.read_text())
    if not report.get("asset_root"):
        raise ValueError("smoke report does not contain recorded public assets")
    if args.video_stride < 1:
        raise ValueError("video stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = yaml.safe_load(
        (ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text()
    )
    demo = {
        "demo_camera": benchmark["backend"]["demo_camera"],
        "demo_camera_pose": benchmark["backend"]["demo_camera_pose"],
    }
    combined_path = args.output_dir / "piu_v0_full_pipeline_seed1399.mp4"
    combined_writer = imageio.get_writer(
        combined_path, fps=20, codec="libx264", quality=8
    )
    produced: list[dict] = []
    try:
        for row in report["rows"]:
            case_dir = args.output_dir / row["case"]
            case_dir.mkdir(parents=True, exist_ok=True)
            plots = {
                "belief_update": case_dir / "belief_update.png",
                "uncertainty_before": case_dir / "uncertainty_map_before.png",
                "action_utility": case_dir / "action_utility.png",
                "effect_forecast": case_dir / "effect_forecast.png",
                "evaluator_visibility": case_dir / "evaluator_visibility.png",
                "public_storyboard": case_dir / "public_history_storyboard.png",
            }
            render_belief_update(row, plots["belief_update"])
            render_uncertainty_map(
                row,
                belief_event(row, "INITIAL_DECISION"),
                plots["uncertainty_before"],
                "before",
            )
            after = belief_event(row, "POST_OUTCOME_REPLAN")
            if after is not None:
                plots["uncertainty_after"] = case_dir / "uncertainty_map_after.png"
                render_uncertainty_map(
                    row, after, plots["uncertainty_after"], "after update"
                )
            render_action_utility(row, plots["action_utility"])
            render_effect_forecast(row, plots["effect_forecast"])
            render_visibility(row, plots["evaluator_visibility"])
            render_storyboard(row, plots["public_storyboard"])
            demo_path = case_dir / f"{row['case']}_wrist_global_replay.mp4"
            render_demo(
                row,
                demo_path,
                combined_writer,
                wait_steps=args.wait_steps,
                stride=args.video_stride,
                demo=demo,
            )
            for kind, path in {**plots, "demo": demo_path}.items():
                produced.append(
                    {
                        "case": row["case"],
                        "kind": kind,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": digest(path),
                    }
                )
    finally:
        combined_writer.close()
    produced.append(
        {
            "case": "all",
            "kind": "combined_demo",
            "path": str(combined_path.relative_to(ROOT)),
            "sha256": digest(combined_path),
        }
    )
    manifest = {
        "schema_version": "interaction-uncertainty.piu-visual-assets.v0",
        "source_report": str(args.report.relative_to(ROOT)),
        "source_report_sha256": digest(args.report),
        "claim_status": report["claim_status"],
        "demo_contract": {
            "left": "wrist RGB from deterministic evaluator replay of the public action history",
            "right": "evaluator-only global RGB generated after controller termination",
            "controller_global_view_reads": 0,
            "policy_or_outcome_model_uses_rendered_assets": False,
        },
        "map_contract": "structured task-fact/node maps; V0 does not claim pixel grounding",
        "assets": produced,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"assets": len(produced), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
