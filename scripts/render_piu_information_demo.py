#!/usr/bin/env python3
"""Assemble one auditable PIU information-acquisition trace and public demo.

The renderer never reads simulator state.  Evaluator-only values already
sealed in the option report are copied into a separately marked report field
after all controller decisions have been assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#10151d")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    *,
    fill: str = "white",
    spacing: int = 8,
) -> None:
    font = ImageFont.load_default(size=18)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += 22 + spacing


def annotated_panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    panel = fit(Image.open(path).convert("RGB"), size)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size[0], 34), fill=(10, 16, 24, 220))
    draw.text((10, 8), label, fill="#ffffff", font=ImageFont.load_default(size=17))
    return panel


def render_demo(
    *,
    option_video: Path,
    initial_maps: list[tuple[Path, str]],
    post_maps: list[tuple[Path, str]],
    output: Path,
    prompt: str,
    initial_action: str,
    outcome: str,
    next_action: str,
) -> None:
    reader = imageio.get_reader(option_video)
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps", 20.0))
    frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
    reader.close()
    if not frames:
        raise ValueError("option video contains no frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        hold = max(1, round(fps * 2.0))
        sequence = [frames[0]] * hold + frames + [frames[-1]] * hold
        physical_start = hold
        physical_end = hold + len(frames)
        for index, frame in enumerate(sequence):
            ratio = (
                0.0
                if index < physical_start
                else 1.0
                if index >= physical_end
                else (index - physical_start) / max(1, len(frames) - 1)
            )
            if index < physical_start:
                stage = f"DECIDE  |  {initial_action}"
                maps = initial_maps
                color = "#ffcf5a"
            elif index < physical_end:
                stage = "EXECUTE  |  frozen pi0.5 OPEN_AND_OBSERVE"
                maps = initial_maps if ratio < 0.72 else post_maps
                color = "#7ac7ff"
            else:
                stage = f"PUBLIC RGB  |  {outcome}  ->  {next_action}"
                maps = post_maps
                color = "#69e39a"

            canvas = Image.new("RGB", (1280, 720), "#0c1118")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 1280, 92), fill="#151d29")
            text_block(
                draw,
                (24, 14),
                [f"Prompt: {prompt}", stage],
                fill="white",
                spacing=2,
            )
            draw.rectangle((18, 102, 824, 684), outline=color, width=4)
            canvas.paste(fit(frame, (798, 574)), (22, 106))

            first = annotated_panel(maps[0][0], maps[0][1], (420, 272))
            second = annotated_panel(maps[1][0], maps[1][1], (420, 272))
            canvas.paste(first, (842, 108))
            canvas.paste(second, (842, 396))
            draw.text(
                (24, 690),
                "Controller inputs: public agentview + wrist RGB, prompt, public state/history | online oracle reads: 0",
                fill="#b9c7d8",
                font=ImageFont.load_default(size=15),
            )
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def render_contact_sheet(
    *,
    panels: list[tuple[Path, str]],
    output: Path,
    prompt: str,
) -> None:
    canvas = Image.new("RGB", (1440, 1030), "#0c1118")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 16),
        f"PIU development trace | {prompt}",
        fill="white",
        font=ImageFont.load_default(size=22),
    )
    for index, (path, label) in enumerate(panels):
        x = 20 + (index % 3) * 470
        y = 70 + (index // 3) * 470
        canvas.paste(annotated_panel(path, label, (450, 430)), (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--option-report", type=Path, required=True)
    parser.add_argument("--outcome-report", type=Path, required=True)
    parser.add_argument("--post-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "initial_report",
        "option_report",
        "outcome_report",
        "post_report",
        "output_dir",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output_dir.exists():
        raise FileExistsError("demo output is immutable")

    initial = load(args.initial_report)
    option = load(args.option_report)
    outcome = load(args.outcome_report)
    post = load(args.post_report)
    prompt = initial["prompt"]
    if post.get("prompt") != prompt or outcome.get("prompt") != prompt:
        raise ValueError("all stages must preserve the original complete prompt")
    for name, report in (
        ("initial", initial),
        ("option", option),
        ("outcome", outcome),
        ("post", post),
    ):
        if report.get("online_oracle_inputs"):
            raise ValueError(f"{name} report declares online oracle inputs")

    initial_action = initial["selected_action"]["action"]
    if initial_action != "OPEN_CONTAINER":
        raise ValueError("demo contract requires the initial PIU decision to open")
    prediction_set = outcome["prediction"]["prediction_set"]
    if prediction_set != ["REVEALED"]:
        raise ValueError("demo contract requires singleton public-RGB REVEALED")
    if outcome.get("camera_fusion") != "v13-complementary":
        raise ValueError("demo contract requires explicit complementary camera fusion")
    next_action = post["selected_action"]["action"]

    option_video = ROOT / option["controller"]["video"]
    before_agentview = ROOT / option["controller"]["keyframes"][0]["image_paths"]["agentview"]
    wrist_logits = outcome["prediction"]["frame_logits"]["wrist_target"]
    reveal_index = max(range(len(wrist_logits)), key=wrist_logits.__getitem__)
    reveal_wrist = ROOT / option["controller"]["keyframes"][reveal_index]["image_paths"]["wrist"]
    initial_maps = [
        (
            ROOT / initial["visualizations"]["agentview"]["uncertainty_overlay"],
            "Initial prompt-aligned uncertainty",
        ),
        (
            ROOT / initial["visualizations"]["agentview"]["interaction_field"],
            f"Selected: {initial_action}",
        ),
    ]
    post_maps = [
        (
            ROOT / post["visualizations"]["wrist"]["uncertainty_overlay"],
            "Updated uncertainty after REVEALED",
        ),
        (
            ROOT / post["visualizations"]["wrist"]["interaction_field"],
            f"Replan: {next_action}",
        ),
    ]
    required_assets = [option_video, before_agentview, reveal_wrist]
    required_assets.extend(path for path, _ in initial_maps + post_maps)
    for path in required_assets:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True)
    video_path = args.output_dir / "piu_information_acquisition.mp4"
    sheet_path = args.output_dir / "piu_information_acquisition_contact_sheet.png"
    trace_path = args.output_dir / "piu_information_acquisition_trace.json"
    render_demo(
        option_video=option_video,
        initial_maps=initial_maps,
        post_maps=post_maps,
        output=video_path,
        prompt=prompt,
        initial_action=initial_action,
        outcome="REVEALED",
        next_action=next_action,
    )
    render_contact_sheet(
        panels=[
            (before_agentview, "1. Before: target hidden (agentview)"),
            initial_maps[0],
            initial_maps[1],
            (reveal_wrist, "4. Wrist evidence: target revealed"),
            post_maps[0],
            post_maps[1],
        ],
        output=sheet_path,
        prompt=prompt,
    )

    fresh_policy_sampling = bool(
        option.get("controller", {}).get(
            "fresh_policy_sampling",
            option.get("frozen_pi05_action_option", {}).get(
                "fresh_policy_sampling", False
            ),
        )
    )
    executor_label = (
        "fresh frozen pi05_libero policy sampling"
        if fresh_policy_sampling
        else "frozen pi05_libero public action option replay"
    )
    limitations = [
        "development/disposable run, not clean validation or sealed audit",
        "the next selected action is replanned but not physically executed in this trace",
        "final butter placement is not demonstrated",
        "v13 complementary camera composition has unit tests but no clean validation yet",
    ]
    if not fresh_policy_sampling:
        limitations.insert(
            1,
            "OPEN_AND_OBSERVE actions were replayed from a frozen public pi0.5 option rather than freshly sampled",
        )

    trace = {
        "schema_version": "interaction-uncertainty.information-acquisition-demo.v1",
        "claim_status": (
            "disposable development information-acquisition chain; "
            + ("fresh frozen pi0.5 sampling" if fresh_policy_sampling else "frozen public pi0.5 action replay")
            + "; not clean or sealed"
        ),
        "prompt": prompt,
        "endpoint": "prompt-relevant target observability",
        "stages": [
            {
                "stage": "INITIAL_INFERENCE",
                "report": reference(args.initial_report),
                "selected_action": initial_action,
                "target_id": initial["selected_action"]["target_id"],
                "online_oracle_inputs": [],
            },
            {
                "stage": "OPEN_AND_OBSERVE",
                "report": reference(args.option_report),
                "executor": executor_label,
                "fresh_policy_sampling": fresh_policy_sampling,
                "online_oracle_inputs": [],
            },
            {
                "stage": "PUBLIC_RGB_OUTCOME",
                "report": reference(args.outcome_report),
                "camera_fusion": outcome["camera_fusion"],
                "prediction_set": prediction_set,
                "target_source": outcome["prediction"]["target_source"],
                "online_oracle_inputs": [],
            },
            {
                "stage": "BELIEF_UPDATE_AND_REPLAN",
                "report": reference(args.post_report),
                "public_outcome": "EVIDENCE_ACQUIRED",
                "next_action": next_action,
                "online_oracle_inputs": [],
            },
        ],
        "terminal": "INFORMATION_ACQUIRED",
        "target_observability_success": True,
        "final_task_executed": False,
        "final_task_success": None,
        "online_oracle_inputs": [],
        "evaluator_only_after_controller_terminal": option.get(
            "evaluator_only_after_controller_terminal",
            option.get("evaluator_only", {}),
        ),
        "limitations": limitations,
        "assets": {
            "video": reference(video_path),
            "contact_sheet": reference(sheet_path),
        },
    }
    trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trace": str(trace_path.relative_to(ROOT)),
                "video": str(video_path.relative_to(ROOT)),
                "contact_sheet": str(sheet_path.relative_to(ROOT)),
                "terminal": trace["terminal"],
                "next_action": next_action,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
