#!/usr/bin/env python3
"""Render the representative seed-1402 evidence trace used by the paper draft."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

PANELS = (
    (
        "(a) closed: butter hidden",
        "direct_butter_closed/assets/00_before_agentview.png",
    ),
    (
        "(b) OPEN: butter exposed",
        "open_butter/assets/04_subtask_end_agentview.png",
    ),
    (
        "(c) post-OPEN DIRECT: wrong object",
        "direct_butter_after_open/assets/04_subtask_end_agentview.png",
    ),
    (
        "(d) visible DIRECT: pick/place test",
        "direct_cream_exact/assets/04_subtask_end_agentview.png",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1402)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/figures/system_trace_seed1402.png",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source = ROOT / "runs/paper_cycle_executor_v2" / f"seed{args.seed}"
    panel_width = 384
    image_height = 384
    caption_height = 52
    margin = 12
    canvas = Image.new(
        "RGB",
        (len(PANELS) * panel_width, image_height + caption_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, (caption, relative) in enumerate(PANELS):
        with Image.open(source / relative) as raw:
            image = raw.convert("RGB").resize((image_height, image_height))
        left = index * panel_width
        canvas.paste(image, (left, 0))
        draw.text((left + margin, image_height + 14), caption, fill="black", font=font)
        if index:
            draw.line((left, 0, left, image_height + caption_height), fill="black", width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output.resolve().relative_to(ROOT))


if __name__ == "__main__":
    main()
