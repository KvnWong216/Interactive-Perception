#!/usr/bin/env python3
"""Curate a run's artifacts into the tracked ``results/`` tree.

``outputs/`` is gitignored because a sweep writes gigabytes of frames and raw
traces. What belongs in the repository is the small, readable subset a reader
needs to check a claim: the figures, one demo per condition, the aggregate
summary, and the certificates.

The manifest records the provenance of each copied file along with the run's
headline numbers, so a figure in the repository can always be traced back to the
sweep that produced it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default=str(root / "outputs"))
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument(
        "--max-video-mb",
        type=float,
        default=8.0,
        help="skip demo videos still larger than this after transcoding",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="H.264 quality for the tracked demos; lower is better and larger",
    )
    parser.add_argument(
        "--no-transcode",
        action="store_true",
        help="copy demos verbatim instead of re-encoding them",
    )
    parser.add_argument("--clean", action="store_true", help="empty results/ first")
    return parser.parse_args()


def _load(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _ffmpeg() -> str | None:
    """The encoder imageio already depends on, so this adds no new requirement."""

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _transcode(source: Path, destination: Path, *, crf: int, exe: str) -> bool:
    """Re-encode for the repository at native resolution.

    Resolution is preserved deliberately: the uncertainty curve is drawn into
    the frame as an inset with small tick labels, and downscaling to save bytes
    is what makes the one thing the demo exists to show unreadable. Quality is
    given up instead, via the rate factor.
    """

    result = subprocess.run(
        [
            exe, "-y", "-loglevel", "error",
            "-i", str(source),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.exists():
        if destination.exists():
            destination.unlink()
        return False
    return True


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs).expanduser().resolve()
    results = Path(args.results).expanduser().resolve()
    if args.clean and results.exists():
        shutil.rmtree(results)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    (results / "demos").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"figures": [], "demos": [], "skipped": []}

    for figure in sorted((outputs / "figures").glob("*.png")):
        shutil.copy2(figure, results / "figures" / figure.name)
        manifest["figures"].append(figure.name)

    exe = None if args.no_transcode else _ffmpeg()
    for video in sorted((outputs / "figures").glob("*.mp4")):
        source_mb = video.stat().st_size / 2**20
        destination = results / "demos" / video.name
        transcoded = bool(exe) and _transcode(video, destination, crf=args.crf, exe=exe)
        if not transcoded:
            shutil.copy2(video, destination)
        size_mb = destination.stat().st_size / 2**20
        if size_mb > args.max_video_mb:
            destination.unlink()
            manifest["skipped"].append(
                {
                    "file": video.name,
                    "size_mb": round(size_mb, 1),
                    "reason": f"still over {args.max_video_mb} MB after encoding",
                }
            )
            continue
        manifest["demos"].append(
            {
                "file": video.name,
                "size_mb": round(size_mb, 2),
                "source_mb": round(source_mb, 2),
                "transcoded": transcoded,
            }
        )

    summary = _load(outputs / "challenge_rollout" / "rollout_summary.json")
    if summary is not None:
        with (results / "rollout_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        manifest["conditions"] = summary.get("conditions", {})
        for key in ("capability_minus_implicit", "explicit_minus_implicit"):
            if key in summary:
                manifest[key] = summary[key]

    certificates = _load(outputs / "nbv_certificates" / "nbv_certificates.json")
    if certificates is not None:
        trimmed = {
            key: value for key, value in certificates.items() if key != "rows"
        }
        trimmed["rows"] = [
            {
                "task_id": row.get("task_id"),
                "verdict": row.get("verdict"),
                "max_pre_px": (row.get("pre_manipulation_sweep") or {}).get(
                    "max_visible_pixels"
                ),
                "max_post_px": (row.get("post_manipulation_sweep") or {}).get(
                    "max_visible_pixels"
                ),
                "poses_with_visible_target": (
                    row.get("pre_manipulation_sweep") or {}
                ).get("poses_with_visible_target"),
            }
            for row in certificates.get("rows", [])
        ]
        with (results / "nbv_certificates.json").open("w", encoding="utf-8") as file:
            json.dump(trimmed, file, indent=2, ensure_ascii=False)

    gate = next((outputs / "repro_gate").glob("repro_*.json"), None)
    if gate is not None:
        shutil.copy2(gate, results / gate.name)
        manifest["reproduction_gate"] = _load(gate).get("success_rate")

    with (results / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    total = sum(
        path.stat().st_size for path in results.rglob("*") if path.is_file()
    ) / 2**20
    print(f"figures: {len(manifest['figures'])}")
    print(f"demos:   {len(manifest['demos'])}")
    if manifest["skipped"]:
        print(f"skipped: {len(manifest['skipped'])} (over {args.max_video_mb} MB)")
    print(f"results/ total: {total:.1f} MB")


if __name__ == "__main__":
    main()
