#!/usr/bin/env python3
"""Run and attest one exact frozen Heuristic V0 inference on an external GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import load_public_transitions, public_observation_sha256

FROZEN_COMMIT = "e7db12b7f35d9be416fc3ed57d36b12560e40cf0"
FROZEN_TAG = "baseline/heuristic-v0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def tree_identity(path: Path) -> dict[str, int | str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        item_sha = sha256(item)
        size = item.stat().st_size
        total += size
        digest.update(f"{relative}\0{size}\0{item_sha}\n".encode())
    return {
        "schema_version": "piu.checkpoint-tree-sha256.v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


def verify_worktree(path: Path) -> None:
    if git(path, "rev-parse", "HEAD") != FROZEN_COMMIT:
        raise ValueError("Heuristic V0 worktree is not at the frozen commit")
    if git(ROOT, "rev-parse", f"{FROZEN_TAG}^{{commit}}") != FROZEN_COMMIT:
        raise ValueError("Heuristic V0 tag no longer resolves to its frozen commit")
    tracked = git(path, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise ValueError("Heuristic V0 worktree has tracked modifications")


def capture_inputs(path: Path) -> tuple[dict, object, dict[str, dict[str, str]]]:
    capture = json.loads(path.read_text())
    if (
        capture.get("schema_version") != "piu.initial-observation-capture.v1"
        or capture.get("online_oracle_inputs") != []
        or capture.get("evaluator_fields_copied") != []
    ):
        raise ValueError("B2 inference requires a public initial capture")
    transition_spec = capture.get("public_transition", {})
    transition_path = resolve(Path(transition_spec["path"]))
    if sha256(transition_path) != transition_spec.get("sha256"):
        raise ValueError("B2 capture transition differs from its content hash")
    rows = load_public_transitions(transition_path)
    if len(rows) != 1 or rows[0].sample_id != capture.get("sample_id"):
        raise ValueError("B2 capture must contain exactly its declared sample")
    transition = rows[0]
    if public_observation_sha256(transition.observations["pre_interaction"]) != (
        public_observation_sha256(transition.observations["post_interaction"])
    ):
        raise ValueError("B2 initial capture must be an exact observation null")
    images = {}
    for camera in ("agentview", "wrist"):
        specification = transition.observations["post_interaction"]["images"][camera]
        image_path = resolve(Path(specification["path"]))
        pixels = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        pixel_sha = hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
        if (
            sha256(image_path) != specification.get("sha256")
            or pixel_sha != specification.get("pixel_sha256")
        ):
            raise ValueError("B2 public inference image differs from its hashes")
        images[camera] = {
            "path": str(image_path),
            "sha256": sha256(image_path),
            "pixel_sha256": pixel_sha,
        }
    return capture, transition, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-runtime-confirmed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    capture_path = resolve(args.capture_report)
    output_dir = args.output_dir.resolve()
    verify_worktree(worktree)
    capture, transition, images = capture_inputs(capture_path)
    try:
        output_dir.relative_to(worktree)
    except ValueError as error:
        raise ValueError("legacy output directory must be inside its frozen worktree") from error
    report = output_dir / "inference.json"
    attestation = output_dir / "attestation.json"
    command = [
        args.python,
        str(worktree / "scripts/pipeline/infer.py"),
        "--agentview",
        images["agentview"]["path"],
        "--wrist",
        images["wrist"]["path"],
        "--prompt",
        transition.prompt,
        "--device",
        "cuda",
        "--asset-dir",
        str(output_dir / "assets"),
        "--output",
        str(report),
    ]
    plan = {
        "schema_version": "piu.heuristic-v0-inference-plan.v1",
        "method_id": "B2",
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": FROZEN_COMMIT,
        "external_gpu_required": True,
        "local_gpu_run_allowed": False,
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.external_runtime_confirmed or os.environ.get(
        "PIU_EXTERNAL_GPU_RUNTIME"
    ) != "1":
        raise ValueError(
            "B2 inference is prohibited locally; use an external runtime and set "
            "PIU_EXTERNAL_GPU_RUNTIME=1"
        )
    if output_dir.exists():
        raise FileExistsError("Heuristic V0 inference outputs are immutable")
    subprocess.run(command, cwd=worktree, check=True)
    value = json.loads(report.read_text())
    if (
        value.get("schema_version")
        != "interaction-uncertainty.qwen-observation-pipeline.v0"
        or value.get("prompt") != transition.prompt
        or value.get("online_oracle_inputs") != []
    ):
        raise ValueError("frozen Heuristic V0 report violates its public contract")
    model_identities = {}
    for name, relative in value.get("model_paths", {}).items():
        model_path = Path(relative)
        model_path = model_path if model_path.is_absolute() else worktree / model_path
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        model_identities[str(name)] = tree_identity(model_path)
    source = capture["source_state"]
    source_path = resolve(Path(source["path"]))
    if sha256(source_path) != source.get("sha256"):
        raise ValueError("B2 captured source state differs from its hash")
    result = {
        "schema_version": "piu.heuristic-v0-inference-attestation.v1",
        "method_id": "B2",
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": FROZEN_COMMIT,
        "model_identities": model_identities,
        "one_step_baseline": True,
        "decision_observation_sha256": public_observation_sha256(
            transition.observations["post_interaction"]
        ),
        "public_images": images,
        "source_state": {
            "path": portable(source_path),
            "sha256": sha256(source_path),
            "state_key": source["state_key"],
        },
        "inputs": {
            "capture_report": {
                "path": portable(capture_path),
                "sha256": sha256(capture_path),
            },
            "public_transition": {
                "path": portable(
                    resolve(Path(capture["public_transition"]["path"]))
                ),
                "sha256": capture["public_transition"]["sha256"],
            },
        },
        "inference_report": {"path": str(report), "sha256": sha256(report)},
        "online_oracle_inputs": [],
        "local_gpu_actions_performed": False,
        "paper_method_claim_allowed": False,
    }
    attestation.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
