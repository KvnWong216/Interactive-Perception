#!/usr/bin/env python3
"""Train a public-RGB per-frame target or coverage evidence head.

Target segmentation is used only to construct offline frame labels.  At
inference time the model receives the six stock agentview/wrist RGB pairs and
implements the episode rule as a temporal OR over per-frame visual evidence.
All datasets, split ranges, crops, label resolution, and output paths are
parameters. Simulator labels are used only while constructing offline targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    build_rgb_evidence_cnn,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


CAMERAS = ("agentview", "wrist")
HISTORY_NAMES = (
    "before",
    "history_01",
    "history_02",
    "history_03",
    "history_04",
    "after",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def history_by_name(row: dict) -> dict[str, dict]:
    values = {point["name"]: point for point in row["public_history"]}
    if tuple(values) != HISTORY_NAMES:
        raise ValueError(f"unexpected public history for seed {row['seed']}: {tuple(values)}")
    return values


def truth_by_name(
    row: dict, evidence_kind: str = "target"
) -> dict[str, dict[str, int]]:
    if evidence_kind not in {"target", "coverage"}:
        raise ValueError(f"unknown evidence kind {evidence_kind!r}")
    values = {
        point["name"]: point["target_pixels"]
        for point in row["evaluator_only"]["visibility_history"]
    }
    if tuple(values) != HISTORY_NAMES:
        raise ValueError(f"unexpected evaluator history for seed {row['seed']}")
    if evidence_kind == "target":
        return values
    counterfactual = {
        point["name"]: point["target_pixels"]
        for point in row["evaluator_only"]["counterfactual_visibility_history"]
    }
    if counterfactual and tuple(counterfactual) != HISTORY_NAMES:
        raise ValueError(f"unexpected coverage history for seed {row['seed']}")
    return {
        name: {
            camera: max(
                int(values[name][camera]),
                int(counterfactual.get(name, {}).get(camera, 0)),
            )
            for camera in CAMERAS
        }
        for name in HISTORY_NAMES
    }


@dataclass(frozen=True)
class ImageExample:
    path: Path
    camera: str
    visible: bool


def examples(
    rows: Iterable[dict],
    cameras: tuple[str, ...] = CAMERAS,
    evidence_kind: str = "target",
    minimum_positive_pixels: int = 256,
) -> list[ImageExample]:
    result: list[ImageExample] = []
    for row in rows:
        public = history_by_name(row)
        truth = truth_by_name(row, evidence_kind)
        for name in HISTORY_NAMES:
            for camera in cameras:
                result.append(
                    ImageExample(
                        path=ROOT / public[name]["image_paths"][camera],
                        camera=camera,
                        visible=truth[name][camera] >= minimum_positive_pixels,
                    )
                )
    return result


def build_model(torch):
    return build_rgb_evidence_cnn(torch)


def load_image(
    path: Path,
    camera: str,
    size: int,
    *,
    crop_by_camera: dict[str, tuple[float, float, float, float]],
    augment: bool,
    rng,
    Image,
):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = crop_by_camera[camera]
    image = image.crop(
        (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    )
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    if augment:
        gain = rng.uniform(0.92, 1.08)
        bias = rng.uniform(-0.025, 0.025)
        values = np.clip(values * gain + bias, 0.0, 1.0)
    values = (values - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(values, (2, 0, 1)).copy()


def batches(items, batch_size: int, *, shuffle: bool, rng: random.Random):
    indices = list(range(len(items)))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [items[index] for index in indices[start : start + batch_size]]


def predict_examples(
    model, items, *, device, size, batch_size, crop_by_camera, torch, Image
):
    output: dict[tuple[str, str], float] = {}
    model.eval()
    with torch.inference_mode():
        for batch in batches(items, batch_size, shuffle=False, rng=random.Random(0)):
            array = np.stack(
                [
                    load_image(
                        item.path,
                        item.camera,
                        size,
                        crop_by_camera=crop_by_camera,
                        augment=False,
                        rng=random.Random(0),
                        Image=Image,
                    )
                    for item in batch
                ]
            )
            logits = model(torch.from_numpy(array).to(device)).cpu().numpy()
            for item, logit in zip(batch, logits, strict=True):
                output[(str(item.path), item.camera)] = float(logit)
    return output


def episode_score(
    row: dict,
    scores: dict[tuple[str, str], float],
    cameras: tuple[str, ...],
) -> float:
    public = history_by_name(row)
    return max(
        scores[(str(ROOT / public[name]["image_paths"][camera]), camera)]
        for name in HISTORY_NAMES
        for camera in cameras
    )


def evidence_labels(evidence_kind: str) -> tuple[str, str]:
    if evidence_kind == "target":
        return "REVEALED", "NOT_REVEALED"
    if evidence_kind == "coverage":
        return "COMPLETED", "FAILED"
    raise ValueError(f"unknown evidence kind {evidence_kind!r}")


def episode_truth(
    row: dict,
    evidence_kind: str = "target",
    minimum_positive_pixels: int = 256,
    cameras: tuple[str, ...] = CAMERAS,
) -> str:
    truth = truth_by_name(row, evidence_kind)
    positive_label, negative_label = evidence_labels(evidence_kind)
    return (
        positive_label
        if any(
            truth[name][camera] >= minimum_positive_pixels
            for name in HISTORY_NAMES
            for camera in cameras
        )
        else negative_label
    )


def binary_evidence(
    logit: float,
    positive_label: str = "REVEALED",
    negative_label: str = "NOT_REVEALED",
) -> dict[str, float]:
    """Convert a finite logit to non-negative two-class conformal evidence."""

    if logit >= 0:
        probability = 1.0 / (1.0 + float(np.exp(-logit)))
    else:
        exponential = float(np.exp(logit))
        probability = exponential / (1.0 + exponential)
    return {positive_label: probability, negative_label: 1.0 - probability}


def evaluate(
    rows,
    scores,
    calibrator,
    cameras,
    evidence_kind="target",
    minimum_positive_pixels=256,
):
    positive_label, negative_label = evidence_labels(evidence_kind)
    records = []
    for row in rows:
        score = episode_score(row, scores, cameras)
        evidence = binary_evidence(score, positive_label, negative_label)
        prediction = calibrator.predict(evidence)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": episode_truth(
                    row, evidence_kind, minimum_positive_pixels, cameras
                ),
                "maximum_frame_logit": score,
                "prediction_set": list(prediction),
            }
        )
    metrics = {}
    for label in (positive_label, negative_label):
        chosen = [record for record in records if record["truth"] == label]
        metrics[label] = {
            "trials": len(chosen),
            "coverage": float(np.mean([label in row["prediction_set"] for row in chosen])),
            "singleton_accuracy": float(
                np.mean([row["prediction_set"] == [label] for row in chosen])
            ),
            "mean_prediction_set_size": float(
                np.mean([len(row["prediction_set"]) for row in chosen])
            ),
        }
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        required=True,
        help="May be repeated.",
    )
    parser.add_argument("--train-seeds", required=True)
    parser.add_argument("--calibration-seeds", required=True)
    parser.add_argument("--development-seeds", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--minimum-positive-pixels", type=int, default=256)
    parser.add_argument(
        "--agentview-crop", type=float, nargs=4, default=(0.36, 0.22, 1.0, 0.88)
    )
    parser.add_argument(
        "--wrist-crop", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0)
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evidence-kind",
        choices=("target", "coverage"),
        default="target",
        help="Target presence or same-camera searched-region coverage.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("agentview", "wrist", "both"),
        default="agentview",
        help="Online target-evidence stream.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    datasets = [path if path.is_absolute() else ROOT / path for path in args.dataset]
    for name in ("output", "checkpoint"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("candidate artifact/checkpoint already exists")

    def parse_ranges(value: str) -> tuple[tuple[int, int], ...]:
        result = []
        for block in value.split(","):
            lo, hi = block.split("-", maxsplit=1)
            result.append((int(lo), int(hi)))
        return tuple(result)

    def selected(seed: int, ranges: tuple[tuple[int, int], ...]) -> bool:
        return any(lo <= seed <= hi for lo, hi in ranges)

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; refusing an accidental CPU training run")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rows = load_rows(datasets)
    ranges = {
        "train": parse_ranges(args.train_seeds),
        "calibration": parse_ranges(args.calibration_seeds),
        "development": parse_ranges(args.development_seeds),
    }
    split = {
        name: [
            row
            for row in rows
            if selected(int(row["seed"]), selection)
        ]
        for name, selection in ranges.items()
    }
    selected_cameras = (
        CAMERAS if args.camera_mode == "both" else (args.camera_mode,)
    )
    crop_by_camera = {
        "agentview": tuple(args.agentview_crop),
        "wrist": tuple(args.wrist_crop),
    }
    for camera, crop in crop_by_camera.items():
        x0, y0, x1, y1 = crop
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"invalid normalized crop for {camera}: {crop}")
    if args.minimum_positive_pixels <= 0:
        raise ValueError("--minimum-positive-pixels must be positive")
    split_seeds = {
        name: {
            int(row["seed"])
            for row in values
        }
        for name, values in split.items()
    }
    for left, right in (("train", "calibration"), ("train", "development"), ("calibration", "development")):
        overlap = split_seeds[left] & split_seeds[right]
        if overlap:
            raise ValueError(f"seed leakage between {left} and {right}: {sorted(overlap)}")
    if any(not values for values in split.values()):
        sizes = {name: len(values) for name, values in split.items()}
        raise ValueError(f"every split must contain episodes: {sizes}")
    train_examples = examples(
        split["train"],
        selected_cameras,
        args.evidence_kind,
        args.minimum_positive_pixels,
    )
    positive = sum(item.visible for item in train_examples)
    negative = len(train_examples) - positive
    if not positive or not negative:
        raise ValueError(
            f"training frames require both labels; positive={positive}, negative={negative}"
        )
    positive_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)

    model = build_model(torch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rng = random.Random(args.seed)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        for batch in batches(train_examples, args.batch_size, shuffle=True, rng=rng):
            array = np.stack(
                [
                    load_image(
                        item.path,
                        item.camera,
                        args.image_size,
                        crop_by_camera=crop_by_camera,
                        augment=True,
                        rng=rng,
                        Image=Image,
                    )
                    for item in batch
                ]
            )
            target = np.asarray([item.visible for item in batch], dtype=np.float32)
            values = torch.from_numpy(array).to(device)
            labels = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(running / count)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "train_loss": losses[-1]}), flush=True)

    all_examples = examples(
        [row for values in split.values() for row in values],
        selected_cameras,
        args.evidence_kind,
        args.minimum_positive_pixels,
    )
    scores = predict_examples(
        model,
        all_examples,
        device=device,
        size=args.image_size,
        batch_size=args.batch_size,
        crop_by_camera=crop_by_camera,
        torch=torch,
        Image=Image,
    )
    calibration_pairs = []
    positive_label, negative_label = evidence_labels(args.evidence_kind)
    for row in split["calibration"]:
        score = episode_score(row, scores, selected_cameras)
        calibration_pairs.append(
            (
                binary_evidence(score, positive_label, negative_label),
                episode_truth(
                    row,
                    args.evidence_kind,
                    args.minimum_positive_pixels,
                    selected_cameras,
                ),
            )
        )
    calibrator = MondrianSemanticConformalCalibrator.fit(
        calibration_pairs,
        alpha=args.alpha,
        policy_id=f"public_rgb_{args.camera_mode}_{args.evidence_kind}_evidence_cnn_v1",
        split_id=(
            f"{args.evidence_kind}_evidence_seed"
            f"{args.calibration_seeds.replace('-', '_').replace(',', '__')}"
        ),
    )
    train_metrics, _ = evaluate(
        split["train"], scores, calibrator, selected_cameras,
        args.evidence_kind, args.minimum_positive_pixels
    )
    calibration_metrics, _ = evaluate(
        split["calibration"], scores, calibrator, selected_cameras,
        args.evidence_kind, args.minimum_positive_pixels
    )
    development_metrics, development_records = evaluate(
        split["development"], scores, calibrator, selected_cameras,
        args.evidence_kind, args.minimum_positive_pixels
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": f"public_rgb_{args.evidence_kind}_evidence_cnn_v1",
            "image_size": args.image_size,
            "crop_fraction_by_camera": {
                camera: list(crop) for camera, crop in crop_by_camera.items()
            },
        },
        args.checkpoint,
    )
    artifact = {
        "schema_version": f"interactive-perception.rgb-{args.evidence_kind}-evidence.v1-candidate",
        "claim": (
            "prompt-resolvable target evidence at any public RGB history point"
            if args.evidence_kind == "target"
            else "same-camera searched-region coverage at any public RGB history point"
        ),
        "claim_eligible": False,
        "reason": "training artifact; paper claims require a separately frozen clean evaluation",
        "datasets": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
            }
            for path in datasets
        ],
        "split": {name: value for name, value in args.__dict__.items() if name.endswith("seeds")},
        "model": {
            "architecture": "four-layer small CNN",
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "image_size": args.image_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "random_seed": args.seed,
            "camera_mode": args.camera_mode,
            "crop_fraction_by_camera": {
                camera: list(crop) for camera, crop in crop_by_camera.items()
            },
            "evidence_kind": args.evidence_kind,
            "train_loss": losses,
        },
        "minimum_positive_pixels": args.minimum_positive_pixels,
        "offline_label_inputs": [
            "evaluator-only target segmentation pixel counts"
            if args.evidence_kind == "target"
            else "evaluator-only actual/counterfactual same-camera target pixel counts"
        ],
        "online_inputs": [
            f"six stock {camera} RGB frames" for camera in selected_cameras
        ],
        "online_oracle_inputs": [],
        "temporal_rule": "maximum selected-camera frame logit over the six public history points",
        "conformal": calibrator.to_dict(),
        "metrics": {
            "train_nonclaim": train_metrics,
            "calibration_nonclaim": calibration_metrics,
            "development_contaminated_nonclaim": development_metrics,
        },
        "development_rows": development_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": str(args.output), **artifact["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
