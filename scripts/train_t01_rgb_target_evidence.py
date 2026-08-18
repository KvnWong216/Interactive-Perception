#!/usr/bin/env python3
"""Train a public-RGB, per-frame butter evidence head for T01.

Target segmentation is used only to construct offline frame labels.  At
inference time the model receives the six stock agentview/wrist RGB pairs and
implements the episode rule as a temporal OR over per-frame visual evidence.
This script is a development model-selection tool; its default evaluation
includes contaminated seeds and is therefore never claim-bearing.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_outcome import (  # noqa: E402
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
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


def truth_by_name(row: dict) -> dict[str, dict[str, int]]:
    values = {
        point["name"]: point["target_pixels"]
        for point in row["evaluator_only"]["visibility_history"]
    }
    if tuple(values) != HISTORY_NAMES:
        raise ValueError(f"unexpected evaluator history for seed {row['seed']}")
    return values


@dataclass(frozen=True)
class ImageExample:
    path: Path
    camera: str
    visible: bool


def examples(
    rows: Iterable[dict], cameras: tuple[str, ...] = CAMERAS
) -> list[ImageExample]:
    result: list[ImageExample] = []
    for row in rows:
        public = history_by_name(row)
        truth = truth_by_name(row)
        for name in HISTORY_NAMES:
            for camera in cameras:
                result.append(
                    ImageExample(
                        path=ROOT / public[name]["image_paths"][camera],
                        camera=camera,
                        visible=(
                            truth[name][camera]
                            >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
                        ),
                    )
                )
    return result


def build_model(torch):
    nn = torch.nn

    class TargetEvidenceCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.SiLU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.SiLU(),
                nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.SiLU(),
                nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 1),
            )

        def forward(self, values):
            return self.classifier(self.features(values)).squeeze(1)

    return TargetEvidenceCNN()


def load_image(path: Path, camera: str, size: int, *, augment: bool, rng, Image):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if camera == "agentview":
        # Fixed stock T01 camera crop.  It includes the drawer front/interior and
        # intentionally excludes most task-irrelevant table texture.
        image = image.crop((int(0.36 * width), int(0.22 * height), width, int(0.88 * height)))
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


def predict_examples(model, items, *, device, size, batch_size, torch, Image):
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


def episode_truth(row: dict) -> str:
    truth = truth_by_name(row)
    return (
        "REVEALED"
        if any(
            truth[name][camera] >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
            for name in HISTORY_NAMES
            for camera in CAMERAS
        )
        else "NOT_REVEALED"
    )


def binary_evidence(logit: float) -> dict[str, float]:
    """Convert a finite logit to non-negative two-class conformal evidence."""

    if logit >= 0:
        probability = 1.0 / (1.0 + float(np.exp(-logit)))
    else:
        exponential = float(np.exp(logit))
        probability = exponential / (1.0 + exponential)
    return {"REVEALED": probability, "NOT_REVEALED": 1.0 - probability}


def evaluate(rows, scores, calibrator, cameras):
    records = []
    for row in rows:
        score = episode_score(row, scores, cameras)
        evidence = binary_evidence(score)
        prediction = calibrator.predict(evidence)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": episode_truth(row),
                "maximum_frame_logit": score,
                "prediction_set": list(prediction),
            }
        )
    metrics = {}
    for label in ("REVEALED", "NOT_REVEALED"):
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
        default=None,
        help="May be repeated; defaults to v3 and v4 extension.",
    )
    parser.add_argument("--train-seeds", default="600-619")
    parser.add_argument("--calibration-seeds", default="620-652")
    parser.add_argument("--development-seeds", default="653-699")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--camera-mode",
        choices=("agentview", "wrist", "both"),
        default="agentview",
        help="Online target-evidence stream. T01 defaults to the stable stock agentview.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_rgb_target_evidence_v1_candidate.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/models/t01_rgb_target_evidence_v1_candidate.pt",
    )
    args = parser.parse_args()
    datasets = args.dataset or [
        ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
        ROOT / "data/calibration/t01_open_and_observe_effect_v4_extension.jsonl",
    ]
    datasets = [path if path.is_absolute() else ROOT / path for path in datasets]
    for name in ("output", "checkpoint"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("candidate artifact/checkpoint already exists")

    def parse_range(value: str) -> tuple[int, int]:
        lo, hi = value.split("-", maxsplit=1)
        return int(lo), int(hi)

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; refusing an accidental CPU training run")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rows = load_rows(datasets)
    ranges = {
        "train": parse_range(args.train_seeds),
        "calibration": parse_range(args.calibration_seeds),
        "development": parse_range(args.development_seeds),
    }
    split = {
        name: [row for row in rows if lo <= int(row["seed"]) <= hi]
        for name, (lo, hi) in ranges.items()
    }
    selected_cameras = (
        CAMERAS if args.camera_mode == "both" else (args.camera_mode,)
    )
    train_examples = examples(split["train"], selected_cameras)
    positive = sum(item.visible for item in train_examples)
    negative = len(train_examples) - positive
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
        [row for values in split.values() for row in values], selected_cameras
    )
    scores = predict_examples(
        model,
        all_examples,
        device=device,
        size=args.image_size,
        batch_size=args.batch_size,
        torch=torch,
        Image=Image,
    )
    calibration_pairs = []
    for row in split["calibration"]:
        score = episode_score(row, scores, selected_cameras)
        calibration_pairs.append(
            (binary_evidence(score), episode_truth(row))
        )
    calibrator = MondrianSemanticConformalCalibrator.fit(
        calibration_pairs,
        alpha=args.alpha,
        policy_id="t01_public_rgb_target_evidence_cnn_v1",
        split_id=f"t01_target_evidence_seed{args.calibration_seeds.replace('-', '_')}",
    )
    train_metrics, _ = evaluate(
        split["train"], scores, calibrator, selected_cameras
    )
    calibration_metrics, _ = evaluate(
        split["calibration"], scores, calibrator, selected_cameras
    )
    development_metrics, development_records = evaluate(
        split["development"], scores, calibrator, selected_cameras
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "t01_target_evidence_cnn_v1",
            "image_size": args.image_size,
            "agentview_crop_fraction": [0.36, 0.22, 1.0, 0.88],
        },
        args.checkpoint,
    )
    artifact = {
        "schema_version": "interactive-perception.rgb-target-evidence.v1-candidate",
        "claim": "prompt-resolvable butter evidence at any public RGB history point in T01",
        "claim_eligible": False,
        "reason": "architecture selection uses contaminated development seeds",
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
            "train_loss": losses,
        },
        "minimum_resolvable_target_pixels": PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
        "offline_label_inputs": ["evaluator-only target segmentation pixel counts"],
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
