#!/usr/bin/env python3
"""Build public-RGB object Scene Packets with Grounding DINO, SAM, and DINOv2.

The input index contains only policy-visible image paths, prompts, and optional
open-vocabulary query lists.  Simulator labels are deliberately not accepted.
Grounding DINO proposes boxes, SAM converts them to masks, and DINOv2 produces
mask-pooled region features.  The output JSONL and NPZ are therefore safe to
consume on the controller side.

Run this script in the OpenPI ``uv`` environment, which already contains a
CUDA-enabled PyTorch and Transformers installation.  The three public models
are loaded from local, versioned directories; no network access occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from interaction_uncertainty.grounding_dino_compat import (
    grounding_dino_post_process_identity,
    post_process_grounded_object_detection_compat,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = (
    "object",
    "food package",
    "drawer",
    "cabinet",
    "refrigerator",
    "fridge",
    "refrigerator door",
    "mini refrigerator",
    "basket",
    "container",
    "bottle",
    "box",
    "bowl",
    "package",
)
FORBIDDEN_INPUT_KEYS = {
    "segmentation",
    "simulator_segmentation",
    "drawer_joint",
    "hidden_pose",
    "target_pose",
    "task_predicate",
    "semantic_id",
    "global_camera",
    "bev",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Prefer repository-relative provenance without rejecting external smoke inputs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "object"


def validate_public_row(row: dict[str, Any]) -> None:
    flattened = {str(key).lower() for key in row}
    flattened.update(str(key).lower() for key in row.get("policy_inputs", {}))
    overlap = flattened & FORBIDDEN_INPUT_KEYS
    if overlap:
        raise ValueError(f"input index contains forbidden controller keys: {sorted(overlap)}")
    policy = row.get("policy_inputs", {})
    images = policy.get("image_paths")
    if not isinstance(images, dict) or not {"agentview", "wrist"} <= set(images):
        raise ValueError("each row requires policy_inputs.image_paths agentview+wrist")
    if not str(row.get("prompt", "")).strip():
        raise ValueError("each row requires a non-empty prompt")


def normalize_queries(row: dict[str, Any]) -> tuple[str, ...]:
    explicit = row.get("visual_queries", ())
    values = [str(value).strip() for value in explicit if str(value).strip()]
    target = row.get("target")
    destination = row.get("destination")
    if not target:
        match = re.search(
            r"(?:place|put|pick up)\s+(?:the\s+)?(.+?)\s+(?:in|into|on)\s+(?:the\s+)?(.+?)[.!]?$",
            str(row.get("prompt", "")),
            flags=re.IGNORECASE,
        )
        if match:
            target = match.group(1).strip()
            destination = destination or match.group(2).strip()
    if target:
        target_text = str(target).strip()
        values.insert(0, target_text)
        values.insert(1, f"{target_text} package")
        values.insert(2, f"{target_text} box")
    if destination:
        values.append(str(destination).strip())
    values.extend(DEFAULT_QUERIES)
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key and key not in seen:
            unique.append(value)
            seen.add(key)
    return tuple(unique)


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0.0 else 0.0


def label_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    *,
    threshold: float,
) -> list[int]:
    keep: list[int] = []
    for index in np.argsort(-scores).tolist():
        if all(
            labels[index] != labels[previous]
            or box_iou(boxes[index], boxes[previous]) < threshold
            for previous in keep
        ):
            keep.append(index)
    return keep


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    runs: list[int] = []
    last = 0
    length = 0
    for value in flat:
        current = int(value)
        if current == last:
            length += 1
        else:
            runs.append(length)
            length = 1
            last = current
    runs.append(length)
    return {"size": list(mask.shape), "counts": runs, "starts_with": 0}


class PublicObjectFrontend:
    def __init__(
        self,
        *,
        grounding_model: Path,
        sam_model: Path,
        dino_model: Path,
        device: str,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        precision: str,
        dense_grid: int,
        dense_min_iou: float,
        max_dense_proposals: int,
    ) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self.torch = torch
        self.device = torch.device(device)
        # Grounding DINO's text-enhancer path in Transformers 4.53 creates
        # float32 intermediates, so blanket fp16 loading is not numerically
        # compatible.  Float32 fits the local 16 GB card for these three small
        # models and is the frozen default; mixed precision can be revisited as
        # a separate, explicitly validated backend.
        self.dtype = torch.float32 if precision == "float32" else torch.bfloat16
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.dense_grid = int(dense_grid)
        self.dense_min_iou = float(dense_min_iou)
        self.max_dense_proposals = int(max_dense_proposals)

        self.grounding_processor = AutoProcessor.from_pretrained(
            grounding_model, local_files_only=True
        )
        self.grounding_dino_post_process = grounding_dino_post_process_identity(
            self.grounding_processor.post_process_grounded_object_detection
        )
        self.grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_model, local_files_only=True)
        self.sam = SamModel.from_pretrained(
            sam_model,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_model, local_files_only=True
        )
        self.dino = AutoModel.from_pretrained(
            dino_model,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

    def _move(self, values: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in values.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif getattr(value, "is_floating_point", lambda: False)():
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def detect(
        self, image: Image.Image, queries: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
        # Query separately.  Grounding DINO may concatenate adjacent phrases
        # (for example "container box package") when a long phrase ensemble is
        # submitted at once, which destroys the object-label contract.
        all_boxes: list[np.ndarray] = []
        all_scores: list[float] = []
        all_labels: list[str] = []
        target_sizes = self.torch.tensor(
            [[image.height, image.width]], device=self.device
        )
        for query in queries:
            inputs = self._move(
                self.grounding_processor(
                    images=image, text=f"{query}.", return_tensors="pt"
                )
            )
            with self.torch.inference_mode():
                outputs = self.grounding(**inputs)
            results = post_process_grounded_object_detection_compat(
                self.grounding_processor.post_process_grounded_object_detection,
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
            query_boxes = results["boxes"].detach().float().cpu().numpy()
            query_scores = results["scores"].detach().float().cpu().numpy()
            all_boxes.extend(query_boxes)
            all_scores.extend(float(value) for value in query_scores)
            all_labels.extend([query] * len(query_boxes))
        if not all_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []
        boxes = np.asarray(all_boxes, dtype=np.float32)
        scores = np.asarray(all_scores, dtype=np.float32)
        keep = label_aware_nms(boxes, scores, all_labels, threshold=self.nms_iou)

        # Merge high-IoU cross-query aliases into one object node and retain a
        # normalized open-vocabulary label distribution.
        groups: list[list[int]] = []
        for index in sorted(keep, key=lambda item: -float(scores[item])):
            destination = next(
                (
                    group
                    for group in groups
                    if box_iou(boxes[index], boxes[group[0]]) >= self.nms_iou
                ),
                None,
            )
            if destination is None:
                groups.append([index])
            else:
                destination.append(index)
        merged_boxes: list[np.ndarray] = []
        merged_scores: list[float] = []
        merged_labels: list[dict[str, float]] = []
        for group in groups:
            primary = max(group, key=lambda item: float(scores[item]))
            label_scores: dict[str, float] = {}
            for index in group:
                label_scores[all_labels[index]] = max(
                    label_scores.get(all_labels[index], 0.0), float(scores[index])
                )
            total = sum(label_scores.values())
            merged_boxes.append(boxes[primary])
            merged_scores.append(float(scores[primary]))
            merged_labels.append(
                {label: value / total for label, value in label_scores.items()}
            )
        return (
            np.asarray(merged_boxes, dtype=np.float32),
            np.asarray(merged_scores, dtype=np.float32),
            merged_labels,
        )

    def segment(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return (
                np.zeros((0, image.height, image.width), dtype=bool),
                np.zeros(0, dtype=np.float32),
            )
        inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"].clone()
        reshaped_sizes = inputs["reshaped_input_sizes"].clone()
        inputs = self._move(inputs)
        with self.torch.inference_mode():
            outputs = self.sam(**inputs, multimask_output=True)
        processed = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            reshaped_sizes,
        )[0]
        iou = outputs.iou_scores.detach().float().cpu()[0]
        best = iou.argmax(dim=-1)
        masks = self.torch.stack(
            [processed[index, best[index]] for index in range(len(boxes))]
        ).numpy() > 0.0
        scores = iou[self.torch.arange(len(boxes)), best].numpy()
        return masks, np.clip(scores, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _mask_box(mask: np.ndarray) -> np.ndarray:
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.asarray(
            [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
            dtype=np.float32,
        )

    @staticmethod
    def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
        intersection = np.logical_and(left, right).sum()
        union = np.logical_or(left, right).sum()
        return float(intersection / union) if union else 0.0

    def dense_proposals(
        self,
        image: Image.Image,
        *,
        grounded_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate class-agnostic SAM proposals so small targets are retained."""

        if self.dense_grid <= 0 or self.max_dense_proposals <= 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, image.height, image.width), dtype=bool),
                np.zeros(0, dtype=np.float32),
            )
        xs = np.linspace(
            image.width / (2 * self.dense_grid),
            image.width - image.width / (2 * self.dense_grid),
            self.dense_grid,
        )
        ys = np.linspace(
            image.height / (2 * self.dense_grid),
            image.height - image.height / (2 * self.dense_grid),
            self.dense_grid,
        )
        points = [[[[float(x), float(y)]] for y in ys for x in xs]]
        inputs = self.sam_processor(
            images=image,
            input_points=points,
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"].clone()
        reshaped_sizes = inputs["reshaped_input_sizes"].clone()
        inputs = self._move(inputs)
        with self.torch.inference_mode():
            outputs = self.sam(**inputs, multimask_output=True)
        processed = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            reshaped_sizes,
        )[0]
        iou = outputs.iou_scores.detach().float().cpu()[0]
        candidates: list[tuple[float, np.ndarray]] = []
        image_area = image.width * image.height
        for index in range(processed.shape[0]):
            for candidate_index in range(processed.shape[1]):
                score = float(iou[index, candidate_index])
                mask = processed[index, candidate_index].numpy() > 0.0
                area = int(mask.sum())
                if (
                    score < self.dense_min_iou
                    or area < 16
                    or area > int(0.12 * image_area)
                    or any(
                        self._mask_iou(mask, existing) >= 0.80
                        for existing in grounded_masks
                    )
                ):
                    continue
                candidates.append((min(1.0, score), mask))
        selected: list[tuple[float, np.ndarray]] = []
        for score, mask in sorted(candidates, key=lambda item: (-item[0], item[1].sum())):
            if any(self._mask_iou(mask, existing) >= 0.75 for _, existing in selected):
                continue
            selected.append((score, mask))
            if len(selected) >= self.max_dense_proposals:
                break
        if not selected:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, image.height, image.width), dtype=bool),
                np.zeros(0, dtype=np.float32),
            )
        masks = np.stack([mask for _, mask in selected])
        return (
            np.stack([self._mask_box(mask) for mask in masks]),
            masks,
            np.asarray([score for score, _ in selected], dtype=np.float32),
        )

    def dino_token_grid(self, image: Image.Image) -> np.ndarray:
        """Return the public DINOv2 patch grid before region pooling."""
        inputs = self._move(self.dino_processor(images=image, return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.dino(**inputs).last_hidden_state[:, 1:]
        tokens = output[0].detach().float().cpu().numpy()
        side = int(round(math.sqrt(tokens.shape[0])))
        if side * side != tokens.shape[0]:
            raise ValueError(f"DINO patch count is not square: {tokens.shape[0]}")
        return tokens.reshape(side, side, tokens.shape[-1])

    def dino_features(self, token_grid: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if len(masks) == 0:
            return np.zeros((0, token_grid.shape[-1]), dtype=np.float32)
        side = int(token_grid.shape[0])
        pooled = []
        for mask in masks:
            resized = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (side, side), Image.Resampling.NEAREST
                )
            ) > 0
            if not resized.any():
                pooled.append(token_grid.mean(axis=(0, 1)))
            else:
                pooled.append(token_grid[resized].mean(axis=0))
        return np.asarray(pooled, dtype=np.float32)

    @staticmethod
    def save_dino_maps(
        *, token_grid: np.ndarray, image_size: tuple[int, int], pca_path: Path, norm_path: Path
    ) -> None:
        """Visualize frozen DINOv2 patch features without task labels.

        PCA color preserves the three dominant feature directions. Feature
        norm is a separate scalar activation view; neither is an uncertainty
        map or semantic classifier output.
        """

        height, width, dimension = token_grid.shape
        flattened = token_grid.reshape(height * width, dimension).astype(np.float64)
        centered = flattened - flattened.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        projected = (centered @ right[:3].T).reshape(height, width, 3)
        pca_rgb = np.zeros_like(projected, dtype=np.uint8)
        for channel in range(3):
            values = projected[..., channel]
            low, high = np.percentile(values, (2.0, 98.0))
            scaled = np.zeros_like(values) if high <= low else (values - low) / (high - low)
            pca_rgb[..., channel] = np.uint8(np.clip(scaled, 0.0, 1.0) * 255.0)
        Image.fromarray(pca_rgb).resize(image_size, Image.Resampling.NEAREST).save(pca_path)

        norms = np.linalg.norm(token_grid, axis=-1)
        low, high = np.percentile(norms, (2.0, 98.0))
        scaled = np.zeros_like(norms) if high <= low else (norms - low) / (high - low)
        # Compact blue-to-yellow scalar palette, implemented without plotting deps.
        norm_rgb = np.stack(
            (
                np.clip(2.0 * scaled - 0.25, 0.0, 1.0),
                np.clip(1.7 * scaled, 0.0, 1.0),
                np.clip(1.25 - 1.5 * scaled, 0.0, 1.0),
            ),
            axis=-1,
        )
        Image.fromarray(np.uint8(norm_rgb * 255.0)).resize(
            image_size, Image.Resampling.NEAREST
        ).save(norm_path)

    def process_view(
        self,
        *,
        image: Image.Image,
        view: str,
        queries: tuple[str, ...],
        asset_dir: Path,
        sample_id: str,
    ) -> tuple[list[dict[str, Any]], np.ndarray, Path, dict[str, Path]]:
        boxes, grounding_scores, label_distributions = self.detect(image, queries)
        grounded_count = len(boxes)
        grounding_overlay = image.copy()
        grounding_draw = ImageDraw.Draw(grounding_overlay)
        for index, (box, labels, score) in enumerate(
            zip(boxes, label_distributions, grounding_scores, strict=True)
        ):
            label = max(labels, key=labels.get)
            color = ((255, 72, 72), (72, 180, 255), (92, 220, 130), (255, 196, 72))[
                index % 4
            ]
            grounding_draw.rectangle(tuple(float(value) for value in box), outline=color, width=2)
            grounding_draw.text(
                (float(box[0]) + 2, float(box[1]) + 2),
                f"{label} {float(score):.2f}",
                fill=color,
            )
        masks, mask_scores = self.segment(image, boxes)
        dense_boxes, dense_masks, dense_scores = self.dense_proposals(
            image, grounded_masks=masks
        )
        if len(dense_boxes):
            boxes = np.concatenate((boxes, dense_boxes), axis=0)
            grounding_scores = np.concatenate(
                (grounding_scores, np.zeros(len(dense_boxes), dtype=np.float32))
            )
            label_distributions.extend(
                [{"unknown_object": 1.0} for _ in range(len(dense_boxes))]
            )
            masks = np.concatenate((masks, dense_masks), axis=0)
            mask_scores = np.concatenate((mask_scores, dense_scores), axis=0)
        token_grid = self.dino_token_grid(image)
        features = self.dino_features(token_grid, masks)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        sam_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        nodes: list[dict[str, Any]] = []
        colors = ((255, 72, 72), (72, 180, 255), (92, 220, 130), (255, 196, 72))
        for index, (box, labels, grounding_score, mask_score, mask, feature) in enumerate(
            zip(
                boxes,
                label_distributions,
                grounding_scores,
                mask_scores,
                masks,
                features,
                strict=True,
            )
        ):
            label = max(labels, key=labels.get)
            object_id = f"{view}_{slug(label)}_{index:02d}"
            display_id = f"{'A' if view == 'agentview' else 'W'}{index:02d}"
            mask_relative = Path(slug(sample_id)) / view / f"{object_id}_mask.png"
            mask_path = asset_dir / mask_relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            color = (180, 180, 180) if label == "unknown_object" else colors[index % len(colors)]
            sam_array[mask] = 0.45 * sam_array[mask] + 0.55 * np.asarray(color)
            draw.rectangle(tuple(float(value) for value in box), outline=color, width=2)
            draw.text(
                (float(box[0]) + 2, float(box[1]) + 2),
                f"{display_id}:{label}",
                fill=color,
            )
            nodes.append(
                {
                    "object_id": object_id,
                    "display_id": display_id,
                    "view": view,
                    "label_candidates": labels,
                    "bbox_xyxy": [float(value) for value in box],
                    "grounding_score": float(grounding_score),
                    "mask_score": float(mask_score),
                    "visible_area": int(mask.sum()),
                    "mask_rle": rle_encode(mask),
                    "mask_path": str(mask_path.relative_to(ROOT)),
                    "feature_dimension": int(feature.shape[0]),
                }
            )
        overlay_relative = Path(slug(sample_id)) / f"{view}_objects.png"
        overlay_path = asset_dir / overlay_relative
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        grounding_path = asset_dir / Path(slug(sample_id)) / f"{view}_grounding_dino.png"
        grounding_overlay.save(grounding_path)
        sam_path = asset_dir / Path(slug(sample_id)) / f"{view}_sam_masks.png"
        sam_image = Image.fromarray(np.uint8(np.clip(sam_array, 0, 255)))
        sam_draw = ImageDraw.Draw(sam_image)
        for index, (box, score) in enumerate(zip(boxes, mask_scores, strict=True)):
            sam_draw.text(
                (float(box[0]) + 2, max(0.0, float(box[1]) - 10)),
                f"{'G' if index < grounded_count else 'D'}{index:02d} {float(score):.2f}",
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        sam_image.save(sam_path)
        dino_pca_path = asset_dir / Path(slug(sample_id)) / f"{view}_dinov2_pca.png"
        dino_norm_path = asset_dir / Path(slug(sample_id)) / f"{view}_dinov2_norm.png"
        self.save_dino_maps(
            token_grid=token_grid,
            image_size=image.size,
            pca_path=dino_pca_path,
            norm_path=dino_norm_path,
        )
        return nodes, features, overlay_path, {
            "grounding_dino": grounding_path,
            "sam_masks": sam_path,
            "dinov2_pca": dino_pca_path,
            "dinov2_norm": dino_norm_path,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--grounding-model",
        type=Path,
        default=ROOT / "checkpoints/perception/grounding-dino-tiny",
    )
    parser.add_argument(
        "--sam-model",
        type=Path,
        default=ROOT / "checkpoints/perception/sam-vit-base",
    )
    parser.add_argument(
        "--dino-model",
        type=Path,
        default=ROOT / "checkpoints/perception/dinov2-small",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--dense-grid", type=int, default=16)
    parser.add_argument("--dense-min-iou", type=float, default=0.80)
    parser.add_argument("--max-dense-proposals", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-type", default="INITIAL_TASK_BELIEF")
    args = parser.parse_args()
    for name in (
        "input_index",
        "output_index",
        "feature_store",
        "asset_dir",
        "grounding_model",
        "sam_model",
        "dino_model",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    manifest = args.output_index.with_suffix(".manifest.json")
    for path in (args.output_index, args.feature_store, manifest):
        if path.exists():
            raise FileExistsError(f"immutable output already exists: {path}")
    rows = [
        json.loads(line)
        for line in args.input_index.read_text().splitlines()
        if line.strip()
    ]
    if args.sample_type:
        rows = [row for row in rows if row.get("sample_type") == args.sample_type]
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no matching rows in input index")
    for row in rows:
        validate_public_row(row)

    frontend = PublicObjectFrontend(
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        dino_model=args.dino_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou=args.nms_iou,
        precision=args.precision,
        dense_grid=args.dense_grid,
        dense_min_iou=args.dense_min_iou,
        max_dense_proposals=args.max_dense_proposals,
    )
    output_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    feature_sample_ids: list[str] = []
    feature_object_ids: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        queries = normalize_queries(row)
        objects: list[dict[str, Any]] = []
        overlay_paths: dict[str, str] = {}
        retained_source_paths: dict[str, str] = {}
        frontend_visualizations: dict[str, dict[str, str]] = {}
        feature_start = len(feature_rows)
        for view in ("agentview", "wrist"):
            image_path = ROOT / row["policy_inputs"]["image_paths"][view]
            image = Image.open(image_path).convert("RGB")
            retained_source = args.asset_dir / slug(sample_id) / f"{view}_rgb.png"
            retained_source.parent.mkdir(parents=True, exist_ok=True)
            image.save(retained_source)
            retained_source_paths[view] = str(retained_source.relative_to(ROOT))
            nodes, features, overlay, modalities = frontend.process_view(
                image=image,
                view=view,
                queries=queries,
                asset_dir=args.asset_dir,
                sample_id=sample_id,
            )
            for node, feature in zip(nodes, features, strict=True):
                node["feature_row"] = len(feature_rows)
                feature_rows.append(feature)
                feature_sample_ids.append(sample_id)
                feature_object_ids.append(str(node["object_id"]))
            objects.extend(nodes)
            overlay_paths[view] = str(overlay.relative_to(ROOT))
            frontend_visualizations[view] = {
                name: str(path.relative_to(ROOT)) for name, path in modalities.items()
            }
        output_rows.append(
            {
                "schema_version": "interaction-uncertainty.object-scene-packet.v1",
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "visual_queries": list(queries),
                "objects": objects,
                "feature_rows": [feature_start, len(feature_rows)],
                "feature_store": str(args.feature_store.relative_to(ROOT)),
                "overlay_paths": overlay_paths,
                "frontend_visualizations": frontend_visualizations,
                "source_image_paths": retained_source_paths,
                "source_metadata": row.get("metadata", {}),
                "source_image_sha256": row["policy_inputs"].get("image_sha256", {}),
                "backend": {
                    "grounding": "IDEA-Research/grounding-dino-tiny",
                    "grounding_dino_post_process": frontend.grounding_dino_post_process,
                    "segmentation": "facebook/sam-vit-base",
                    "region_features": "facebook/dinov2-small",
                },
                "online_oracle_inputs": [],
            }
        )
        print(
            f"[{row_number}/{len(rows)}] {sample_id}: {len(objects)} public objects",
            flush=True,
        )

    if feature_rows:
        feature_matrix = np.stack(feature_rows).astype(np.float32)
    else:
        feature_matrix = np.zeros((0, int(frontend.dino.config.hidden_size)), dtype=np.float32)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.feature_store.parent.mkdir(parents=True, exist_ok=True)
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_rows)
    )
    np.savez_compressed(
        args.feature_store,
        features=feature_matrix,
        sample_id=np.asarray(feature_sample_ids),
        object_id=np.asarray(feature_object_ids),
    )
    model_files = {
        "grounding": args.grounding_model / "model.safetensors",
        "sam": args.sam_model / "model.safetensors",
        "dino": args.dino_model / "model.safetensors",
    }
    report = {
        "schema_version": "interaction-uncertainty.object-scene-packet-manifest.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source": {"path": portable_path(args.input_index), "sha256": digest(args.input_index)},
        "scene_packets": {"path": str(args.output_index.relative_to(ROOT)), "sha256": digest(args.output_index)},
        "feature_store": {
            "path": str(args.feature_store.relative_to(ROOT)),
            "sha256": digest(args.feature_store),
            "shape": list(feature_matrix.shape),
        },
        "assets": str(args.asset_dir.relative_to(ROOT)),
        "samples": len(output_rows),
        "objects": len(feature_rows),
        "objects_per_sample": dict(Counter(len(row["objects"]) for row in output_rows)),
        "model_sha256": {name: digest(path) for name, path in model_files.items()},
        "thresholds": {
            "box": args.box_threshold,
            "text": args.text_threshold,
            "nms_iou": args.nms_iou,
            "dense_grid": args.dense_grid,
            "dense_min_iou": args.dense_min_iou,
            "max_dense_proposals": args.max_dense_proposals,
        },
        "precision": args.precision,
        "policy_inputs": ["agentview RGB", "wrist RGB", "prompt-derived open-vocabulary queries"],
        "online_oracle_inputs": [],
        "offline_labels_consumed": [],
        "claim_status": "frontend construction; downstream scene-disjoint gates required",
    }
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
