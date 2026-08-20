#!/usr/bin/env python3
"""Run public RGB -> DINO/SAM ScenePacket -> Qwen PIU -> typed action once.

This is the executable V0 bridge requested by the paper design.  Perception
models and Qwen are loaded sequentially so the pipeline fits a 16 GB GPU.
The output is controller-safe: no simulator state, segmentation truth, object
pose, joint, semantic ID, or task predicate is accepted by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from build_piu_object_scene_packets import PublicObjectFrontend  # noqa: E402
from interaction_uncertainty.pre_vlm_belief import (  # noqa: E402
    FrozenSiglipBeliefFrontend,
    parse_manipulation_prompt,
)
from interaction_uncertainty.vlm_reasoner import (  # noqa: E402
    Qwen25VLSemanticReasoner,
    SemanticAction,
    assessment_from_pre_vlm_field,
    select_semantic_action,
)


ENVIRONMENT_QUERIES = (
    "object",
    "food package",
    "drawer",
    "cabinet",
    "basket",
    "bottle",
    "box",
    "bowl",
    "plate",
    "container",
)
PROMPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "from",
    "in",
    "into",
    "it",
    "of",
    "on",
    "pick",
    "place",
    "put",
    "retrieve",
    "set",
    "take",
    "the",
    "then",
    "to",
    "up",
}


def prompt_visual_queries(prompt: str) -> tuple[str, ...]:
    """Create detector hints without encoding a task- or object-specific branch."""

    task = parse_manipulation_prompt(prompt)
    noun_phrases = [
        str(value).strip()
        for value in (task.get("target"), task.get("destination"), task.get("source_hint"))
        if value
    ]
    tokens = [
        token
        for phrase in noun_phrases
        for token in re.findall(r"[a-z0-9]+", phrase.lower())
        if token not in PROMPT_STOPWORDS and len(token) > 1
    ]
    phrases = [*noun_phrases, *tokens]
    # Grounding DINO consumes referential noun phrases.  Passing the complete
    # manipulation instruction creates spurious proposals whose "label" is the
    # whole command and can dominate prompt relevance despite not denoting an
    # object.  Task semantics remain available to SigLIP/Qwen downstream.
    ordered = list(dict.fromkeys([*phrases, *ENVIRONMENT_QUERIES]))
    return tuple(value for value in ordered if value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def add_cross_view_matches(
    nodes: list[dict[str, Any]], features: np.ndarray
) -> None:
    by_view = {
        view: [index for index, row in enumerate(nodes) if row["view"] == view]
        for view in ("agentview", "wrist")
    }
    for view, other in (("agentview", "wrist"), ("wrist", "agentview")):
        for index in by_view[view]:
            vector = normalized(features[index])
            scores = [
                (float(vector @ normalized(features[candidate])), candidate)
                for candidate in by_view[other]
            ]
            if scores:
                score, candidate = max(scores)
                nodes[index]["cross_view_best_match"] = {
                    "object_id": nodes[candidate]["object_id"],
                    "display_id": nodes[candidate].get("display_id"),
                    "dino_cosine_similarity": score,
                }
            else:
                nodes[index]["cross_view_best_match"] = None


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = (float(value) for value in left)
    rx0, ry0, rx1, ry1 = (float(value) for value in right)
    x0, y0 = max(lx0, rx0), max(ly0, ry0)
    x1, y1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def associate_previous_tracks(
    *,
    nodes: list[dict[str, Any]],
    features: np.ndarray,
    previous_scene_path: Path,
    previous_features_path: Path,
    minimum_score: float = 0.55,
) -> list[dict[str, Any]]:
    """Greedy public DINO+box association; returns an explicit audit table."""

    previous_scene = json.loads(previous_scene_path.read_text())
    previous_nodes = list(previous_scene["objects"])
    with np.load(previous_features_path) as store:
        previous_features = np.asarray(store["features"], dtype=np.float32)
    candidates: list[tuple[float, int, int, float, float]] = []
    for current_index, current in enumerate(nodes):
        for previous_index, previous in enumerate(previous_nodes):
            if current["view"] != previous["view"]:
                continue
            cosine = float(
                normalized(features[current_index])
                @ normalized(previous_features[previous_index])
            )
            overlap = box_iou(current["bbox_xyxy"], previous["bbox_xyxy"])
            score = 0.70 * max(0.0, cosine) + 0.30 * overlap
            candidates.append((score, current_index, previous_index, cosine, overlap))
    used_current: set[int] = set()
    used_previous: set[int] = set()
    audit: list[dict[str, Any]] = []
    for score, current_index, previous_index, cosine, overlap in sorted(
        candidates, reverse=True
    ):
        if score < minimum_score:
            break
        if current_index in used_current or previous_index in used_previous:
            continue
        old_ephemeral_id = str(nodes[current_index]["object_id"])
        stable_id = str(previous_nodes[previous_index]["object_id"])
        nodes[current_index]["object_id"] = stable_id
        nodes[current_index]["track_source"] = "previous_public_scene_packet"
        nodes[current_index]["current_ephemeral_id"] = old_ephemeral_id
        used_current.add(current_index)
        used_previous.add(previous_index)
        audit.append(
            {
                "current_ephemeral_id": old_ephemeral_id,
                "stable_object_id": stable_id,
                "previous_object_id": stable_id,
                "association_score": score,
                "dino_cosine_similarity": cosine,
                "bbox_iou": overlap,
            }
        )
    return audit


def reasoner_subset(
    nodes: Sequence[dict[str, Any]], *, max_per_view: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for view in ("agentview", "wrist"):
        view_rows = [row for row in nodes if row["view"] == view]

        def priority(row: Mapping[str, Any]) -> tuple[float, float]:
            # Grounding was already conditioned on open-vocabulary prompt queries;
            # no benchmark object names belong in proposal selection.
            grounded = float(row.get("grounding_score", 0.0))
            area_score = min(1.0, math.sqrt(float(row.get("visible_area", 0))) / 64.0)
            return grounded, float(row.get("mask_score", 0.0)) * area_score

        selected.extend(sorted(view_rows, key=priority, reverse=True)[:max_per_view])
    return selected


def registered_action_candidates(
    *, field: Mapping[str, Any], scene_objects: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Enumerate a bounded generic action set from the pre-VLM field."""

    regions = list(field.get("regions", ()))
    region_by_id = {str(row["object_id"]): row for row in regions}
    node_by_id = {str(row["object_id"]): row for row in scene_objects}
    candidates: list[dict[str, Any]] = []
    unobserved_ids = {
        str(row["object_id"]) for row in field.get("unobserved_regions", ())
    }

    def add(
        action: SemanticAction,
        target_id: str,
        *,
        success: float,
        cost: float,
        risk: float,
        hint: str,
    ) -> None:
        pair = (action.value, target_id)
        if any((row["action"], row["target_id"]) == pair for row in candidates):
            return
        candidates.append(
            {
                "action": action.value,
                "target_id": target_id,
                "execution_success_prior": success,
                "normalized_cost_prior": cost,
                "normalized_risk_prior": risk,
                "semantic_subtask_hint": hint,
            }
        )

    wrist_regions = [
        row
        for row in regions
        if row.get("view") == "wrist"
        and str(row["object_id"]) not in unobserved_ids
    ]
    location = field["target_location_belief"]
    uncertain = sorted(
        wrist_regions,
        key=lambda row: float(row.get("uncertainty_mass", 0.0))
        + float(location.get(str(row["object_id"]), 0.0))
        * (
            0.65 * float(row.get("identity_entropy", 0.0))
            + 0.20 * float(row.get("resolution_uncertainty", 0.0))
            + 0.15 * float(row.get("occlusion_uncertainty", 0.0))
        ),
        reverse=True,
    )
    for rank, uncertain_row in enumerate(uncertain[:2]):
        target_id = str(uncertain_row["object_id"])
        display = node_by_id.get(target_id, {}).get("display_id", target_id)
        add(
            SemanticAction.MOVE_CLOSER,
            target_id,
            success=0.90,
            cost=0.06,
            risk=0.02,
            hint=f"Move the wrist camera closer to region {display} and observe it clearly.",
        )
        if rank == 0:
            add(
                SemanticAction.NEXT_BEST_VIEW,
                target_id,
                success=0.85,
                cost=0.10,
                risk=0.03,
                hint=f"Move the wrist camera to a clearer view of region {display}.",
            )

    occluders = sorted(
        wrist_regions,
        key=lambda row: (
            float(row.get("prompt_relevance", 0.0))
            * float(row.get("occlusion_uncertainty", 0.0))
            * (1.0 - float(row.get("identity_belief", {}).get("target", 0.0)))
        ),
        reverse=True,
    )
    if occluders:
        target_id = str(occluders[0]["object_id"])
        display = node_by_id.get(target_id, {}).get("display_id", target_id)
        add(
            SemanticAction.REMOVE_OCCLUDER,
            target_id,
            success=0.65,
            cost=0.20,
            risk=0.15,
            hint=f"Move region {display} aside only if it physically blocks relevant evidence.",
        )

    unobserved = sorted(
        field.get("unobserved_regions", ()),
        key=lambda row: float(row.get("target_probability", 0.0))
        * float(row.get("inspectability", 0.0)),
        reverse=True,
    )
    if unobserved:
        target_id = str(unobserved[0]["object_id"])
        display = node_by_id.get(target_id, {}).get("display_id", target_id)
        add(
            SemanticAction.OPEN_CONTAINER,
            target_id,
            success=0.95,
            # T01 physical evidence is 97/100 joint-open success.  Keep the
            # registered prior conservative without charging an unsupported
            # 0.08 failure risk on top of p_exec.
            cost=0.12,
            risk=0.03,
            hint=f"Open container region {display} and observe its interior.",
        )

    visible_ids = [
        object_id
        for object_id, row in region_by_id.items()
        if object_id in location
        and object_id not in unobserved_ids
        and row.get("view") == "wrist"
    ]
    if visible_ids:
        target_id = max(visible_ids, key=lambda value: float(location[value]))
        add(
            SemanticAction.ACT,
            target_id,
            success=0.60,
            cost=0.08,
            risk=0.10,
            hint=str(field["task_spec"]["prompt"]),
        )

    return candidates


def crop_montage(
    *,
    nodes: Sequence[Mapping[str, Any]],
    images: Mapping[str, Image.Image],
    path: Path,
    tile_size: int = 128,
    columns: int = 4,
) -> None:
    rows = max(1, math.ceil(len(nodes) / columns))
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    draw = ImageDraw.Draw(canvas)
    for index, node in enumerate(nodes):
        source = images[str(node["view"])]
        x0, y0, x1, y1 = (float(value) for value in node["bbox_xyxy"])
        pad = 6
        crop = source.crop(
            (
                max(0, int(math.floor(x0)) - pad),
                max(0, int(math.floor(y0)) - pad),
                min(source.width, int(math.ceil(x1)) + pad),
                min(source.height, int(math.ceil(y1)) + pad),
            )
        )
        crop.thumbnail((tile_size - 8, tile_size - 24), Image.Resampling.LANCZOS)
        column, row = index % columns, index // columns
        left = column * tile_size
        top = row * tile_size
        canvas.paste(crop, (left + (tile_size - crop.width) // 2, top + 20))
        label = f"{node.get('display_id')} {max(node.get('label_candidates', {'?': 1}), key=node.get('label_candidates', {'?': 1}).get)}"
        draw.rectangle((left, top, left + tile_size - 1, top + 19), fill=(20, 20, 20))
        draw.text((left + 3, top + 3), label[:20], fill=(255, 255, 255))
    canvas.save(path)


def load_mask(node: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(ROOT / str(node["mask_path"])) .convert("L")) > 0


def render_map(
    *,
    image: Image.Image,
    nodes: Sequence[Mapping[str, Any]],
    masses: Mapping[str, float],
    raw_path: Path,
    overlay_path: Path,
) -> None:
    heat = np.zeros((image.height, image.width), dtype=np.float32)
    for node in nodes:
        mass = float(masses.get(str(node["object_id"]), 0.0))
        if mass <= 0.0:
            continue
        mask = load_mask(node)
        heat = np.maximum(heat, mask.astype(np.float32) * mass)
    maximum = float(heat.max())
    normalized_heat = heat / maximum if maximum > 0.0 else heat
    Image.fromarray(np.uint8(np.clip(normalized_heat * 255.0, 0, 255))).save(raw_path)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    color[..., 1] = 210.0 * (1.0 - normalized_heat)
    alpha = (0.65 * normalized_heat)[..., None]
    rendered = np.uint8(np.clip(base * (1.0 - alpha) + color * alpha, 0, 255))
    Image.fromarray(rendered).save(overlay_path)


ACTION_COLORS = {
    "MOVE_CLOSER": (46, 134, 222),
    "NEXT_BEST_VIEW": (52, 199, 193),
    "REMOVE_OCCLUDER": (170, 84, 209),
    "OPEN_CONTAINER": (244, 151, 41),
    "ACT": (48, 184, 92),
}


def render_interaction_field(
    *,
    image: Image.Image,
    nodes: Sequence[Mapping[str, Any]],
    option_utilities: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Render one action-conditioned field: hue=action, alpha=positive utility."""

    best: dict[str, Mapping[str, Any]] = {}
    for option in option_utilities:
        target_id = str(option.get("target_id"))
        utility = option.get("utility")
        if target_id == "GLOBAL" or utility is None or not option.get("legal"):
            continue
        if target_id not in best or float(utility) > float(best[target_id]["utility"]):
            best[target_id] = option
    maximum = max((max(0.0, float(row["utility"])) for row in best.values()), default=0.0)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    rendered = base.copy()
    draw_labels: list[tuple[float, float, str, tuple[int, int, int]]] = []
    for node in nodes:
        object_id = str(node["object_id"])
        option = best.get(object_id)
        if option is None:
            continue
        utility = max(0.0, float(option["utility"]))
        strength = utility / maximum if maximum > 0.0 else 0.0
        if strength <= 0.0:
            continue
        action = str(option["action"])
        color = np.asarray(ACTION_COLORS[action], dtype=np.float32)
        mask = load_mask(node)
        alpha = 0.25 + 0.55 * strength
        rendered[mask] = rendered[mask] * (1.0 - alpha) + color * alpha
        x0, y0, _, _ = (float(value) for value in node["bbox_xyxy"])
        draw_labels.append(
            (x0, y0, f"{node.get('display_id')} {action} {utility:.2f}", tuple(color.astype(int)))
        )
    output = Image.fromarray(np.uint8(np.clip(rendered, 0, 255)))
    draw = ImageDraw.Draw(output)
    for x, y, label, color in draw_labels:
        label_x = min(max(0.0, x), float(max(0, output.width - 1)))
        label_bottom = min(
            max(12.0, y), float(max(12, output.height - 1))
        )
        label_right = min(
            float(output.width - 1),
            label_x + min(170.0, 6.0 * len(label)),
        )
        draw.rectangle(
            (label_x, label_bottom - 12.0, label_right, label_bottom),
            fill=(0, 0, 0),
        )
        draw.text((label_x + 1.0, label_bottom - 11.0), label, fill=color)
    output.save(path)


def action_contract(
    action: SemanticAction,
    target_id: str | None,
    stop_reason: str | None,
    prompt: str,
    nodes: Mapping[str, Mapping[str, Any]],
    interaction_options: Sequence[Any],
) -> dict[str, Any]:
    target = nodes.get(target_id, {}) if target_id is not None else {}
    selected_option = next(
        (
            option
            for option in interaction_options
            if option.action is action and option.target_id == target_id
        ),
        None,
    )
    semantic_subtask = (
        selected_option.semantic_subtask if selected_option is not None else None
    )
    if action is SemanticAction.MOVE_CLOSER:
        return {
            "mode": "OBSERVE",
            "primitive": "MOVE_CLOSER",
            "executor": "public RGB image-space observe controller",
            "target_id": target_id,
            "target_view": target.get("view"),
            "target_bbox_xyxy": target.get("bbox_xyxy"),
            "query": semantic_subtask,
        }
    if action is SemanticAction.NEXT_BEST_VIEW:
        return {
            "mode": "OBSERVE",
            "primitive": "NEXT_BEST_VIEW",
            "executor": "public RGB next-best-view controller",
            "target_id": target_id,
            "target_view": target.get("view"),
            "target_bbox_xyxy": target.get("bbox_xyxy"),
            "query": semantic_subtask,
        }
    if action is SemanticAction.REMOVE_OCCLUDER:
        return {
            "mode": "OBSERVE",
            "primitive": "REMOVE_OCCLUDER",
            "executor": "registered frozen manipulation option plus public-RGB observation",
            "target_id": target_id,
            "subtask": semantic_subtask,
        }
    if action is SemanticAction.OPEN_CONTAINER:
        return {
            "mode": "OBSERVE",
            "primitive": "OPEN_CONTAINER",
            "executor": "frozen pi0.5 drawer-opening option followed by public-RGB observation",
            "target_id": target_id,
            "subtask": semantic_subtask,
        }
    if action is SemanticAction.ACT:
        return {
            "mode": "ACT",
            "primitive": "DIRECT_ACT",
            "executor": "frozen pi0.5",
            "target_id": target_id,
            "subtask": semantic_subtask or prompt,
        }
    return {
        "mode": "STOP",
        "primitive": "STOP",
        "executor": None,
        "target_id": None,
        "reason": stop_reason,
        "report": "target not found" if stop_reason == "NOT_FOUND" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentview", type=Path, required=True)
    parser.add_argument("--wrist", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--previous-report", type=Path)
    parser.add_argument(
        "--executed-action",
        choices=[action.value for action in SemanticAction if action is not SemanticAction.STOP],
    )
    parser.add_argument(
        "--observed-outcome",
        choices=("FAILED", "EVIDENCE_ACQUIRED", "EMPTY", "AMBIGUOUS", "TASK_PROGRESS", "TASK_COMPLETED"),
    )
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
    parser.add_argument(
        "--siglip-model",
        type=Path,
        default=ROOT / "checkpoints/perception/siglip-base-patch16-224",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=ROOT / "checkpoints/perception/qwen2.5-vl-3b-instruct",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-proposals-per-view", type=int, default=8)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "agentview",
        "wrist",
        "grounding_model",
        "sam_model",
        "dino_model",
        "siglip_model",
        "qwen_model",
        "asset_dir",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.history is not None and not args.history.is_absolute():
        args.history = ROOT / args.history
    if args.previous_report is not None and not args.previous_report.is_absolute():
        args.previous_report = ROOT / args.previous_report
    if args.asset_dir.exists() or args.output.exists():
        raise FileExistsError("pipeline artifacts are immutable")
    for path in (
        args.agentview,
        args.wrist,
        args.grounding_model,
        args.sam_model,
        args.dino_model,
        args.siglip_model,
        args.qwen_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    history = []
    if args.history is not None:
        loaded = json.loads(args.history.read_text())
        history = loaded if isinstance(loaded, list) else [loaded]
    previous_report = None
    if args.previous_report is not None:
        if args.executed_action is None or args.observed_outcome is None:
            raise ValueError(
                "previous-report updates require executed-action and observed-outcome"
            )
        previous_report = json.loads(args.previous_report.read_text())
        if previous_report.get("prompt") != args.prompt:
            raise ValueError("belief updates must preserve the complete task prompt")
        if previous_report.get("online_oracle_inputs"):
            raise ValueError("previous field contains forbidden online oracle inputs")
        history.append(
            {
                "previous_interaction_field": previous_report[
                    "prompt_conditioned_interaction_field"
                ],
                "executed_action": args.executed_action,
                "public_observed_outcome": args.observed_outcome,
                "searched_or_eliminated_memory": previous_report.get(
                    "public_history", []
                ),
            }
        )
    args.asset_dir.mkdir(parents=True)
    images = {
        "agentview": Image.open(args.agentview).convert("RGB"),
        "wrist": Image.open(args.wrist).convert("RGB"),
    }

    frontend = PublicObjectFrontend(
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        dino_model=args.dino_model,
        device=args.device,
        box_threshold=0.25,
        text_threshold=0.20,
        nms_iou=0.70,
        precision="float32",
        dense_grid=16,
        dense_min_iou=0.80,
        max_dense_proposals=12,
    )
    nodes: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    overlays: dict[str, Path] = {}
    visual_queries = prompt_visual_queries(args.prompt)
    for view in ("agentview", "wrist"):
        view_nodes, features, overlay = frontend.process_view(
            image=images[view],
            view=view,
            queries=visual_queries,
            asset_dir=args.asset_dir,
            sample_id="current_observation",
        )
        nodes.extend(view_nodes)
        feature_rows.extend(features)
        overlays[view] = overlay
    feature_matrix = np.asarray(feature_rows, dtype=np.float32)
    temporal_association: list[dict[str, Any]] = []
    if previous_report is not None:
        previous_scene_path = ROOT / previous_report["scene_packet"]["path"]
        previous_features_path = ROOT / previous_report["scene_packet"]["dino_features"]
        temporal_association = associate_previous_tracks(
            nodes=nodes,
            features=feature_matrix,
            previous_scene_path=previous_scene_path,
            previous_features_path=previous_features_path,
        )
    add_cross_view_matches(nodes, feature_matrix)
    selected_nodes = reasoner_subset(nodes, max_per_view=args.max_proposals_per_view)
    selected_ids = {str(row["object_id"]) for row in selected_nodes}
    selected_feature_indices = [
        index for index, row in enumerate(nodes) if str(row["object_id"]) in selected_ids
    ]
    crop_path = args.asset_dir / "candidate_crops.png"
    crop_montage(nodes=selected_nodes, images=images, path=crop_path)
    scene_path = args.asset_dir / "scene_packet.json"
    feature_path = args.asset_dir / "dino_features.npz"
    scene_packet = {
        "schema_version": "interaction-uncertainty.live-scene-packet.v1",
        "prompt": args.prompt,
        "lexical_query_hints": list(visual_queries),
        "objects": nodes,
        "reasoner_object_ids": sorted(selected_ids),
        "source_images": {
            "agentview": str(args.agentview.relative_to(ROOT)),
            "wrist": str(args.wrist.relative_to(ROOT)),
        },
        "online_oracle_inputs": [],
        "temporal_association": temporal_association,
    }
    scene_path.write_text(json.dumps(scene_packet, indent=2) + "\n")
    np.savez_compressed(
        feature_path,
        features=feature_matrix,
        object_id=np.asarray([row["object_id"] for row in nodes]),
        reasoner_feature_rows=np.asarray(selected_feature_indices),
    )

    # Current belief is computed before the VLM. Models run sequentially so a
    # single 16 GB GPU never holds perception, SigLIP, and Qwen together.
    torch = frontend.torch
    del frontend
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    previous_field = (
        None
        if previous_report is None
        else previous_report["prompt_conditioned_interaction_field"]
    )
    belief_frontend = FrozenSiglipBeliefFrontend(
        args.siglip_model, device=args.device
    )
    pre_vlm_field = belief_frontend.build_field(
        prompt=args.prompt,
        scene_objects=selected_nodes,
        images=images,
        previous_field=previous_field,
        executed_action=args.executed_action,
        executed_target_id=(
            None
            if previous_report is None
            else previous_report.get("selected_action", {}).get("target_id")
        ),
        public_outcome=args.observed_outcome,
    )
    del belief_frontend
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidates = registered_action_candidates(
        field=pre_vlm_field, scene_objects=selected_nodes
    )

    reasoner = Qwen25VLSemanticReasoner(args.qwen_model, device=args.device)
    qwen_images = [
        images["agentview"],
        Image.open(overlays["agentview"]).convert("RGB"),
        images["wrist"],
        Image.open(overlays["wrist"]).convert("RGB"),
        Image.open(crop_path).convert("RGB"),
    ]
    try:
        effects, raw = reasoner.assess_effects(
            images=qwen_images,
            prompt=args.prompt,
            pre_vlm_field=pre_vlm_field,
            scene_objects=selected_nodes,
            registered_candidates=candidates,
            belief_history=history,
        )
    except ValueError:
        failure_path = args.asset_dir / "qwen_failed_attempts.json"
        failure_path.write_text(json.dumps(reasoner.failed_attempts, indent=2) + "\n")
        raise
    assessment = assessment_from_pre_vlm_field(
        pre_vlm_field,
        effects,
        allowed_object_ids=[str(row["object_id"]) for row in selected_nodes],
    )
    decision = select_semantic_action(assessment, scene_objects=selected_nodes)
    raw_path = args.asset_dir / "qwen_raw.json"
    raw_path.write_text(raw.strip() + "\n")

    uncertainty_masses = {
        str(row["object_id"]): float(row["uncertainty_mass"])
        for row in pre_vlm_field["regions"]
    }
    for row in pre_vlm_field["unobserved_regions"]:
        uncertainty_masses[str(row["object_id"])] = max(
            uncertainty_masses.get(str(row["object_id"]), 0.0),
            float(row["target_probability"]),
        )
    map_paths: dict[str, dict[str, str]] = {}
    for view in ("agentview", "wrist"):
        view_nodes = [row for row in selected_nodes if row["view"] == view]
        view_paths: dict[str, str] = {}
        raw_map = args.asset_dir / f"{view}_uncertainty_map.png"
        overlay_map = args.asset_dir / f"{view}_uncertainty_overlay.png"
        render_map(
            image=images[view],
            nodes=view_nodes,
            masses=uncertainty_masses,
            raw_path=raw_map,
            overlay_path=overlay_map,
        )
        field_path = args.asset_dir / f"{view}_interaction_field.png"
        render_interaction_field(
            image=images[view],
            nodes=view_nodes,
            option_utilities=decision.option_utilities,
            path=field_path,
        )
        view_paths["uncertainty_map"] = str(raw_map.relative_to(ROOT))
        view_paths["uncertainty_overlay"] = str(overlay_map.relative_to(ROOT))
        view_paths["interaction_field"] = str(field_path.relative_to(ROOT))
        map_paths[view] = view_paths

    node_lookup = {str(row["object_id"]): row for row in selected_nodes}
    report = {
        "schema_version": "interaction-uncertainty.qwen-observation-pipeline.v0",
        "claim_status": "disposable V0 inference; uncalibrated and not clean/sealed evidence",
        "prompt": args.prompt,
        "task_spec": {
            "target": pre_vlm_field["task_spec"]["target"],
            "destination": pre_vlm_field["task_spec"]["destination"],
            "goal_relation": pre_vlm_field["task_spec"]["goal_relation"],
            "required_facts": pre_vlm_field["task_spec"]["required_facts"],
        },
        "pipeline": [
            "public agentview/wrist RGB",
            "Grounding DINO open-vocabulary proposals",
            "SAM masks",
            "DINOv2 region features and cross-view matches",
            "frozen SigLIP prompt-conditioned current belief before VLM",
            "separate Qwen2.5-VL action-conditioned future-effect prediction",
            "deterministic information-utility selector",
        ],
        "model_paths": {
            "grounding_dino": str(args.grounding_model.relative_to(ROOT)),
            "sam": str(args.sam_model.relative_to(ROOT)),
            "dinov2": str(args.dino_model.relative_to(ROOT)),
            "siglip": str(args.siglip_model.relative_to(ROOT)),
            "qwen": str(args.qwen_model.relative_to(ROOT)),
        },
        "scene_packet": {
            "path": str(scene_path.relative_to(ROOT)),
            "sha256": digest(scene_path),
            "all_proposals": len(nodes),
            "reasoner_proposals": len(selected_nodes),
            "dino_features": str(feature_path.relative_to(ROOT)),
            "candidate_crops": str(crop_path.relative_to(ROOT)),
            "object_overlays": {
                view: str(path.relative_to(ROOT)) for view, path in overlays.items()
            },
        },
        "pre_vlm_current_field": pre_vlm_field,
        "qwen_action_effect_assessment": effects.to_dict(),
        "joined_planning_assessment": assessment.to_dict(),
        "registered_action_candidates": candidates,
        "prompt_conditioned_interaction_field": {
            **pre_vlm_field,
            "region_uncertainty_mass": uncertainty_masses,
            "action_conditioned_effects": [
                option.to_dict(current_uncertainty=decision.task_uncertainty)
                for option in assessment.interaction_options
            ],
            "continuous_utility_audit": [
                dict(value) for value in decision.option_utilities
            ],
            "update_contract": "new public RGB + previous field + executed action/outcome -> next field",
        },
        "selected_action": decision.to_dict(),
        "execution_contract": action_contract(
            decision.action,
            decision.target_id,
            decision.stop_reason,
            args.prompt,
            node_lookup,
            assessment.interaction_options,
        ),
        "visualizations": map_paths,
        "public_history": history,
        "previous_report": (
            {
                "path": str(args.previous_report.relative_to(ROOT)),
                "sha256": digest(args.previous_report),
                "executed_action": args.executed_action,
                "public_observed_outcome": args.observed_outcome,
            }
            if args.previous_report is not None
            else None
        ),
        "online_oracle_inputs": [],
        "limitations": [
            "SigLIP-derived current probabilities are disposable V0 and are not calibrated",
            "Qwen predicts future action effects only; those effect probabilities are not calibrated",
            "ACT is not paper-authorized until the continuous readiness/effect model is scene-disjoint calibrated",
            "this one-step runner does not itself execute the selected physical action",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_action": decision.action.value,
                "target_id": decision.target_id,
                "prediction_set": list(decision.target_prediction_set),
                "task_uncertainty": decision.task_uncertainty,
                "report": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
