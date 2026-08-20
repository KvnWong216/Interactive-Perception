"""Prompt-conditioned belief computed before VLM subtask reasoning.

This module is intentionally model-agnostic at its boundary.  A frozen SigLIP
backend supplies relative visual evidence over SAM proposals; DINO associations,
public geometry, and the prompt produce one updateable interaction field.  The
raw V0 values are explicitly uncalibrated and cannot authorize paper claims.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def normalized_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1e-12, 1.0)
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def parse_manipulation_prompt(prompt: str) -> dict[str, Any]:
    """Generic retrieval/rearrangement grammar; never names benchmark objects."""

    normalized = " ".join(prompt.strip().lower().rstrip(".").split())
    compound = re.match(
        r"^(?:please )?(?:pick up|pick|take|retrieve) (?:the )?(.+?)"
        r"(?: (?:inside|in|from) (?:the )?(.+?))? and "
        r"(?:place|put|set) it (?:in|into|inside|on|onto) (?:the )?(.+)$",
        normalized,
    )
    source_hint = None
    if compound:
        target, source_hint, destination = compound.groups()
        target = target.strip()
        source_hint = None if source_hint is None else source_hint.strip()
        destination = destination.strip()
        relation = "inside"
    else:
        match = re.match(
            r"^(?:please )?(?:pick up|place|put|move|set|retrieve|take) "
            r"(?:the )?(.+?) (?:in|into|inside|on|onto|to) (?:the )?(.+)$",
            normalized,
        )
    if not compound and match:
        target, destination = (value.strip() for value in match.groups())
        relation_match = re.search(r"\b(in|into|inside|on|onto|to)\b", normalized)
        relation = relation_match.group(1) if relation_match else "task_relation"
    elif not compound:
        match = re.match(
            r"^(?:please )?(?:find|inspect|observe|locate) (?:the )?(.+)$",
            normalized,
        )
        if not match:
            raise ValueError(
                "prompt parser supports open-vocabulary manipulation, retrieval, and inspection grammar"
            )
        target = match.group(1).strip()
        destination = None
        relation = "observe"
    if not target:
        raise ValueError("task target must be non-empty")
    facts = ["target_identity", "target_location", "target_visibility"]
    if destination is not None:
        facts.extend(
            ("target_accessibility", "destination_location", "goal_completion")
        )
    return {
        "prompt": prompt,
        "target": target,
        "destination": destination,
        "source_hint": source_hint,
        "goal_relation": relation,
        "required_facts": facts,
        "parser_stamp": "open-vocabulary-manipulation-grammar-v1",
    }


def crop_with_context(image: Any, bbox: Sequence[float], *, scale: float = 1.20) -> Any:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_width = max(2.0, (x1 - x0) * scale / 2.0)
    half_height = max(2.0, (y1 - y0) * scale / 2.0)
    return image.crop(
        (
            max(0, int(center_x - half_width)),
            max(0, int(center_y - half_height)),
            min(image.width, int(math.ceil(center_x + half_width))),
            min(image.height, int(math.ceil(center_y + half_height))),
        )
    ).convert("RGB")


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _labels(node: Mapping[str, Any]) -> str:
    return " ".join(str(value).lower() for value in node.get("label_candidates", {}))


def _container_affordance(node: Mapping[str, Any]) -> float:
    # Generic words such as "box" or weak secondary detector matches are not
    # sufficient evidence of an actuated container. Keep the detector score so
    # state evidence and openability must agree before creating unknown volume.
    scores = node.get("label_candidates", {})
    openable_words = ("drawer", "cabinet", "container", "refrigerator", "fridge")
    if not scores:
        return 0.0
    top_label = max(scores, key=scores.get)
    words = str(top_label).lower().split()
    return (
        float(scores[top_label])
        if any(word in words for word in openable_words)
        else 0.0
    )


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in first)
    bx0, by0, bx1, by1 = (float(value) for value in second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class FrozenSiglipBeliefFrontend:
    """Compute the current field before Qwen; probabilities require calibration."""

    def __init__(self, model_path: Path, *, device: str = "cuda") -> None:
        import torch
        from transformers import SiglipModel, SiglipProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.processor = SiglipProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            SiglipModel.from_pretrained(model_path, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.model_stamp = f"siglip:{model_path.name}"

    def build_field(
        self,
        *,
        prompt: str,
        scene_objects: Sequence[Mapping[str, Any]],
        images: Mapping[str, Any],
        previous_field: Mapping[str, Any] | None = None,
        executed_action: str | None = None,
        executed_target_id: str | None = None,
        public_outcome: str | None = None,
    ) -> dict[str, Any]:
        task = parse_manipulation_prompt(prompt)
        target = str(task["target"])
        previous_memory = (
            previous_field.get("control_memory", {})
            if previous_field is not None
            else {}
        )
        searched_ids = set(previous_memory.get("searched_object_ids", ()))
        if (
            executed_action == "OPEN_CONTAINER"
            and public_outcome in {"EVIDENCE_ACQUIRED", "EMPTY"}
            and executed_target_id is not None
        ):
            searched_ids.add(executed_target_id)
        searched_nodes = [
            node for node in scene_objects if str(node["object_id"]) in searched_ids
        ]
        crops = [
            crop_with_context(images[str(node["view"])], node["bbox_xyxy"])
            for node in scene_objects
        ]
        texts = [
            f"a clear image of the target object: {target}",
            f"a visually similar but different object from {target}",
            f"an unreadable or ambiguous candidate for {target}",
            "an unrelated object",
            f"an object relevant to this robot task: {prompt}",
            f"the target object {target}",
            "a clear isolated object that a robot can grasp now",
            "an object blocked, enclosed, or not currently graspable",
            "insufficient visual evidence to judge graspability",
            "a closed container with unobserved interior",
            "an open container whose interior is visible",
        ]
        inputs = self.processor(
            text=texts, images=crops, padding="max_length", return_tensors="pt"
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits_per_image.float().cpu().numpy()
        identity = _softmax(logits[:, 0:4])
        relevance = 1.0 / (1.0 + np.exp(-np.maximum(logits[:, 4], logits[:, 5])))
        graspability = _softmax(logits[:, 6:9])
        container_state = _softmax(logits[:, 9:11])

        regions: list[dict[str, Any]] = []
        visible_weights: dict[str, float] = {}
        hidden_weights: dict[str, float] = {}
        unobserved: list[dict[str, Any]] = []
        identity_names = (
            "target",
            "visually_similar_non_target",
            "insufficient_visual_evidence",
            "other",
        )
        grasp_names = ("GRASPABLE", "NOT_GRASPABLE", "INSUFFICIENT_EVIDENCE")
        target_words = set(re.findall(r"[a-z0-9]+", target.lower()))
        for index, node in enumerate(scene_objects):
            width = max(1.0, float(node["bbox_xyxy"][2]) - float(node["bbox_xyxy"][0]))
            height = max(1.0, float(node["bbox_xyxy"][3]) - float(node["bbox_xyxy"][1]))
            area = float(node.get("visible_area", width * height))
            resolution_uncertainty = 1.0 - min(1.0, area / 1024.0)
            x0, y0, x1, y1 = (float(value) for value in node["bbox_xyxy"])
            border_margin = min(x0, y0, images[str(node["view"])].width - x1, images[str(node["view"])].height - y1)
            border_uncertainty = 1.0 if border_margin < 4.0 else 0.0
            occlusion_uncertainty = min(
                1.0,
                0.55 * (1.0 - float(node.get("mask_score", 0.0)))
                + 0.45 * border_uncertainty,
            )
            label_scores = node.get("label_candidates", {})
            grounding_target_score = max(
                (
                    float(score)
                    for label, score in label_scores.items()
                    if target_words
                    & set(re.findall(r"[a-z0-9]+", str(label).lower()))
                ),
                default=0.0,
            )
            raw_identity = {
                name: float(value)
                for name, value in zip(identity_names, identity[index], strict=True)
            }
            # SigLIP alone can call any centered object the target. Grounding
            # supplies independent prompt-localized evidence; absence of such
            # evidence contributes explicit OTHER mass rather than silently
            # preserving an overconfident target score.
            identity_values = {
                "target": raw_identity["target"]
                * (0.10 + 0.90 * grounding_target_score),
                "visually_similar_non_target": raw_identity[
                    "visually_similar_non_target"
                ],
                "insufficient_visual_evidence": raw_identity[
                    "insufficient_visual_evidence"
                ],
                "other": raw_identity["other"]
                + 0.50 * (1.0 - grounding_target_score),
            }
            identity_total = sum(identity_values.values())
            identity_row = {
                name: value / identity_total
                for name, value in identity_values.items()
            }
            grasp_row = {
                name: float(value)
                for name, value in zip(grasp_names, graspability[index], strict=True)
            }
            identity_entropy = normalized_entropy(list(identity_row.values()))
            object_id = str(node["object_id"])
            container_score = _container_affordance(node)
            searched_affinity = max(
                (
                    _bbox_iou(node["bbox_xyxy"], searched["bbox_xyxy"])
                    for searched in searched_nodes
                    if searched.get("view") == node.get("view")
                ),
                default=0.0,
            )
            if object_id in searched_ids:
                searched_affinity = 1.0
            unsearched_fraction = 1.0 - searched_affinity
            closed_probability = (
                float(container_state[index, 0])
                * container_score
                * unsearched_fraction
            )
            state_uncertainty = (
                normalized_entropy(container_state[index]) * container_score
            )
            prompt_relevance = max(
                grounding_target_score,
                0.50 * float(relevance[index]),
            )
            uncertainty_mass = prompt_relevance * (
                0.60 * identity_entropy
                + 0.20 * resolution_uncertainty
                + 0.10 * occlusion_uncertainty
                + 0.10 * state_uncertainty
            )
            visible_weights[object_id] = max(
                1e-6, identity_row["target"] * prompt_relevance
            )
            if container_score > 0.0:
                hidden_weights[object_id] = max(1e-6, 0.55 * closed_probability)
                unobserved.append(
                    {
                        "object_id": object_id,
                        "target_probability": 0.0,
                        "inspectability": closed_probability,
                        "closed_probability": closed_probability,
                        "reason": "public visual container-state evidence indicates unobserved interior",
                    }
                )
            regions.append(
                {
                    "object_id": object_id,
                    "view": node["view"],
                    "bbox_xyxy": list(node["bbox_xyxy"]),
                    "visible_area": int(node.get("visible_area", 0)),
                    "prompt_relevance": prompt_relevance,
                    "grounding_target_score": grounding_target_score,
                    "identity_belief": identity_row,
                    "identity_entropy": identity_entropy,
                    "graspability_belief": grasp_row,
                    "resolution_uncertainty": resolution_uncertainty,
                    "occlusion_uncertainty": occlusion_uncertainty,
                    "state_uncertainty": state_uncertainty,
                    "closed_container_probability": closed_probability,
                    "searched_region_affinity": searched_affinity,
                    "uncertainty_mass": uncertainty_mass,
                    "raw_siglip_logits": {
                        text: float(logits[index, text_index])
                        for text_index, text in enumerate(texts)
                    },
                }
            )

        # Propagate strong target evidence across independently detected camera
        # views only when the receiving proposal also has lexical target
        # support. This recovers a small/clipped wrist target without allowing
        # an unrelated cross-view nearest neighbor to inherit target identity.
        region_by_id = {str(row["object_id"]): row for row in regions}
        for node in scene_objects:
            object_id = str(node["object_id"])
            match = node.get("cross_view_best_match") or {}
            matched_id = str(match.get("object_id", ""))
            similarity = float(match.get("dino_cosine_similarity", 0.0))
            row = region_by_id[object_id]
            if matched_id in visible_weights:
                visible_weights[object_id] = max(
                    visible_weights[object_id],
                    math.sqrt(math.sqrt(row["grounding_target_score"]))
                    * max(0.0, similarity)
                    * visible_weights[matched_id],
                )

        weights = {**visible_weights}
        for object_id, value in hidden_weights.items():
            weights[object_id] = weights.get(object_id, 0.0) + value
        evidence_acquired = public_outcome == "EVIDENCE_ACQUIRED"
        weights["OTHER_UNSEARCHED"] = 0.02 if evidence_acquired else 0.15
        weights["ABSENT"] = 0.01 if evidence_acquired else 0.05

        # Actual updates always use the new observation.  History contributes
        # only conservative persistence; FAILED never eliminates a hypothesis.
        if previous_field is not None and public_outcome != "FAILED":
            previous = previous_field.get("target_location_belief", {})
            for name in set(weights) & set(previous):
                weights[name] = 0.75 * weights[name] + 0.25 * float(previous[name])
        total = sum(weights.values())
        location = {name: value / total for name, value in weights.items()}
        for row in unobserved:
            row["target_probability"] = location.get(row["object_id"], 0.0)
        task_uncertainty = max(
            normalized_entropy(list(location.values())),
            max((row["uncertainty_mass"] for row in regions), default=0.0),
        )
        return {
            "schema_version": "interaction-uncertainty.pre-vlm-field.v1",
            "task_spec": task,
            "target_location_belief": location,
            "regions": regions,
            "unobserved_regions": unobserved,
            "task_uncertainty": task_uncertainty,
            "history_update": {
                "previous_field_used": previous_field is not None,
                "executed_action": executed_action,
                "executed_target_id": executed_target_id,
                "public_outcome": public_outcome,
            },
            "control_memory": {
                "searched_object_ids": sorted(searched_ids),
            },
            "model_stamp": self.model_stamp,
            "calibration_status": "DISPOSABLE_UNCALIBRATED_V0",
            "online_oracle_inputs": [],
        }
