"""Explicit mechanism gates before a transport objective can update the VLA."""

import json
import math
from pathlib import Path

import torch


def mechanism_gate(live, blind):
    """Predeclared engineering criteria, not statistical significance claims."""
    required = ("loss", "copy_loss", "swapped_loss", "action_variance_explained")
    if any(not math.isfinite(live.get(k, float("nan"))) for k in required):
        raise ValueError("invalid qualification metrics")
    if not math.isfinite(blind.get("loss", float("nan"))):
        raise ValueError("invalid blind metrics")
    if min(live["loss"], live["copy_loss"], blind["loss"]) <= 0:
        raise ValueError("qualification losses must be positive")
    return {
        "beats_copy_10_percent": live["loss"] <= 0.9 * live["copy_loss"],
        "beats_blind_5_percent": live["loss"] <= 0.95 * blind["loss"],
        "wrong_action_cost_5_percent": live["swapped_loss"] >= 1.05 * live["loss"],
        "explains_action_variance_10_percent": live["action_variance_explained"] >= 0.1,
    }


def load_qualified_predictor(
    head_path, report_path, predictor, policy_path, config, *, policy_metadata=None
):
    """Reject stale/incompatible evidence before any predictor parameter mutation."""
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") != "transport-qualification-v1" or not report.get("passed"):
        raise ValueError("independent transport qualification has not passed")
    if (
        report.get("manual_seeds") != [17, 29, 43]
        or report.get("confirmation_families", 0) < 3
    ):
        raise ValueError(
            "qualification needs three seeds and three fresh task families"
        )
    if report.get("selected_head") != str(Path(head_path).resolve()):
        raise ValueError("qualified head differs from requested initialization")
    source = report.get("source_policy") or {}
    if source.get("checkpoint") != str(Path(policy_path).resolve()):
        raise ValueError("predictor and joint policy encoding provenance differ")
    if not report.get("disjoint_families_and_episodes") or not report.get(
        "fresh_confirmation"
    ):
        raise ValueError("confirmation split audit failed")
    if [r.get("manual_seed") for r in report.get("seed_results", [])] != [17, 29, 43]:
        raise ValueError("qualification seed rows differ")
    for pair in report.get("seed_results", []):
        if not all(mechanism_gate(pair["live"], pair["blind"]).values()):
            raise ValueError("qualification metric gate failed")
    if len(report.get("seed_results", [])) != 3:
        raise ValueError("incomplete qualification seed results")
    if policy_metadata is None or any(
        source.get(k) != policy_metadata.get(v)
        for k, v in (
            ("updates", "source_updates"),
            ("manual_seed", "source_manual_seed"),
            ("config", "source_config"),
        )
    ):
        raise ValueError("qualification differs from actual policy checkpoint metadata")
    payload = torch.load(head_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "transport-head-v1"
        or payload.get("manual_seed") != 17
        or not payload.get("normalize_context")
        or payload.get("fusion") != "control_residual"
        or payload["width"] != config.predictor_dim
        or payload["time_scale"] != config.total_steps
    ):
        raise ValueError("qualified predictor architecture differs")
    expected_kind = {"transport": "affine", "local_transport": "local"}.get(
        getattr(config, "predictor_kind", "transport")
    )
    if expected_kind is None or any(
        item.get("model_kind", "affine") != expected_kind for item in (payload, report)
    ):
        raise ValueError("qualified predictor kind differs from production config")
    state = payload["predictor"]
    expected = predictor.state_dict()
    if state.keys() != expected.keys() or any(
        t.shape != expected[k].shape or not torch.isfinite(t).all()
        for k, t in state.items()
    ):
        raise ValueError("invalid qualified predictor tensors")
    predictor.load_state_dict(state, strict=True)
    return {
        "head": str(Path(head_path).resolve()),
        "qualification": str(Path(report_path).resolve()),
    }
