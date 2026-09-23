"""Evaluate three frozen heads and action-blind controls on NEW confirmation forks.

No fitting or candidate selection is done here. Seed 17 is preselected for joint
initialization. Reused September diagnostic confirmation data are prohibited.
"""

import argparse
import json
from pathlib import Path

import torch
from diagnose_transition_targets import ROOT, SOURCE, load
from train_transition_predictor import evaluate

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.qualification import mechanism_gate
from grounded_interaction.predictive_vla.transport import (
    LocalTransportPredictor,
    TransportPredictor,
)


def inside(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("paths must stay in repository")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--features-directory", required=True)
    p.add_argument("--live", nargs=3, required=True)
    p.add_argument("--blind", nargs=3, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source = inside(a.source)
    features = inside(a.features_directory)
    output = inside(a.output)
    if source == SOURCE.resolve():
        raise ValueError(
            "old confirmation was already examined; collect new confirmation cases"
        )
    if output.exists():
        raise ValueError("refusing to overwrite a qualification decision")
    provenance = json.loads((features / "encoding.json").read_text())
    set_manual_seed(8227)
    torch.set_num_threads(2)
    cases = load("cuda", source, features, splits=("confirmation",))
    if not cases:
        raise ValueError("missing fresh confirmation cases")
    families = {c["case"]["family"] for c in cases}
    episodes = {(c["case"]["source"], c["case"]["episode"]) for c in cases}
    if len(families) < 3:
        raise ValueError("confirmation requires at least three new task/scene families")
    results = []
    locked_settings = None
    confirmation_profile = json.loads((source / "plan.json").read_text()).get(
        "render_profile", "native"
    )
    for seed, live_path, blind_path in zip([17, 29, 43], a.live, a.blind, strict=True):
        row = {}
        fit_settings = []
        for label, path in [("live", live_path), ("blind", blind_path)]:
            path = inside(path)
            head = torch.load(path, map_location="cpu", weights_only=True)
            report = json.loads(inside(head["report"]).read_text())
            if (
                report["manual_seed"] != seed
                or head["manual_seed"] != seed
                or report["blind"] != (label == "blind")
                or report.get("zero_context")
                or not report.get("normalize_context")
                or report.get("fusion") != "control_residual"
            ):
                raise ValueError("seed or comparison condition mismatch")
            if report.get("source_policy") != provenance:
                raise ValueError("policy encoding provenance mismatch")
            if report.get("render_profile", "native") != confirmation_profile:
                raise ValueError("confirmation and fitting render profiles differ")
            fit_settings.append(
                {
                    k: report.get(k)
                    for k in (
                        "steps",
                        "learning_rate",
                        "effective_batch",
                        "optimizer",
                        "source",
                        "features_directory",
                        "render_profile",
                        "model_kind",
                        "effect_weight",
                    )
                }
            )
            used = report["result"]["rows"]
            if families & {r["case"]["family"] for r in used} or episodes & {
                (r["case"]["source"], r["case"]["episode"]) for r in used
            }:
                raise ValueError(
                    "confirmation overlaps fitting/development families or episodes"
                )
            kind = head.get("model_kind", "affine")
            if kind not in {"affine", "local"} or kind != report.get(
                "model_kind", "affine"
            ):
                raise ValueError("unsupported or mismatched predictor architecture")
            if head.get("effect_weight", 0.0) != report.get("effect_weight", 0.0):
                raise ValueError("predictor objective metadata mismatch")
            model_class = (
                LocalTransportPredictor if kind == "local" else TransportPredictor
            )
            model = model_class(
                head["native_width"], width=head["width"], time_scale=head["time_scale"]
            ).cuda()
            model.load_state_dict(head["predictor"], strict=True)
            model.eval()
            result = evaluate(model, cases, blind=label == "blind")
            row[label] = result["summary"]["confirmation"]
            row[f"{label}_rows"] = result["rows"]
        if fit_settings[0] != fit_settings[1]:
            raise ValueError("live and blind training budgets differ")
        if locked_settings is not None and fit_settings[0] != locked_settings:
            raise ValueError("training settings differ across manual seeds")
        locked_settings = fit_settings[0]
        row["manual_seed"] = seed
        row["checks"] = mechanism_gate(row["live"], row["blind"])
        results.append(row)
    report = {
        "schema": "transport-qualification-v1",
        "model_kind": kind,
        "fit_settings": locked_settings,
        "manual_seeds": [17, 29, 43],
        "confirmation_families": len(families),
        "confirmation_cases": len(cases),
        "source_policy": provenance,
        "confirmation_source": str(source),
        "render_profile": json.loads((source / "plan.json").read_text()).get(
            "render_profile", "native"
        ),
        "disjoint_families_and_episodes": True,
        "fresh_confirmation": True,
        "selected_head": str(inside(a.live[0])),
        "seed_results": results,
        "passed": all(all(r["checks"].values()) for r in results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(output)}))


if __name__ == "__main__":
    main()
