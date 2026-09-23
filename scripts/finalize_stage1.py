"""Finalize conservative scene-family splits and audit every exported episode."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

from grounded_interaction.predictive_vla.config import load_config
from grounded_interaction.predictive_vla.data import TrajectoryDataset

ROOT = Path(__file__).resolve().parents[1]


def asset_family(xml):
    """Ignore task, poses and XML ordering; retain the whole scene's asset inventory."""
    root = ET.fromstring(xml)
    assets = []
    for node in root.find("asset"):
        file = node.get("file")
        if file:
            parts = Path(file).parts
            marker = "assets" if "assets" in parts else "robosuite"
            if marker in parts:
                file = "/".join(
                    parts[max(i for i, p in enumerate(parts) if p == marker) :]
                )
            else:
                raise ValueError("unknown source asset path")
            assets.append((node.tag, file, node.get("scale", "1 1 1")))
    bodies = sorted(
        node.get("name", "") for node in root.find("worldbody") if node.tag == "body"
    )
    signature = json.dumps({"assets": sorted(assets), "bodies": bodies}, sort_keys=True)
    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared/stage1")
    parser.add_argument("--manual-seed", type=int, default=17)
    parser.add_argument("--minimum-episodes", type=int, default=200)
    args = parser.parse_args()
    root = (ROOT / args.data).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("data directory escaped repository")
    report = json.loads((root / "preparation_report.json").read_text())
    if report["audit_only"] or len(report["episodes"]) != report["planned_episodes"]:
        raise ValueError("candidate preparation is incomplete")
    manifest = json.loads((root / "manifest.json").read_text())
    grouped = {}
    for entry in manifest["episodes"]:
        provenance = json.loads((root / entry["path"]).with_suffix(".json").read_text())
        source = (ROOT / provenance["source"]).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError("source escaped repository")
        grouped.setdefault(source, []).append((entry, provenance))
    for source, examples in grouped.items():
        with h5py.File(source, "r") as archive:
            for entry, provenance in examples:
                family = asset_family(
                    archive[f"data/{provenance['episode']}"].attrs["model_file"]
                )
                entry["reset_family"] = family
    family_names = {
        family: f"scene_{i + 1:02d}"
        for i, family in enumerate(
            sorted({e["reset_family"] for e in manifest["episodes"]})
        )
    }
    for entry in manifest["episodes"]:
        entry["reset_family"] = family_names[entry["reset_family"]]
    counts = {
        family: sum(e["reset_family"] == family for e in manifest["episodes"])
        for family in {e["reset_family"] for e in manifest["episodes"]}
    }
    families = sorted(counts)
    np.random.default_rng(args.manual_seed).shuffle(families)
    if len(families) < 3:
        raise ValueError("insufficient disjoint scene families")
    largest = max(families, key=counts.get)
    candidates = [f for f in families if f != largest]
    target = len(manifest["episodes"]) * 0.1

    def take_families():
        selected, total = set(), 0
        while candidates:
            choice = min(candidates, key=lambda f: abs(total + counts[f] - target))
            if selected and abs(total + counts[choice] - target) >= abs(total - target):
                break
            candidates.remove(choice)
            selected.add(choice)
            total += counts[choice]
        return selected

    validation, test = take_families(), take_families()
    if not validation or not test:
        raise ValueError("cannot form disjoint validation and test families")
    for entry in manifest["episodes"]:
        entry["split"] = (
            "validation"
            if entry["reset_family"] in validation
            else "test"
            if entry["reset_family"] in test
            else "train"
        )
    temporary = root / "manifest.finalizing.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    dataset = TrajectoryDataset(
        temporary, load_config(ROOT / "experiments/stage1_policy.yaml")
    )
    started = time.monotonic()

    def progress(completed, total):
        if completed % 25 and completed != total:
            return
        value = {
            "manual_seed": args.manual_seed,
            "workers": 16,
            "completed": completed,
            "total": total,
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary = root / "validation_progress.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(root / "validation_progress.json")
        print(json.dumps({"final_validation": value}), flush=True)

    summary = dataset.validate(workers=16, progress=progress)
    if (
        summary["episodes"] < args.minimum_episodes
        or summary["rgbd_episodes"] != summary["episodes"]
    ):
        raise ValueError("insufficient calibrated episodes for the chosen data budget")
    temporary.replace(root / "manifest.json")
    audit = {
        "passed": True,
        "summary": summary,
        "grouping": "whole scene asset inventory and top-level bodies, independent of XML order/task/reset pose",
        "family_counts": counts,
        "manual_seed": args.manual_seed,
        "unseen_to_pretrained_base": False,
    }
    (root / "final_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"manual_seed": args.manual_seed, **summary}), flush=True)


if __name__ == "__main__":
    main()
