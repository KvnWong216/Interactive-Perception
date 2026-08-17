#!/usr/bin/env python3
"""One-time pre-freeze migration for the interrupted v1 development collection.

The first formal process was already running when the EMPTY oracle issue was
found. Preserve its original dataset/manifest, then apply the same conservative
seed-paired counterfactual rule now implemented by the collector. This command
must run before feature extraction or critic fitting and is intentionally
ineligible once the critic artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_action_effect_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/calibration/t01_action_effect_v1.manifest.json",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "outputs/t01_action_effect_v1_pre_empty_counterfactual_label",
    )
    parser.add_argument(
        "--critic-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_v1.json",
    )
    args = parser.parse_args()
    for name in ("dataset", "manifest", "archive_dir", "critic_artifact"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.critic_artifact.exists():
        raise FileExistsError("cannot revise labels after the critic artifact is frozen")
    if not args.dataset.exists() or not args.manifest.exists():
        raise FileNotFoundError("completed dataset and manifest are required")
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_sha256"] != digest(args.dataset):
        raise ValueError("pre-migration dataset hash does not match its manifest")
    if manifest.get("empty_label_contract"):
        raise ValueError("dataset already uses the counterfactual EMPTY contract")

    args.archive_dir.mkdir(parents=True, exist_ok=False)
    archived_dataset = args.archive_dir / args.dataset.name
    archived_manifest = args.archive_dir / args.manifest.name
    shutil.copy2(args.dataset, archived_dataset)
    shutil.copy2(args.manifest, archived_manifest)

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    revealed = {
        int(row["seed"]): row["outcome"] == "REVEALED"
        for row in rows
        if row["regime"] == "revealed_full"
    }
    changed = []
    for row in rows:
        if row["regime"] != "empty_full":
            row["evaluator_only"].setdefault(
                "empty_counterfactual_reveal_certified", None
            )
            continue
        seed = int(row["seed"])
        if seed not in revealed:
            raise ValueError(f"missing target-present counterpart for seed {seed}")
        certified = revealed[seed]
        row["evaluator_only"]["empty_counterfactual_reveal_certified"] = certified
        if row["outcome"] == "EMPTY" and not certified:
            row["outcome"] = "FAILED"
            changed.append(seed)

    temporary = args.dataset.with_suffix(".counterfactual.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(args.dataset)
    manifest["dataset_sha256"] = digest(args.dataset)
    manifest["empty_label_contract"] = (
        "EMPTY additionally requires the seed-matched target-present counterpart "
        "to reach REVEALED; otherwise OPENED_UNOBSERVED is FAILED"
    )
    manifest["label_revision"] = {
        "reason": "prevent simulator-absence oracle from authorizing EMPTY when the searched volume was not visible",
        "changed_empty_seeds": changed,
        "archived_dataset": str(archived_dataset.relative_to(ROOT)),
        "archived_dataset_sha256": digest(archived_dataset),
        "archived_manifest": str(archived_manifest.relative_to(ROOT)),
        "archived_manifest_sha256": digest(archived_manifest),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "changed_empty_seeds": changed,
                "dataset_sha256": manifest["dataset_sha256"],
                "archive_dir": str(args.archive_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
