from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/calibrated_interaction/original_drawer_v1"
FEATURES = ROOT / "outputs/calibrated_interaction/original_drawer_v1"
SPLITS = ("train", "development", "calibration", "test")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_original_drawer_prompt_pairs_are_grouped_identical_rgb() -> None:
    seed_owners: dict[int, str] = {}
    forbidden = {
        "target_instance",
        "target_location",
        "target_pixels",
        "target_mask_policy_resolution_rle",
        "prompt_resolvable_initially",
    }
    for split in SPLITS:
        public = read_jsonl(DATA / f"{split}.jsonl")
        labels = read_jsonl(DATA / f"{split}_labels.jsonl")
        assert {row["sample_id"] for row in public} == {
            row["sample_id"] for row in labels
        }
        paired: dict[int, list[dict]] = defaultdict(list)
        for row in public:
            assert row["online_oracle_inputs"] == []
            assert forbidden.isdisjoint(row)
            assert forbidden.isdisjoint(row["policy_inputs"])
            seed = int(row["seed"])
            assert seed_owners.setdefault(seed, split) == split
            paired[seed].append(row)
            for view, path in row["policy_inputs"]["image_paths"].items():
                expected = row["policy_inputs"]["image_sha256"][view]
                assert (
                    hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
                )
        for rows in paired.values():
            assert len(rows) == 2
            assert rows[0]["prompt"] != rows[1]["prompt"]
            assert (
                rows[0]["policy_inputs"]["image_sha256"]
                == rows[1]["policy_inputs"]["image_sha256"]
            )


def test_original_drawer_shared_vlm_features_match_public_indices() -> None:
    expected_candidates = (
        "direct_requested_to_basket",
        "open_middle_drawer",
        "stop_unsupported",
    )
    for split in SPLITS:
        public = read_jsonl(DATA / f"{split}.jsonl")
        store = np.load(FEATURES / split / "shared_vlm_features.npz")
        manifest = json.loads(
            (FEATURES / split / "shared_vlm_features.json").read_text()
        )
        assert set(store["sample_id"].astype(str)) == {
            row["sample_id"] for row in public
        }
        assert store["context_tokens"].shape == (len(public), 4, 2048)
        assert store["candidate_tokens"].shape == (len(public), 3, 2048)
        assert tuple(store["candidate_id"].astype(str)) == expected_candidates
        assert np.isfinite(store["context_tokens"]).all()
        assert np.isfinite(store["candidate_tokens"]).all()
        assert manifest["shared_encoder_for_context_and_candidates"] is True
        assert manifest["online_oracle_inputs"] == []
        assert manifest["route_or_effect_labels_loaded"] is False
