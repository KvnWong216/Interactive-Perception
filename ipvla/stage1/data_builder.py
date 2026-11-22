#!/usr/bin/env python3
"""Build Formal Stage-1 Mixed Video Pretraining Suite (100% English, ZERO ActiveArena data).

Sources (strictly Stage-1 video data per VIDEO_TRAINING_RECIPE.md):
1. RoboTurk Object Search causal-checked prefixes (883 deduplicated train rows across 71 recordings,
   23 operators; plus sequential target-specific evidence-reversal pairs and held-out unseen operators).
2. OS/OST Part 3 Human Egocentric Object Search causal-checked prefixes (1,250 deduplicated rows across
   12 recordings: sequences >=8 [os:A11, os:A14, os:B10, os:B15, os:C8, os:C17 = 750 rows] in train;
   sequences 1..7 split strictly by participant into validation [os:B5, os:B7 = 247 rows] and held-out
   test [os:A1, os:A7, os:B3 = 253 rows]).
3. Target-specific evidence-reversal pairs (pre-reveal, post-reveal full causal history, and counterfactual
   masked history with byte-identical current frame) extracted from sequential (search/reveal -> grasp
   concrete named object) events in RoboTurk and OS/OST Part 3.
4. Activity-description semantic alignment slice (20% general visual-language alignment prompt, tracked
   separately from next_task_selection per VIDEO_TRAINING_RECIPE.md).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/reports/rss2027_autoresearch")
FULL_SOL = ROOT / "data/teacher_chunks/full_sol_v1"

STOP_NOUNS = {
    "object", "objects", "item", "items", "it", "them", "cloth", "fabric", "towel", "cover",
    "covering", "obstruction", "area", "bin", "box", "opening", "work area", "visible work area",
    "clutter", "pile", "container", "table", "surface", "hand", "gripper", "arm", "material",
    "covering material", "obstructing material", "occluding cloth", "something", "anything",
    "papers", "sheet", "printed sheet", "form", "checklist", "room", "counter", "desk", "shelf",
}

EXCLUDED_CASES = {
    "3afcea53e8af1c43-000-s008",
    "1ca7691f86b30819-001-s004",
    "1ca7691f86b30819-003-s001",
}

OS_VAL_PARTICIPANTS = {"os:B5", "os:B7"}
OS_TEST_PARTICIPANTS = {"os:A1", "os:A7", "os:B3"}
OS_TRAIN_PARTICIPANTS = {"os:A11", "os:A14", "os:B10", "os:B15", "os:C8", "os:C17"}


def classify_skill(text: str) -> str:
    s = text.strip().lower()
    # 1. Search / clear occlusion / inspect / scan
    if re.search(
        r"\b(?:move|push|pull|shift|sweep|lift|clear|displace)\b.*?\b(?:aside|away|obstruction|cloth|fabric|towel|cover|covering|clutter|material)\b",
        s,
    ):
        return "search"
    if any(w in s for w in (
        "move aside", "push aside", "pull aside", "shift", "uncover", "reveal", "clear",
        "probe", "search", "inspect", "look", "scan", "check", "locate", "find", "examine",
        "open the bag", "open the drawer", "open the box", "open the container", "sweep", "displace",
        "change viewpoint", "changing viewpoints", "turn the ", "rotate the "
    )) and not s.startswith(("grasp", "pick up", "place", "put", "carry", "transfer", "drop", "release", "set ")):
        return "search"
    # 2. Place / set down / align / insert / pack
    if re.search(r"\b(?:set|place|put|drop|release|insert|align|deposit|pack|lower)\b.*?\b(?:on|onto|into|in|down|at|table|box|opening|slot|bag|location)\b", s) and not s.startswith(("reach ", "grasp ", "pick up ", "search ", "inspect ", "scan ")):
        return "place"
    if any(w in s for w in ("place", "put", "drop", "release", "insert", "align", "deposit", "set down", "pack")):
        return "place"
    # 3. Carry / transport / transfer / bring
    if any(w in s for w in ("carry", "transfer", "transport", "move the grasped", "move the held", "move the object to", "bring")):
        return "carry"
    # 4. Acquire / grasp / pick up / reach for
    if any(w in s for w in ("grasp", "pick up", "retrieve", "grab", "lift", "take", "hold", "regrasp", "pull out", "acquire", "reach for", "reach into", "reach down")):
        return "acquire"
    # 5. Return / retract
    if any(w in s for w in ("return", "reach back", "move back", "retract", "withdraw")):
        return "return"
    return "other"


def extract_concrete_noun(skill: str) -> str | None:
    s = skill.strip()
    # Normalize compound verb prefixes like "Grasp and pull a cable", "Pick up and turn over a small jar", "Lift up and inspect the bottle"
    s = re.sub(
        r"^(?:grasp|pick up|lift|retrieve|take|hold|pull out)\s+(?:up\s+)?(?:and\s+(?:lift|inspect|check|hold|carry|pull|turn(?: over)?|move|pack|shift|examine)\s+)?",
        "grasp ",
        s,
        flags=re.I,
    )
    m = re.search(
        r"^grasp\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9\-\s]{3,35}?)(?:\s+(?:from|in|into|onto|to|out|on|and|with|at|near|by|under|for|off|up)\b|$|\.)",
        s,
        re.I,
    )
    if not m:
        return None
    cand = m.group(1).strip().lower()
    for prep in (
        " at the ", " in the ", " from the ", " to the ", " on the ", " near the ",
        " beside the ", " off the ", " for a ", " for ", " up for "
    ):
        if prep in cand:
            cand = cand.split(prep)[0].strip()
    cand = re.sub(r"^(?:the|a|an|up|over)\s+", "", cand).strip()
    cand = re.sub(r"\s+(?:up|off|aside|away)$", "", cand).strip()
    if cand in STOP_NOUNS or len(cand) < 3:
        return None
    if cand.startswith(("and ", "or ", "to ", "for ", "with ")):
        return None
    if any(cand.endswith(" " + w) or cand == w or (" " + w + " ") in cand for w in (
        "cloth", "cover", "covering", "obstruction", "towel", "bin", "box", "opening",
        "object", "objects", "item", "items", "material", "fabric", "edge", "clutter",
        "sheet", "form", "checklist", "backpack"
    )):
        return None
    return cand


def verify_images(images: list[dict]) -> list[dict]:
    out = []
    for im in images:
        p = Path(im["path"])
        assert p.is_file(), f"Missing frame: {p}"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if "sha256" in im:
            assert sha == im["sha256"], f"SHA mismatch for {p}"
        out.append({
            "frame_index": im.get("frame_index", 0),
            "pts_s": float(im["pts_s"]),
            "path": str(p),
            "sha256": sha,
        })
    assert len(out) == 4, f"Expected 4 causal frames, got {len(out)}"
    return out


def make_current_only_images(images: list[dict]) -> list[dict]:
    last = dict(images[-1])
    return [dict(last) for _ in range(4)]


def make_masked_history_images(pre_images: list[dict], post_images: list[dict]) -> list[dict]:
    # Frames 1..3 come from pre-reveal history (before target is exposed), Frame 4 is byte-identical post-reveal current frame
    return [
        dict(pre_images[1]),
        dict(pre_images[2]),
        dict(pre_images[3]),
        dict(post_images[3]),
    ]


OS_SEARCH_TEMPLATES = (
    "Scan the counter to locate the {noun}",
    "Inspect the area to search for the {noun}",
    "Look around the room for the {noun}",
    "Check the surface to find the {noun}",
    "Search the room for the {noun}",
)
RT_SEARCH_TEMPLATES = (
    "Move obstructions aside to search for the {noun}",
    "Shift the covering aside to locate the {noun}",
    "Inspect the cluttered bin for the {noun}",
    "Probe under the clutter to find the {noun}",
)


def noun_core_tokens(noun: str) -> set[str]:
    return {
        w for w in re.split(r"[\s\-]+", noun.lower().strip())
        if len(w) >= 3 and w not in {"the", "and", "small", "large", "item", "object", "colored", "flat"}
    }


def nouns_disjoint(a: str, b: str) -> bool:
    ta, tb = noun_core_tokens(a), noun_core_tokens(b)
    if not ta or not tb:
        return a.lower().strip() != b.lower().strip()
    return len(ta & tb) == 0


def load_os_participant_map_and_vocab() -> tuple[dict[str, str], dict[str, str]]:
    os_map: dict[str, str] = {}
    ep_vocab: dict[str, list[str]] = defaultdict(list)
    for af in sorted((FULL_SOL / "role_a_results").glob("*.json")):
        d = json.loads(af.read_text())
        ep = d["episode_id"]
        if d.get("dataset") == "os_ost_part3":
            os_map[ep] = d.get("participant_id", "os:unknown")
        for seg in d.get("segments", []):
            ep_vocab[ep].append(str(seg.get("executed_skill_en", "")).lower())
            ep_vocab[ep].append(str(seg.get("observed_outcome_en", "")).lower())
    return os_map, {ep: " ".join(txts) for ep, txts in ep_vocab.items()}


def main() -> None:
    os_part_map, ep_vocab_map = load_os_participant_map_and_vocab()

    # Load all causal-checked Role B receipts and deduplicate per episode (2s minimum spacing, <=128 per video)
    raw_by_ep: dict[str, list[dict]] = {}
    for file in sorted((FULL_SOL / "role_b_cases").glob("*/role_b_results.jsonl")):
        cid = file.parent.name
        if cid in EXCLUDED_CASES:
            continue
        label = json.loads(file.read_text())
        if label["results"]["a"]["applicability"] != "applicable":
            continue
        binding = json.loads((file.parent / "binding.json").read_text())
        packet = json.loads((file.parent / "packet.json").read_text())
        ep = binding["episode_id"]
        raw_split = binding["split"]
        if raw_split == "unmapped_os_sequence":
            source_dataset = "os_ost_part3"
            participant = os_part_map.get(ep, binding.get("participant_id", "os:unknown"))
            if participant in OS_TRAIN_PARTICIPANTS:
                split = "train"
            elif participant in OS_VAL_PARTICIPANTS:
                split = "val"
            elif participant in OS_TEST_PARTICIPANTS:
                split = "test"
            else:
                # Any newly checked OS sequence defaults to train only if sequence number >= 8
                m_num = re.search(r"(\d+)$", participant)
                seq_num = int(m_num.group(1)) if m_num else 0
                split = "train" if seq_num >= 8 else "test"
        else:
            source_dataset = "roboturk"
            participant = binding["participant_id"]
            split = "train" if raw_split == "train" else "test"

        target_en = packet["candidates"]["a"].strip()
        alt_en = packet["candidates"]["b"].strip()
        b_app = label["results"]["b"]["applicability"]
        cls = classify_skill(target_en)
        if target_en.lower().startswith("transfer"):
            cls = "carry"
        if target_en.lower().startswith("reach back"):
            cls = "return"

        raw_by_ep.setdefault(ep, []).append({
            "id": cid,
            "episode_id": ep,
            "source_dataset": source_dataset,
            "split": split,
            "source_video_sha256": binding["source_video_sha256"],
            "participant_id": participant,
            "cutoff_pts_s": float(packet["cutoff_pts_s"]),
            "images": verify_images(packet["images"]),
            "generic_goal": packet["goal"].strip(),
            "target": target_en,
            "alt_candidate": alt_en,
            "b_applicability": b_app,
            "skill_class": cls,
            "teacher_message_id": label["response_message_id"],
        })

    dedup_rows: list[dict] = []
    evidence_pair_rows: list[dict] = []
    activity_align_rows: list[dict] = []

    for ep, items in sorted(raw_by_ep.items()):
        items.sort(key=lambda x: x["cutoff_pts_s"])
        kept: list[dict] = []
        last_t = -10.0
        for item in items:
            if item["cutoff_pts_s"] - last_t < 2.0 or len(kept) >= 128:
                continue
            last_t = item["cutoff_pts_s"]
            kept.append(item)

        # Identify sequential (search/reveal -> grasp/acquire concrete noun) pairs within the same recording
        paired_indices: set[int] = set()
        for i in range(len(kept) - 1):
            r_pre = kept[i]
            r_post = kept[i + 1]
            if r_pre["skill_class"] != "search" or r_post["skill_class"] != "acquire":
                continue
            if r_post["cutoff_pts_s"] - r_pre["cutoff_pts_s"] > 25.0:
                continue
            noun = extract_concrete_noun(r_post["target"])
            if not noun:
                continue
            paired_indices.add(i)
            paired_indices.add(i + 1)
            pair_id = f"stage1_pair_{r_pre['id']}__{r_post['id']}"
            if r_pre["source_dataset"] == "os_ost_part3":
                spec_goal = f"Search the room to locate and pick up the {noun} and bring it to the packing location."
            else:
                spec_goal = f"Locate and retrieve the {noun} from the cluttered bin and move it to the sorting box."

            # 1) Pre-reveal under specific goal -> search/inspect
            evidence_pair_rows.append({
                "id": f"{r_pre['id']}__spec_pre",
                "base_case_id": r_pre["id"],
                "pair_id": pair_id,
                "split": r_pre["split"],
                "source_dataset": r_pre["source_dataset"],
                "family": "stage1_specific_pair",
                "task_type": "next_task_selection",
                "goal_mode": "specific_target",
                "history_view": "full",
                "phase": "pre_reveal",
                "target_noun": noun,
                "source_video_sha256": r_pre["source_video_sha256"],
                "participant_id": r_pre["participant_id"],
                "cutoff_pts_s": r_pre["cutoff_pts_s"],
                "images": r_pre["images"],
                "goal": spec_goal,
                "target": r_pre["target"],
                "skill_class": "search",
                "expected_action_class": "search",
                "b_applicability": "inapplicable",
                "multi_solution": False,
            })
            # 2) Post-reveal under specific goal -> acquire specific target
            evidence_pair_rows.append({
                "id": f"{r_post['id']}__spec_post_full",
                "base_case_id": r_post["id"],
                "pair_id": pair_id,
                "split": r_post["split"],
                "source_dataset": r_post["source_dataset"],
                "family": "stage1_specific_pair",
                "task_type": "next_task_selection",
                "goal_mode": "specific_target",
                "history_view": "full",
                "phase": "post_reveal",
                "target_noun": noun,
                "source_video_sha256": r_post["source_video_sha256"],
                "participant_id": r_post["participant_id"],
                "cutoff_pts_s": r_post["cutoff_pts_s"],
                "images": r_post["images"],
                "goal": spec_goal,
                "target": r_post["target"],
                "skill_class": "acquire",
                "expected_action_class": "acquire",
                "b_applicability": "inapplicable",
                "multi_solution": False,
            })
            # 3) Counterfactual masked history (frames 1..3 from pre-reveal, frame 4 from post-reveal) -> search/inspect
            evidence_pair_rows.append({
                "id": f"{r_post['id']}__spec_post_masked",
                "base_case_id": r_post["id"],
                "pair_id": pair_id,
                "split": r_post["split"],
                "source_dataset": r_post["source_dataset"],
                "family": "stage1_specific_pair",
                "task_type": "next_task_selection",
                "goal_mode": "specific_target",
                "history_view": "masked",
                "phase": "post_reveal_masked",
                "target_noun": noun,
                "source_video_sha256": r_post["source_video_sha256"],
                "participant_id": r_post["participant_id"],
                "cutoff_pts_s": r_post["cutoff_pts_s"],
                "images": make_masked_history_images(r_pre["images"], r_post["images"]),
                "goal": spec_goal,
                "target": r_pre["target"],
                "skill_class": "search",
                "expected_action_class": "search",
                "b_applicability": "inapplicable",
                "multi_solution": False,
            })

        # Add all deduplicated base rows (upgrading single-solution rows with concrete nouns to specific goals)
        for idx, item in enumerate(kept):
            noun = extract_concrete_noun(item["target"])
            use_specific = bool(noun and (item["b_applicability"] == "inapplicable" or idx in paired_indices))
            if use_specific:
                if item["source_dataset"] == "os_ost_part3":
                    goal_text = f"Search the room to locate and pick up the {noun} and bring it to the packing location."
                else:
                    goal_text = f"Locate and retrieve the {noun} from the cluttered bin and move it to the sorting box."
                goal_mode = "specific_target"
            else:
                goal_text = item["generic_goal"]
                goal_mode = "generic_goal"

            row = {
                "id": item["id"],
                "base_case_id": item["id"],
                "pair_id": None,
                "split": item["split"],
                "source_dataset": item["source_dataset"],
                "family": "stage1_causal_video",
                "task_type": "next_task_selection",
                "goal_mode": goal_mode,
                "history_view": "full",
                "phase": "single_prefix",
                "target_noun": noun,
                "source_video_sha256": item["source_video_sha256"],
                "participant_id": item["participant_id"],
                "cutoff_pts_s": item["cutoff_pts_s"],
                "images": item["images"],
                "goal": goal_text,
                "target": item["target"],
                "skill_class": item["skill_class"],
                "expected_action_class": item["skill_class"],
                "alt_candidate": item["alt_candidate"],
                "b_applicability": item["b_applicability"],
                "multi_solution": item["b_applicability"] == "applicable",
            }
            dedup_rows.append(row)

            # 20% activity_description slice on train rows for visual-language semantic alignment (per VIDEO_TRAINING_RECIPE.md)
            if item["split"] == "train" and (idx % 5 == 0):
                activity_align_rows.append({
                    "id": f"{item['id']}__activity_align",
                    "base_case_id": item["id"],
                    "pair_id": None,
                    "split": "train",
                    "source_dataset": item["source_dataset"],
                    "family": "stage1_activity_alignment",
                    "task_type": "activity_description",
                    "goal_mode": "activity_alignment",
                    "history_view": "full",
                    "phase": "activity_clip",
                    "target_noun": noun,
                    "source_video_sha256": item["source_video_sha256"],
                    "participant_id": item["participant_id"],
                    "cutoff_pts_s": item["cutoff_pts_s"],
                    "images": item["images"],
                    "goal": "Describe the manipulation or search action currently being performed in this video clip.",
                    "target": item["target"],
                    "skill_class": item["skill_class"],
                    "expected_action_class": item["skill_class"],
                    "alt_candidate": item["alt_candidate"],
                "b_applicability": item["b_applicability"],
                    "multi_solution": item["b_applicability"] == "applicable",
                })

    # Load curated specific-target RoboTurk occlusion-reveal pairs from evidence_suite.jsonl (ZERO ActiveArena rows!)
    ev_path = ROOT / "results/evidence_confidence_action_v1/e49634f5-72f2-41cf-9baf-29a0a1fcf9fe/evidence_suite.jsonl"
    curated_rt_rows: list[dict] = []
    curated_val_videos: set[str] = set()
    curated_test_videos: set[str] = set()
    curated_val_parts: set[str] = set()
    curated_test_parts: set[str] = set()
    if ev_path.is_file():
        for line in ev_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("domain") == "roboturk":
                sp = r["split"]
                if sp == "val":
                    curated_val_videos.add(r["source_video_sha256"])
                    curated_val_parts.add(r["participant_id"])
                elif sp == "test":
                    curated_test_videos.add(r["source_video_sha256"])
                    curated_test_parts.add(r["participant_id"])
                curated_rt_rows.append({
                    "id": f"curated_{r['id']}",
                    "base_case_id": r["id"],
                    "pair_id": r["pair_id"],
                    "split": sp,
                    "source_dataset": "roboturk",
                    "family": "curated_roboturk_specific_pair",
                    "task_type": "next_task_selection",
                    "goal_mode": "specific_target",
                    "history_view": r.get("history_view", "full"),
                    "phase": r.get("phase", "single_prefix"),
                    "target_noun": r.get("target_attribute"),
                    "source_video_sha256": r["source_video_sha256"],
                    "participant_id": r["participant_id"],
                    "cutoff_pts_s": float(r["cutoff_pts_s"]),
                    "images": verify_images(r["images"]),
                    "goal": r["goal"],
                    "target": r["target"],
                    "skill_class": r["skill_class"],
                    "expected_action_class": r.get("expected_action_class", r["skill_class"]),
                    "b_applicability": "inapplicable",
                    "multi_solution": False,
                })

    # Protect the 4 brand-new additional holdout OS recordings strictly as test (never train or val!)
    add_holdout_path = ROOT / "results/stage1_closeout_20261121/additional_holdout.jsonl"
    if add_holdout_path.is_file():
        for line in add_holdout_path.read_text().splitlines():
            ar = json.loads(line)
            if ar.get("source_video_sha256"):
                curated_test_videos.add(ar["source_video_sha256"])
            if ar.get("participant_id"):
                curated_test_parts.add(ar["participant_id"])

    # Re-assign any base/pair/activity row from curated val/test participants to match val/test
    filtered_rows: list[dict] = []
    for r in dedup_rows + evidence_pair_rows + activity_align_rows:
        if r["participant_id"] in curated_test_parts or r["source_video_sha256"] in curated_test_videos:
            if r["task_type"] == "activity_description":
                continue
            r["split"] = "test"
        elif r["participant_id"] in curated_val_parts or r["source_video_sha256"] in curated_val_videos:
            if r["task_type"] == "activity_description":
                continue
            r["split"] = "val"
        filtered_rows.append(r)

    base_rows = filtered_rows + curated_rt_rows

    # =========================================================================
    # Stage-1 v2 Strictly-Train-Only Augmentations:
    # 1) Anti-false-negative diverse G_mismatch negative-target contrastive pairs
    # 2) Slot-randomized post_reveal positive pairs (Slot-1 & Slot-2 with p3->f4 jump-cut)
    # =========================================================================
    train_base_rows = [r for r in base_rows if r["split"] == "train"]
    train_acq_concrete = [
        r for r in train_base_rows
        if r["task_type"] == "next_task_selection"
        and r["skill_class"] == "acquire"
        and r.get("target_noun")
        and len(str(r["target_noun"]).strip()) >= 3
    ]
    all_train_nouns = sorted({str(r["target_noun"]).strip() for r in train_acq_concrete})
    rng = random.Random(2027)

    gmismatch_rows: list[dict] = []
    for r in train_acq_concrete:
        noun_a = str(r["target_noun"]).strip()
        ep_id = str(r["base_case_id"]).split("-")[0]
        ep_text = ep_vocab_map.get(ep_id, "")
        cands = []
        for n in all_train_nouns:
            if not nouns_disjoint(noun_a, n):
                continue
            # Anti-false-negative check: ensure none of candidate noun's core tokens appear anywhere in this recording's Teacher A descriptions!
            core = noun_core_tokens(n)
            if ep_text and core and any(re.search(rf"\b{re.escape(tok)}\b", ep_text) for tok in core):
                continue
            cands.append(n)
        if not cands:
            continue
        k_samples = min(3, len(cands))
        for k, noun_b in enumerate(rng.sample(cands, k_samples)):
            if r["source_dataset"] == "os_ost_part3":
                goal_b = f"Search the room to locate and pick up the {noun_b} and bring it to the packing location."
                tgt_b = rng.choice(OS_SEARCH_TEMPLATES).format(noun=noun_b)
            else:
                goal_b = f"Locate and retrieve the {noun_b} from the cluttered bin and move it to the sorting box."
                tgt_b = rng.choice(RT_SEARCH_TEMPLATES).format(noun=noun_b)
            gmismatch_rows.append({
                "id": f"{r['id']}__gmismatch_{k}",
                "base_case_id": r["base_case_id"],
                "pair_id": None,
                "split": "train",
                "source_dataset": r["source_dataset"],
                "family": "stage1_v2_gmismatch",
                "task_type": "next_task_selection",
                "goal_mode": "specific_target_mismatch",
                "history_view": "full",
                "phase": "gmismatch_negative",
                "target_noun": noun_b,
                "source_video_sha256": r["source_video_sha256"],
                "participant_id": r["participant_id"],
                "cutoff_pts_s": r["cutoff_pts_s"],
                "images": r["images"],
                "goal": goal_b,
                "target": tgt_b,
                "skill_class": "search",
                "expected_action_class": "search",
                "b_applicability": "inapplicable",
                "multi_solution": False,
            })

    by_train_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in train_base_rows:
        if r.get("pair_id") and r.get("phase") in ("pre_reveal", "post_reveal"):
            by_train_pair[r["pair_id"]][r["phase"]] = r
    slot_pos_rows: list[dict] = []
    for pid, grp in sorted(by_train_pair.items()):
        if not {"pre_reveal", "post_reveal"} <= grp.keys():
            continue
        pre, full = grp["pre_reveal"], grp["post_reveal"]
        f, p = full["images"], pre["images"]
        for tag, imgs in (
            ("slot2_jump_pos", [p[0], f[2], p[2], f[3]]),
            ("slot1_jump_pos", [f[2], p[1], p[2], f[3]]),
        ):
            slot_pos_rows.append(dict(
                full,
                id=f"{full['id']}__{tag}",
                family="stage1_v2_slot_pos",
                history_view=tag,
                phase=tag,
                images=imgs,
            ))

    all_rows = base_rows + gmismatch_rows + slot_pos_rows

    # Verify strict participant & video disjointness between train and (val, test), and zero ActiveArena contamination
    assert all(r["source_dataset"] in ("roboturk", "os_ost_part3") for r in all_rows), "Found non-Stage-1 dataset row!"
    train_videos = {r["source_video_sha256"] for r in all_rows if r["split"] == "train"}
    val_videos = {r["source_video_sha256"] for r in all_rows if r["split"] == "val"}
    test_videos = {r["source_video_sha256"] for r in all_rows if r["split"] == "test"}
    assert not (train_videos & val_videos), "Train/Val video overlap!"
    assert not (train_videos & test_videos), "Train/Test video overlap!"
    assert not (val_videos & test_videos), "Val/Test video overlap!"

    train_parts = {r["participant_id"] for r in all_rows if r["split"] == "train"}
    val_parts = {r["participant_id"] for r in all_rows if r["split"] == "val"}
    test_parts = {r["participant_id"] for r in all_rows if r["split"] == "test"}
    assert not (train_parts & val_parts), f"Train/Val participant overlap: {train_parts & val_parts}"
    assert not (train_parts & test_parts), f"Train/Test participant overlap: {train_parts & test_parts}"

    out_jsonl = HERE / "stage1_formal_suite.jsonl"
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))

    summary = {
        "suite_version": "stage1_v2_rebalanced",
        "total_rows": len(all_rows),
        "activearena_rows": 0,
        "language": "en",
        "by_split": dict(Counter(r["split"] for r in all_rows)),
        "train_by_source": dict(Counter(r["source_dataset"] for r in all_rows if r["split"] == "train")),
        "train_by_task_type": dict(Counter(r["task_type"] for r in all_rows if r["split"] == "train")),
        "train_by_goal_mode": dict(Counter(r["goal_mode"] for r in all_rows if r["split"] == "train")),
        "train_by_family": dict(Counter(r["family"] for r in all_rows if r["split"] == "train")),
        "train_by_skill_class": dict(Counter(r["skill_class"] for r in all_rows if r["split"] == "train")),
        "val_by_source": dict(Counter(r["source_dataset"] for r in all_rows if r["split"] == "val")),
        "test_by_source": dict(Counter(r["source_dataset"] for r in all_rows if r["split"] == "test")),
        "evidence_pair_rows_by_split": dict(Counter(r["split"] for r in evidence_pair_rows + curated_rt_rows)),
        "train_gmismatch_rows": len(gmismatch_rows),
        "train_slot_pos_rows": len(slot_pos_rows),
        "train_recordings": len(train_videos),
        "val_recordings": len(val_videos),
        "test_recordings": len(test_videos),
        "train_participants": len(train_parts),
        "val_participants": len(val_parts),
        "test_participants": len(test_parts),
    }
    (HERE / "stage1_formal_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
