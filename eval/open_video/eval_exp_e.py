#!/usr/bin/env python3
"""Experiment E: Surgical Fixes for the Two Verified Bottlenecks + Confidence Calibration.

Sub-Experiment E1 (ActiveArena Slot-Randomized Revealing-Frame Adaptation, 336x252, LoRA r=8):
  - Trains on the exact same train split of `e49634f5/evidence_suite.jsonl` (0 overlap with the 6 test episodes),
    with randomized revealing-frame slot placement (Slot 1, 2, or 3) + smooth zero-jump-cut no-evidence `[f4]*4`.
  - Evaluates all 6 held-out ActiveArena episodes across 6 conditions:
    `full`, `jump_keep_slot2`, `jump_keep_slot1`, `smooth_no_evidence`, `masked`, `color_swap`.

Sub-Experiment E2 (Open-Video Negative-Target Contrastive `G_mismatch` Adaptation, 448x336, LoRA r=32):
  - Continues from `725c7958` best validation checkpoint (Epoch 4, 0 ActiveArena rows).
  - Constructs strictly train-split-only `G_mismatch` contrastive pairs (smooth close-up of object A paired with
    an absent object B from another training recording -> `Search ... for the <B>`), plus slot-randomized positives.
  - Runs 120 low-LR steps (BS=16 = micro_bs 4 x grad_accum 4, Cosine LR 4e-5 -> 4e-6).
  - Evaluates on:
    (a) Canonical 144-row test suite (OS /45, RoboTurk /45, 16 complete triplets)
    (b) 4 new held-out OS videos (9 triplets + 2 cross-goal scenes `goal_cross`)
    (c) 25-triplet 2x2 Factorial Control (Jump-Cut vs Evidence Removal)
    (d) Cross-validated Ridge + Temperature-Scaled Confidence Head (Overall + per-domain Brier & AUROC).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time

ROOT = Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/reports/rss2027_autoresearch")
OUT_DIR = ROOT / "results/exp_e_slot_and_gmismatch_v1"
AA_RUN = ROOT / "results/evidence_confidence_action_v1/e49634f5-72f2-41cf-9baf-29a0a1fcf9fe"
OPEN_RUN = ROOT / "results/stage1_full_6ep_r32_highres448_v1/725c7958-3dbf-4e3d-9b14-70ce5c9c02f3"
CLOSEOUT = ROOT / "results/stage1_closeout_20261121"
MODEL_PATH = "/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/Qwen3-VL-2B-Instruct"
PILOT = Path("/home/ubuntu/ygwang/IP-VLA/rss2027-local-pilot/research/rss2027_local")
sys.path.insert(0, str(PILOT))
import run as pilot_run

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
COLORS = ("green", "red", "blue", "purple", "orange", "yellow")


def detect_color(text: str) -> str | None:
    tl = text.lower()
    found = [c for c in COLORS if re.search(rf"\b{c}\b", tl)]
    return found[0] if found else None


def classify_aa(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ("place", "put", "set", "drop", "move the gray block to", "move the block to")):
        return "place"
    if any(w in tl for w in ("rotate", "turn", "flip", "inspect", "check", "look", "view", "reveal", "examine")):
        return "inspect"
    return pilot_run.classify(text)


def nouns_disjoint(a: str, b: str) -> bool:
    ta = {w for w in re.split(r"[\s\-]+", a.lower()) if len(w) >= 3 and w not in {"the", "and", "small", "large", "item", "object"}}
    tb = {w for w in re.split(r"[\s\-]+", b.lower()) if len(w) >= 3 and w not in {"the", "and", "small", "large", "item", "object"}}
    if not ta or not tb:
        return a.lower().strip() != b.lower().strip()
    return len(ta & tb) == 0


def build_e1_data() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in (AA_RUN / "evidence_suite.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if r["split"] == "train"]
    test_aa_post = [r for r in rows if r["domain"] == "activearena" and r["split"] == "test" and r["evidence_present"]]
    assert len(test_aa_post) == 6

    # Build slot-randomized ActiveArena training pool (strictly from train_rows!)
    rng = random.Random(17)
    e1_train = []
    for r in train_rows:
        if r["domain"] == "activearena":
            if r["evidence_present"]:
                f = r["images"]
                p = r["masked_images"]
                # 1. Original Slot-3 reveal
                e1_train.append({"id": r["id"] + "__s3", "goal": r["goal"], "target": r["target"], "images": [f[0], f[1], f[2], f[3]]})
                # 2. Slot-2 reveal with p3->f4 jump-cut
                e1_train.append({"id": r["id"] + "__s2", "goal": r["goal"], "target": r["target"], "images": [p[0], f[2], p[2], f[3]]})
                # 3. Slot-1 reveal with p3->f4 jump-cut
                e1_train.append({"id": r["id"] + "__s1", "goal": r["goal"], "target": r["target"], "images": [f[2], p[1], p[2], f[3]]})
                # 4. Masked no-evidence -> inspect
                e1_train.append({"id": r["id"] + "__mask", "goal": r["goal"], "target": r["masked_target"], "images": [p[0], p[1], p[2], f[3]]})
                # 5. Smooth zero-jump-cut no-evidence `[f4]*4` -> inspect
                e1_train.append({"id": r["id"] + "__smooth", "goal": r["goal"], "target": r["masked_target"], "images": [f[3], f[3], f[3], f[3]]})
            else:
                e1_train.append({"id": r["id"], "goal": r["goal"], "target": r["target"], "images": r["images"]})
        else:
            # Keep RoboTurk train rows from e49634f5
            e1_train.append({"id": r["id"], "goal": r["goal"], "target": r["target"], "images": r["images"]})
            if r.get("evidence_present") and r.get("masked_images") and r.get("masked_target"):
                e1_train.append({"id": r["id"] + "__mask", "goal": r["goal"], "target": r["masked_target"], "images": r["masked_images"]})

    # 6 held-out ActiveArena test episodes x 6 conditions
    e1_probes = []
    for idx, r in enumerate(test_aa_post):
        other = test_aa_post[(idx + 1) % len(test_aa_post)]
        f = r["images"]
        p = r["masked_images"]
        tc = r["hidden_color"]
        sc = other["hidden_color"]
        for cond, imgs, exp_cls, exp_color in (
            ("full",               [f[0], f[1], f[2], f[3]],               "place",   tc),
            ("jump_keep_slot2",    [p[0], f[2], p[2], f[3]],               "place",   tc),
            ("jump_keep_slot1",    [f[2], p[1], p[2], f[3]],               "place",   tc),
            ("smooth_no_evidence", [f[3], f[3], f[3], f[3]],               "inspect", None),
            ("masked",             [p[0], p[1], p[2], f[3]],               "inspect", None),
            ("color_swap",         [f[0], f[1], other["images"][2], f[3]], "place",   sc),
        ):
            e1_probes.append({
                "id": f"{r['id']}__{cond}",
                "episode_id": r["id"],
                "condition": cond,
                "goal": r["goal"],
                "true_color": tc,
                "expected_class": exp_cls,
                "expected_color": exp_color,
                "images": imgs,
            })
    return e1_train, e1_probes


def build_e2_data():
    suite_rows = [json.loads(l) for l in (OPEN_RUN / "stage1_formal_suite.jsonl").read_text().splitlines()]
    train_rows = [r for r in suite_rows if r["split"] == "train"]
    val_rows = [r for r in suite_rows if r["split"] == "val"]
    row_by_id = {r["id"]: r for r in suite_rows}

    # Strictly train-only concrete target nouns pool
    train_acq_concrete = [
        r for r in train_rows
        if r["task_type"] == "next_task_selection"
        and r["skill_class"] == "acquire"
        and r.get("target_noun")
        and len(r["target_noun"].strip()) >= 3
    ]
    all_train_nouns = sorted({r["target_noun"].strip() for r in train_acq_concrete})
    rng = random.Random(2027)

    # Construct strictly train-split-only G_mismatch negative-target contrastive pairs
    # (same smooth close-up frames of object A, asking for absent object B -> search!)
    mismatch_rows = []
    for r in train_acq_concrete:
        noun_a = r["target_noun"].strip()
        cands = [n for n in all_train_nouns if nouns_disjoint(noun_a, n)]
        if not cands:
            continue
        for k in range(2):
            noun_b = rng.choice(cands)
            if r["source_dataset"] == "os_ost_part3":
                goal_b = f"Search the room to locate and pick up the {noun_b} and bring it to the packing location."
                tgt_b = f"Scan the room to search for the {noun_b}"
            else:
                goal_b = f"Locate and retrieve the {noun_b} from the cluttered bin and move it to the sorting box."
                tgt_b = f"Move obstructions aside to search for the {noun_b}"
            mismatch_rows.append({
                "id": f"{r['id']}__gmismatch_{k}",
                "split": "train",
                "source_dataset": r["source_dataset"],
                "task_type": "next_task_selection",
                "goal_mode": "specific_target_mismatch",
                "target_noun": noun_b,
                "goal": goal_b,
                "target": tgt_b,
                "skill_class": "search",
                "expected_action_class": "search",
                "images": r["images"],
            })

    # Also build slot-randomized post_reveal positives from train triplets (robustness to jump-cuts when target IS present)
    by_train_pair = defaultdict(dict)
    for r in train_rows:
        if r.get("pair_id") and r.get("phase") in ("pre_reveal", "post_reveal"):
            by_train_pair[r["pair_id"]][r["phase"]] = r
    slot_pos_rows = []
    for pid, grp in sorted(by_train_pair.items()):
        if not {"pre_reveal", "post_reveal"} <= grp.keys():
            continue
        pre, full = grp["pre_reveal"], grp["post_reveal"]
        f, p = full["images"], pre["images"]
        slot_pos_rows.append(dict(full, id=full["id"] + "__late_jump_pos", images=[p[0], f[2], p[2], f[3]]))

    # Canonical 144-row test pool
    parent_gen = ROOT / "results/stage1_bs16_nogc_rank32_highres448_v1/ea40ef8b-e862-4827-b435-5d0334d17559/lora_r32_448x336_step378_test_generations.json"
    canon_ids = [g["row_id"] for g in json.loads(parent_gen.read_text())]
    test_eval_rows = [row_by_id[rid] for rid in canon_ids if rid in row_by_id]

    # Validation calibration pool (exact same selection as 725c7958)
    val_pair_rows = [r for r in val_rows if r.get("pair_id")]
    val_single_sub = pilot_run.select_stratified_subset(
        [r for r in val_rows if not r.get("pair_id")], 32, 17
    )
    val_calib_rows = val_pair_rows + val_single_sub

    # Additional holdout (9 new-video triplets + 2 cross-goal scenes = 31 prefixes)
    add_rows = [json.loads(l) for l in (CLOSEOUT / "additional_holdout.jsonl").read_text().splitlines()]

    # Visibility audit map for 2x2 open-video control
    vis_map = {}
    with (CLOSEOUT / "current_frame_audit.csv").open() as f_csv:
        for row in csv.DictReader(f_csv):
            vis_map[row["row_id"]] = row["visibility"]

    open_2x2_probes = []
    for pool_rows, pool_tag in ((test_eval_rows, "canonical_16"), ([r for r in add_rows if r.get("probe_type") == "new_video_triplet"], "additional_9")):
        by_p = defaultdict(dict)
        for r in pool_rows:
            if r.get("pair_id") and r.get("phase") in ("pre_reveal", "post_reveal", "post_reveal_masked"):
                by_p[r["pair_id"]][r["phase"]] = r
        for pid, grp in sorted(by_p.items()):
            if not {"pre_reveal", "post_reveal", "post_reveal_masked"} <= grp.keys():
                continue
            pre, full = grp["pre_reveal"], grp["post_reveal"]
            f, p = full["images"], pre["images"]
            vis = vis_map.get(full["id"], "unknown")
            for cond, imgs, exp_cls in (
                ("full",               [f[0], f[1], f[2], f[3]], "acquire"),
                ("early_jump_keep",    [p[0], f[1], f[2], f[3]], "acquire"),
                ("late_jump_keep",     [p[0], f[2], p[2], f[3]], "acquire"),
                ("smooth_no_evidence", [f[3], f[3], f[3], f[3]], "search"),
                ("masked",             [p[1], p[2], p[3], f[3]], "search"),
            ):
                open_2x2_probes.append({
                    "id": f"{full['id']}__{cond}",
                    "pair_id": pid,
                    "source_pool": pool_tag,
                    "visibility": vis,
                    "condition": cond,
                    "goal": full["goal"],
                    "goal_mode": "specific_target",
                    "target_noun": full.get("target_noun"),
                    "expected_action_class": exp_cls,
                    "images": imgs,
                })

    return train_rows, mismatch_rows, slot_pos_rows, val_calib_rows, test_eval_rows, add_rows, open_2x2_probes


def fit_ridge_temp_confidence(X_cal, y_cal, X_test, y_test_list, test_domains: list[str]):
    """Fit L2-regularized Ridge Logistic Regression + 5-fold CV Temperature Scaling for calibrated Brier + AUROC."""
    import torch
    from torch import nn

    h_mean = X_cal.mean(dim=0, keepdim=True)
    h_std = X_cal.std(dim=0, keepdim=True).clamp_min(1e-5)
    Xc = (X_cal - h_mean) / h_std
    Xt = (X_test - h_mean) / h_std

    # 1) Standardize & L2-normalize feature rows to prevent 2048-dim norm explosion on small N
    Xc_unit = nn.functional.normalize(Xc, dim=-1)
    Xt_unit = nn.functional.normalize(Xt, dim=-1)

    # 5-fold CV over L2 penalty `lam` on validation set to select optimal shrinkage (minimizing CV Brier)
    n = Xc_unit.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    best_lam = 10.0
    best_cv_brier = 1e9
    best_oof_logits = None

    for lam in (0.5, 2.0, 5.0, 15.0, 40.0, 100.0):
        oof_logits = torch.zeros(n)
        for fold in range(5):
            val_mask = torch.zeros(n, dtype=torch.bool)
            val_mask[perm[fold::5]] = True
            tr_mask = ~val_mask
            head = nn.Linear(Xc_unit.shape[1], 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            opt = torch.optim.LBFGS(head.parameters(), lr=0.5, max_iter=40)

            def closure():
                opt.zero_grad()
                logits = head(Xc_unit[tr_mask]).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_cal[tr_mask])
                loss = loss + 0.5 * lam * (head.weight.pow(2).sum()) / tr_mask.sum()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                oof_logits[val_mask] = head(Xc_unit[val_mask]).squeeze(-1)
        probs = torch.sigmoid(oof_logits)
        cv_brier = float(((probs - y_cal) ** 2).mean().item())
        if cv_brier < best_cv_brier:
            best_cv_brier = cv_brier
            best_lam = lam
            best_oof_logits = oof_logits

    # Fit 1-parameter Platt/Temperature scaling on out-of-fold validation logits
    temp_layer = nn.Linear(1, 1)
    nn.init.ones_(temp_layer.weight)
    nn.init.zeros_(temp_layer.bias)
    t_opt = torch.optim.LBFGS(temp_layer.parameters(), lr=0.5, max_iter=50)

    def t_closure():
        t_opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            temp_layer(best_oof_logits.unsqueeze(-1)).squeeze(-1), y_cal
        )
        loss.backward()
        return loss

    t_opt.step(t_closure)

    # Fit final Ridge head on all validation rows with best_lam
    final_head = nn.Linear(Xc_unit.shape[1], 1)
    nn.init.zeros_(final_head.weight)
    nn.init.zeros_(final_head.bias)
    f_opt = torch.optim.LBFGS(final_head.parameters(), lr=0.5, max_iter=50)

    def f_closure():
        f_opt.zero_grad()
        logits = final_head(Xc_unit).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_cal)
        loss = loss + 0.5 * best_lam * (final_head.weight.pow(2).sum()) / n
        loss.backward()
        return loss

    f_opt.step(f_closure)

    with torch.no_grad():
        raw_test_logits = final_head(Xt_unit).squeeze(-1)
        cal_test_probs = torch.sigmoid(temp_layer(raw_test_logits.unsqueeze(-1)).squeeze(-1)).tolist()
        const_p = float(y_cal.mean().item())
        const_probs = [const_p for _ in y_test_list]

    by_domain = {}
    for dom in ("all", "os_ost_part3", "roboturk", "triplet"):
        idxs = [i for i, d in enumerate(test_domains) if dom == "all" or d == dom]
        if not idxs:
            continue
        yl = [y_test_list[i] for i in idxs]
        pl = [cal_test_probs[i] for i in idxs]
        cl = [const_probs[i] for i in idxs]
        by_domain[dom] = {
            "constant_baseline": pilot_run.compute_confidence_metrics(yl, cl),
            "ridge_temp_hidden_c": pilot_run.compute_confidence_metrics(yl, pl),
        }

    return {
        "best_lam": best_lam,
        "cv_val_brier": round(best_cv_brier, 4),
        "temp_scale_a": round(float(temp_layer.weight.item()), 4),
        "temp_scale_b": round(float(temp_layer.bias.item()), 4),
        "by_domain": by_domain,
    }, cal_test_probs


def main() -> None:
    e1_train, e1_probes = build_e1_data()
    train_rows, mismatch_rows, slot_pos_rows, val_calib_rows, test_eval_rows, add_rows, open_2x2_probes = build_e2_data()

    preflight = {
        "e1_train_rows": len(e1_train),
        "e1_probes": len(e1_probes),
        "e2_base_train_rows": len(train_rows),
        "e2_gmismatch_negatives": len(mismatch_rows),
        "e2_slot_positives": len(slot_pos_rows),
        "e2_val_calib_rows": len(val_calib_rows),
        "e2_test_eval_rows": len(test_eval_rows),
        "e2_add_holdout_rows": len(add_rows),
        "e2_open_2x2_probes": len(open_2x2_probes),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if sys.argv[1:] == ["--cpu-preflight"]:
        for r in e1_probes + add_rows:
            for im in r["images"]:
                assert Path(im["path"]).is_file(), f"Missing frame: {im['path']}"
        print("CPU preflight passed!", flush=True)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {"status": "running", "peak_process_mib": 0, "preflight": preflight}
    stopping = threading.Event()

    def watch():
        while not stopping.wait(0.5):
            try:
                info = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    text=True, timeout=2,
                )
                for line in info.splitlines():
                    pid, mem = (s.strip() for s in line.split(","))
                    if pid == str(os.getpid()):
                        receipt["peak_process_mib"] = max(receipt["peak_process_mib"], int(mem))
            except Exception:
                pass
            if time.monotonic() - started >= 2400 or receipt["peak_process_mib"] > 19968:
                receipt["status"] = "stopped_by_budget"
                (OUT_DIR / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                os._exit(124)

    t = threading.Thread(target=watch, daemon=True)
    t.start()

    try:
        import torch
        from torch import nn
        from PIL import Image
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.manual_seed(17)
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        assert free > 14000 * 2**20, f"Low free VRAM: {free / 2**20:.0f} MiB"
        torch.cuda.set_per_process_memory_fraction(18944 * 2**20 / total)

        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

        class LowRankLinear(nn.Module):
            def __init__(self, original, rank: int, alpha: int):
                super().__init__()
                self.base = original
                self.a = nn.Parameter(torch.empty(rank, original.in_features, device=original.weight.device))
                self.b = nn.Parameter(torch.zeros(original.out_features, rank, device=original.weight.device))
                nn.init.kaiming_uniform_(self.a, a=5**0.5)
                self.scale = alpha / rank

            def forward(self, value):
                output = self.base(value)
                delta = nn.functional.linear(nn.functional.linear(value, self.a.to(value.dtype)), self.b.to(value.dtype))
                return output + delta.to(output.dtype) * self.scale

        def send(inputs: dict) -> dict:
            return {
                k: (v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.cuda())
                for k, v in inputs.items()
            }

        def make_encoder(img_w: int, img_h: int):
            cache: dict[str, tuple[dict, torch.Tensor]] = {}

            def encode(row: dict) -> tuple[dict, torch.Tensor]:
                cid = row["id"]
                if cid in cache:
                    return cache[cid]
                if len(cache) >= 384:
                    del cache[next(iter(cache))]
                imgs = []
                for im in row["images"]:
                    with Image.open(im["path"]) as img:
                        imgs.append(img.convert("RGB").resize((img_w, img_h), Image.Resampling.LANCZOS))
                if row.get("task_type") == "activity_description":
                    prompt_text = (
                        "Task: Describe the manipulation or search action currently being performed "
                        "in this video clip as a short English imperative sentence."
                    )
                else:
                    prompt_text = (
                        f"Goal: {row['goal']}\n"
                        "The views are ordered oldest to newest. "
                        "State one feasible immediate next task as a short imperative sentence."
                    )
                msg = [{"role": "user", "content": [*[{"type": "image"} for _ in imgs], {"type": "text", "text": prompt_text}]}]
                prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                enc = dict(processor(images=imgs, text=[prompt], return_tensors="pt"))
                tgt = processor.tokenizer(row.get("target", "Inspect") + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids
                cache[cid] = (enc, tgt)
                return enc, tgt

            return encode

        def subbatch_loss(model, batch_rows: list[dict], encode_fn) -> torch.Tensor:
            pairs = [encode_fn(r) for r in batch_rows]
            prompts = [enc["input_ids"][0] for enc, _ in pairs]
            answers = [tgt[0] for _, tgt in pairs]
            pw = max(x.numel() for x in prompts)
            aw = max(x.numel() for x in answers)
            ids = torch.full((len(batch_rows), pw + aw), processor.tokenizer.pad_token_id)
            attn = torch.zeros_like(ids)
            labels = torch.full((len(batch_rows), aw), -100)
            for i, (pr, ans) in enumerate(zip(prompts, answers)):
                st = pw - pr.numel()
                en = pw + ans.numel()
                ids[i, st:pw] = pr
                ids[i, pw:en] = ans
                attn[i, st:en] = 1
                labels[i, :ans.numel()] = ans
            inps = {"input_ids": ids, "attention_mask": attn}
            for k in ("pixel_values", "image_grid_thw"):
                inps[k] = torch.cat([enc[k] for enc, _ in pairs], dim=0)
            logits = model(**send(inps), use_cache=False, logits_to_keep=aw + 1).logits[:, :-1]
            labels = labels.cuda()
            tok_l = nn.functional.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")
            return tok_l.sum(dim=1) / labels.ne(-100).sum(dim=1)

        def generate_with_hidden(model, row: dict, encode_fn) -> dict:
            enc, _ = encode_fn(row)
            inps = send(enc)
            plen = inps["input_ids"].shape[1]
            with torch.no_grad():
                gen_ids = model.generate(**inps, max_new_tokens=28, do_sample=False)
                new_ids = gen_ids[0, plen:]
                sentence = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                other = {k: v for k, v in inps.items() if k not in {"input_ids", "attention_mask"}}
                n_new = max(int(new_ids.numel()), 1)
                fw = model(
                    input_ids=gen_ids,
                    attention_mask=torch.ones_like(gen_ids),
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=n_new + 1,
                    **other,
                )
                lps = torch.log_softmax(fw.logits[0, :-1].float(), dim=-1)
                tok_lps = [float(lps[i, int(tid)]) for i, tid in enumerate(new_ids)]
                mean_lp = sum(tok_lps) / max(len(tok_lps), 1)
                last_h = fw.hidden_states[-1][0]
                im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
                h_last = (last_h[-2] if int(new_ids[-1]) == im_end and new_ids.numel() >= 2 else last_h[-1]).float().detach().cpu()
            return {"text": sentence, "mean_token_logprob": round(mean_lp, 4), "_h_last": h_last}

        # =========================================================================
        # SUB-EXPERIMENT E1: ActiveArena Slot-Randomized Adaptation (336x252, r=8)
        # =========================================================================
        print("=== Starting Sub-Experiment E1: ActiveArena Slot-Randomized Adaptation ===", flush=True)
        t_e1 = time.monotonic()
        m1 = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0},
        )
        m1.requires_grad_(False)
        m1.model.language_model.gradient_checkpointing_disable()

        e1_adapters = {}
        for name, mod in list(m1.named_modules()):
            if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent, _, leaf = name.rpartition(".")
                w = LowRankLinear(mod, 8, 16)
                setattr(m1.get_submodule(parent), leaf, w)
                e1_adapters[name] = w

        # Initialize from exact same parent weights as e49634f5
        parent_w = load_file("/home/ubuntu/ygwang/IP-VLA/asa-iteration/reports/rss2027_autoresearch/results/batch_epoch_control/472b11eb-c6c5-4713-8496-e2c17584f8f1/natural.safetensors")
        with torch.no_grad():
            for n, ad in e1_adapters.items():
                ad.a.copy_(parent_w[f"{n}.a"])
                ad.b.copy_(parent_w[f"{n}.b"])

        e1_params = [p for ad in e1_adapters.values() for p in (ad.a, ad.b)]
        e1_opt = torch.optim.AdamW(e1_params, lr=1.5e-4, weight_decay=0.0)
        enc_336 = make_encoder(336, 252)

        rng = random.Random(17)
        e1_steps = 64
        m1.train()
        e1_losses = []
        for step in range(1, e1_steps + 1):
            batch = [rng.choice(e1_train) for _ in range(8)]
            e1_opt.zero_grad(set_to_none=True)
            loss = subbatch_loss(m1, batch, enc_336).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
            e1_opt.step()
            e1_losses.append(float(loss.detach()))
            if step in (1, 32, 64):
                print(f"[E1] step {step}/{e1_steps} loss={e1_losses[-1]:.4f}", flush=True)

        # Save E1 adapter
        save_file({f"{n}.a": ad.a.detach().cpu().contiguous() for n, ad in e1_adapters.items()} |
                  {f"{n}.b": ad.b.detach().cpu().contiguous() for n, ad in e1_adapters.items()},
                  str(OUT_DIR / "e1_activearena_slot_rand_lora.safetensors"))

        m1.eval()
        e1_results = []
        for p in e1_probes:
            out_g = generate_with_hidden(m1, p, enc_336)
            sent = out_g["text"]
            pcls = classify_aa(sent)
            pcol = detect_color(sent)
            cls_ok = pcls == p["expected_class"]
            col_ok = (pcol == p["expected_color"]) if p["expected_color"] else True
            e1_results.append({
                "id": p["id"], "episode_id": p["episode_id"], "condition": p["condition"],
                "true_color": p["true_color"], "expected_class": p["expected_class"],
                "expected_color": p["expected_color"], "text": sent,
                "predicted_class": pcls, "predicted_color": pcol,
                "exact_ok": cls_ok and col_ok,
            })
        e1_summary = {}
        for cond in ("full", "jump_keep_slot2", "jump_keep_slot1", "smooth_no_evidence", "masked", "color_swap"):
            sub = [r for r in e1_results if r["condition"] == cond]
            e1_summary[cond] = {
                "n": len(sub),
                "exact_ok": sum(r["exact_ok"] for r in sub),
                "place_count": sum(r["predicted_class"] == "place" for r in sub),
                "inspect_count": sum(r["predicted_class"] == "inspect" for r in sub),
            }
        print("[E1 Summary]", json.dumps(e1_summary, indent=2), flush=True)
        receipt["e1_activearena_slot_rand"] = {
            "steps": e1_steps,
            "first_loss": round(e1_losses[0], 4),
            "last_loss": round(e1_losses[-1], 4),
            "wall_s": round(time.monotonic() - t_e1, 2),
            "summary": e1_summary,
        }
        (OUT_DIR / "e1_results.json").write_text(json.dumps({"summary": e1_summary, "results": e1_results}, indent=2) + "\n")
        del m1, e1_adapters, e1_opt, e1_params
        gc.collect()
        torch.cuda.empty_cache()

        # =========================================================================
        # SUB-EXPERIMENT E2: Open-Video G_mismatch Contrastive Adaptation (448x336, r=32)
        # =========================================================================
        print("=== Starting Sub-Experiment E2: Open-Video G_mismatch + Slot-Pos Adaptation ===", flush=True)
        t_e2 = time.monotonic()
        m2 = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0},
        )
        m2.requires_grad_(False)
        m2.model.language_model.gradient_checkpointing_disable()

        r32_projs = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        e2_adapters = {}
        for name, mod in list(m2.named_modules()):
            if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in r32_projs:
                parent, _, leaf = name.rpartition(".")
                w = LowRankLinear(mod, 32, 64)
                setattr(m2.get_submodule(parent), leaf, w)
                e2_adapters[name] = w

        # Initialize from 725c7958 Epoch 4 best validation checkpoint!
        best_ckpt = load_file(str(OPEN_RUN / "stage1_lora_r32_448x336_best.safetensors"))
        with torch.no_grad():
            for n, ad in e2_adapters.items():
                ad.a.copy_(best_ckpt[f"{n}.a"])
                ad.b.copy_(best_ckpt[f"{n}.b"])

        e2_params = [p for ad in e2_adapters.values() for p in (ad.a, ad.b)]
        lr_max, lr_min = 4e-5, 4e-6
        e2_opt = torch.optim.AdamW(e2_params, lr=lr_max, weight_decay=0.0)
        enc_448 = make_encoder(448, 336)

        # Stratified contrastive pools (strictly train split!)
        spec_pos_pool = [r for r in train_rows if r["goal_mode"] == "specific_target"] + slot_pos_rows
        rt_pool = [r for r in train_rows if r["source_dataset"] == "roboturk" and r["task_type"] == "next_task_selection"]
        os_pool = [r for r in train_rows if r["source_dataset"] == "os_ost_part3" and r["task_type"] == "next_task_selection"]

        def make_cycler(items):
            buf = []
            while True:
                if not buf:
                    buf = list(items)
                    rng.shuffle(buf)
                yield buf.pop()

        c_mis = make_cycler(mismatch_rows)
        c_pos = make_cycler(spec_pos_pool)
        c_rt = make_cycler(rt_pool)
        c_os = make_cycler(os_pool)

        e2_steps = 120
        micro_bs = 4
        m2.train()
        e2_losses = []
        for step in range(1, e2_steps + 1):
            prog = (step - 1) / max(e2_steps - 1, 1)
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
            for pg in e2_opt.param_groups:
                pg["lr"] = cur_lr

            # Effective BS=16: 4 G_mismatch negatives + 4 specific positives + 4 RoboTurk + 4 OS/OST
            batch = [next(c_mis) for _ in range(4)] + [next(c_pos) for _ in range(4)] + [next(c_rt) for _ in range(4)] + [next(c_os) for _ in range(4)]
            rng.shuffle(batch)

            e2_opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for ci in range(4):
                sub = batch[ci * micro_bs : (ci + 1) * micro_bs]
                sl = subbatch_loss(m2, sub, enc_448).mean() * 0.25
                sl.backward()
                step_loss += float(sl.detach())
            torch.nn.utils.clip_grad_norm_(e2_params, 1.0)
            e2_opt.step()
            e2_losses.append(step_loss)
            if step in (1, 30, 60, 90, 120):
                print(f"[E2] step {step}/{e2_steps} lr={cur_lr:.6f} loss={step_loss:.4f}", flush=True)

        save_file({f"{n}.a": ad.a.detach().cpu().contiguous() for n, ad in e2_adapters.items()} |
                  {f"{n}.b": ad.b.detach().cpu().contiguous() for n, ad in e2_adapters.items()},
                  str(OUT_DIR / "e2_open_video_gmismatch_lora.safetensors"))

        # Evaluate E2 across: (1) Val calib (60), (2) Canonical test (144), (3) Additional holdout (31), (4) 2x2 Open control (125)
        m2.eval()

        def eval_rows(rows_list: list[dict]) -> list[dict]:
            recs = []
            for r in rows_list:
                out_g = generate_with_hidden(m2, r, enc_448)
                adj = pilot_run.adjudicate_generation(r, out_g["text"])
                recs.append({
                    "row_id": r["id"], "pair_id": r.get("pair_id"), "source_dataset": r.get("source_dataset"),
                    "phase": r.get("phase"), "probe_type": r.get("probe_type"), "probe_scene": r.get("probe_scene"),
                    "source_pool": r.get("source_pool"), "visibility": r.get("visibility"), "condition": r.get("condition"),
                    "target_noun": r.get("target_noun"), "goal": r["goal"],
                    "mean_token_logprob": out_g["mean_token_logprob"], **adj, "_h_last": out_g["_h_last"],
                })
            return recs

        val_recs = eval_rows(val_calib_rows)
        test_recs = eval_rows(test_eval_rows)
        add_recs = eval_rows(add_rows)
        open_2x2_recs = eval_rows(open_2x2_probes)

        # Compute canonical 144-row metrics (exact 16 complete triplets!)
        os_single = [r for r in test_recs if not r["pair_id"] and r["source_dataset"] == "os_ost_part3"]
        rt_single = [r for r in test_recs if not r["pair_id"] and r["source_dataset"] == "roboturk"]
        by_canon_pair = defaultdict(dict)
        for r in test_recs:
            if r["pair_id"] and r["phase"] in ("pre_reveal", "post_reveal", "post_reveal_masked"):
                by_canon_pair[r["pair_id"]][r["phase"]] = r
        canon_16 = {k: v for k, v in by_canon_pair.items() if {"pre_reveal", "post_reveal", "post_reveal_masked"} <= v.keys()}

        canon_summary = {
            "os_class_45": sum(r["strict_class_ok"] for r in os_single),
            "os_search_10": sum(r["strict_class_ok"] for r in os_single if r["expected_class"] == "search"),
            "rt_class_45": sum(r["strict_class_ok"] for r in rt_single),
            "rt_search_10": sum(r["strict_class_ok"] for r in rt_single if r["expected_class"] == "search"),
            "rt_acquire_10": sum(r["strict_class_ok"] for r in rt_single if r["expected_class"] == "acquire"),
            "triplets_n": len(canon_16),
            "pre_to_post_16": sum(v["pre_reveal"]["strict_class_ok"] and v["post_reveal"]["strict_class_ok"] for v in canon_16.values()),
            "post_named_16": sum(v["post_reveal"]["strict_exact"] for v in canon_16.values()),
            "same_frame_flip_16": sum(v["post_reveal"]["strict_class_ok"] and v["post_reveal_masked"]["strict_class_ok"] for v in canon_16.values()),
            "same_frame_named_flip_16": sum(v["post_reveal"]["strict_exact"] and v["post_reveal_masked"]["strict_class_ok"] for v in canon_16.values()),
        }

        # Compute additional holdout metrics (9 new triplets + 2 cross-goal scenes!)
        by_add_pair = defaultdict(dict)
        for r in add_recs:
            if r["probe_type"] == "new_video_triplet":
                by_add_pair[r["pair_id"]][r["phase"]] = r
        gc_by_scene = defaultdict(list)
        for r in add_recs:
            if r["probe_type"] == "goal_cross":
                gc_by_scene[r["probe_scene"]].append(r)

        add_summary = {
            "new_video_pre_post_9": sum(v["pre_reveal"]["strict_class_ok"] and v["post_reveal"]["strict_class_ok"] for v in by_add_pair.values()),
            "new_video_named_flip_9": sum(v["post_reveal"]["strict_exact"] and v["post_reveal_masked"]["strict_class_ok"] for v in by_add_pair.values()),
            "goal_cross_both_ok_2": sum(all(x["strict_exact"] for x in grp) for grp in gc_by_scene.values()),
            "goal_cross_details": [
                {"scene": r["probe_scene"], "noun": r["target_noun"], "exp": r["expected_class"], "pred": r["predicted_class"], "exact": r["strict_exact"], "text": r["text"]}
                for r in add_recs if r["probe_type"] == "goal_cross"
            ],
        }

        # Compute 2x2 Open-Video summary
        def sum_2x2(sub):
            res = {}
            for cond in ("full", "early_jump_keep", "late_jump_keep", "smooth_no_evidence", "masked"):
                cr = [r for r in sub if r["condition"] == cond]
                res[cond] = {
                    "n": len(cr),
                    "acquire": sum(r["predicted_class"] == "acquire" for r in cr),
                    "acquire_named": sum(r["predicted_class"] == "acquire" and r["noun_ok"] for r in cr),
                    "search": sum(r["predicted_class"] == "search" for r in cr),
                }
            return res

        open_2x2_summary = {
            "all_25": sum_2x2(open_2x2_recs),
            "ambiguous_12": sum_2x2([r for r in open_2x2_recs if r["visibility"] != "clear"]),
            "clear_13": sum_2x2([r for r in open_2x2_recs if r["visibility"] == "clear"]),
        }

        # Compute Ridge + Temperature-Scaled Confidence Calibration
        X_cal = torch.stack([r["_h_last"] for r in val_recs]).float()
        y_cal = torch.tensor([1.0 if r["accepted"] else 0.0 for r in val_recs]).float()
        X_test = torch.stack([r["_h_last"] for r in test_recs]).float()
        y_test = [1 if r["accepted"] else 0 for r in test_recs]
        test_domains = ["triplet" if r["pair_id"] else r["source_dataset"] for r in test_recs]
        conf_summary, _ = fit_ridge_temp_confidence(X_cal, y_cal, X_test, y_test, test_domains)

        # Save clean outputs
        for fname, rlist in (
            ("e2_canonical_test_generations.json", test_recs),
            ("e2_additional_holdout_generations.json", add_recs),
            ("e2_open_2x2_generations.json", open_2x2_recs),
        ):
            (OUT_DIR / fname).write_text(json.dumps([{k: v for k, v in r.items() if k != "_h_last"} for r in rlist], indent=2, ensure_ascii=False) + "\n")

        receipt["e2_open_video_gmismatch"] = {
            "steps": e2_steps,
            "first_loss": round(e2_losses[0], 4),
            "last_loss": round(e2_losses[-1], 4),
            "wall_s": round(time.monotonic() - t_e2, 2),
            "canonical_144": canon_summary,
            "additional_holdout": add_summary,
            "open_2x2": open_2x2_summary,
            "ridge_temp_confidence": conf_summary,
        }
        receipt["status"] = "completed"
        receipt["wall_seconds"] = round(time.monotonic() - started, 2)
        receipt["gpu_hours_upper_bound"] = round(receipt["wall_seconds"] / 3600, 4)
        (OUT_DIR / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
    finally:
        stopping.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()
