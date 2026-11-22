#!/usr/bin/env python3
"""Stage-1 v2 Full-Scale Rebalanced 6-Epoch Training (448x336, LoRA Rank-32, BS=16, No-GC).

Key Upgrades over Stage-1 v1 (`725c7958`):
1. Expanded 100% English Role B corpus (6,000+ causal-checked cases, strictly 0 ActiveArena rows,
   strictly disjoint from canonical 144-row test AND the 4 brand-new additional holdout OS recordings).
2. Rebalanced 4:2:4:4:2 BS=16 stratified composition:
   - 4/16 (25.0%): Specific-target positives (pre/post/masked + slot-randomized jump-cut positives)
   - 2/16 (12.5%): Anti-false-negative diverse `G_mismatch` contrastive negatives (smooth close-up A + absent B -> search B)
   - 4/16 (25.0%): RoboTurk generic/multi-solution prefixes
   - 4/16 (25.0%): OS/OST Part 3 generic/multi-solution prefixes
   - 2/16 (12.5%): Activity-description visual-language alignment
3. Strict 16-complete-triplet denominator + `accepted = strict_class_ok and noun_ok` adjudication.
4. Larger validation calibration set + Positive-Constrained (`softplus`) 5-Fold CV Ridge + Temperature-Scaled
   Confidence Head evaluated both overall and per-domain (OS / RoboTurk / Triplet).
5. Built-in end-of-run evaluation on best validation checkpoint across:
   - Canonical 144-row test pool
   - 4 brand-new held-out OS videos (9 new triplets + 2 cross-goal `goal_cross` scenes)
   - Full 25-triplet 2x2 Jump-Cut vs. Evidence-Removal Factorial Control (stratified by `current_frame_audit.csv`).
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
import traceback

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ACTION_WORDS = {
    "acquire": r"regrasp|grasp|grab|pick up|retrieve|take|lift|hold|pull out|acquire|reach into|reach down|lower the (?:empty )?gripper into",
    "carry": r"carry|transport|transfer|bring|move (?:the |an )?(?:(?:held|grasped) (?:object|item|bottle|bag|device|can|ball)|object|item|bottle|bag|device|can|ball).*?(?:toward|to)(?:s| the| a)? (?:sort|box|hole|board|pack|location)",
    "place": r"align|place|insert|drop|release|put|deposit|pack|set down|lower (?:the |an )?(?:held|grasped|object|item|bottle)",
    "return": r"return|reach back|move (?:the )?(?:empty )?gripper back|pull (?:the )?(?:empty )?gripper back|retract",
    "search": r"scan|inspect|look (?:for|into|around|at)|search|probe|verify|check|expose|reveal|shift|uncover|move .*?aside|push .*?aside|pull .*?aside|clear (?:the |an )?(?:obstruction|cloth|cover|material)|open the (?:bag|drawer|box|container)",
}
ACTION_PATTERNS = {
    action: re.compile(r"\b(?:" + words + r")\b", flags=re.IGNORECASE)
    for action, words in ACTION_WORDS.items()
}


def classify(sentence: str) -> str:
    s = sentence.strip().lower()
    if re.search(
        r"\b(?:move|push|pull|shift|sweep|lift|clear|displace)\b.*?\b(?:aside|away|obstruction|cloth|fabric|towel|cover|covering|clutter|material)\b",
        s,
    ):
        return "search"
    if re.search(
        r"\b(?:set|place|put|drop|release|insert|align|deposit|pack|lower)\b.*?\b(?:on|onto|into|in|down|at|table|box|opening|slot|bag|location)\b",
        s,
    ) and not s.startswith(("reach ", "grasp ", "pick up ", "search ", "inspect ", "scan ")):
        return "place"
    matches = [
        (match.start(), action)
        for action, pattern in ACTION_PATTERNS.items()
        if (match := pattern.search(s))
    ]
    if matches:
        return min(matches)[1]
    if s.startswith(("move ", "shift ", "open ", "scan ", "inspect ", "check ", "look ", "search ", "probe ")):
        return "search"
    if s.startswith(("grasp ", "pick ", "grab ", "retrieve ", "lift ", "take ", "reach for ")):
        return "acquire"
    if s.startswith(("carry ", "transfer ", "transport ", "bring ")):
        return "carry"
    if s.startswith(("place ", "put ", "insert ", "drop ", "align ", "set ")):
        return "place"
    return "other"


def check_target_noun_match(sentence: str, target_noun: str | None) -> bool:
    if not target_noun:
        return True
    sl = sentence.lower()
    nl = target_noun.lower().strip()
    if nl in sl:
        return True
    tokens = [t for t in re.split(r"[\s\-]+", nl) if len(t) >= 3 and t not in {"the", "and", "item", "object", "small", "flat"}]
    if tokens and all(t in sl for t in tokens):
        return True
    if len(tokens) >= 2 and sum(t in sl for t in tokens) >= len(tokens) - 1:
        return True
    return False


def adjudicate_generation(row: dict, sentence: str) -> dict:
    pred_cls = classify(sentence)
    exp_cls = row["expected_action_class"]
    strict_class_ok = pred_cls == exp_cls

    alt_cls = classify(row.get("alt_candidate", "")) if row.get("alt_candidate") else None
    if row.get("multi_solution") and alt_cls and alt_cls != "other":
        acceptable_class_ok = pred_cls in {exp_cls, alt_cls}
    else:
        acceptable_class_ok = strict_class_ok

    noun_ok = True
    if row.get("goal_mode") == "specific_target" and pred_cls == "acquire" and row.get("target_noun"):
        noun_ok = check_target_noun_match(sentence, row["target_noun"])

    strict_exact = strict_class_ok and noun_ok
    if row.get("pair_id"):
        accepted = strict_class_ok and noun_ok
    else:
        accepted = acceptable_class_ok and noun_ok

    reason = "accepted"
    if not accepted:
        if not noun_ok:
            reason = "wrong_target_noun"
        elif row.get("phase") in ("pre_reveal", "post_reveal_masked") and pred_cls in ("acquire", "carry", "place"):
            reason = "premature_action_without_evidence"
        elif row.get("phase") == "post_reveal" and pred_cls == "search":
            reason = "redundant_search_after_evidence"
        else:
            reason = f"wrong_action_class_{pred_cls}_expected_{exp_cls}"

    return {
        "text": sentence,
        "predicted_class": pred_cls,
        "expected_class": exp_cls,
        "strict_class_ok": strict_class_ok,
        "acceptable_class_ok": acceptable_class_ok,
        "noun_ok": noun_ok,
        "strict_exact": strict_exact,
        "accepted": accepted,
        "reason": reason,
    }


def select_stratified_subset(rows: list[dict], max_count: int, seed: int) -> list[dict]:
    if len(rows) <= max_count:
        return list(rows)
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["expected_action_class"]].append(r)
    for grp in by_cls.values():
        rng.shuffle(grp)
    selected = []
    labels = sorted(by_cls.keys())
    while len(selected) < max_count and any(by_cls.values()):
        for lbl in labels:
            if by_cls[lbl] and len(selected) < max_count:
                selected.append(by_cls[lbl].pop())
    return selected


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def compute_confidence_metrics(labels: list[int], probs: list[float]) -> dict:
    n = len(labels)
    brier = sum((p - y) ** 2 for y, p in zip(labels, probs)) / max(n, 1)
    auroc = compute_auroc(labels, probs)
    bins = {
        "low_lt_0.4": [y for y, p in zip(labels, probs) if p < 0.4],
        "mid_0.4_0.7": [y for y, p in zip(labels, probs) if 0.4 <= p < 0.7],
        "high_ge_0.7": [y for y, p in zip(labels, probs) if p >= 0.7],
    }
    bin_summary = {
        k: {
            "count": len(v),
            "acceptance_rate": round(sum(v) / len(v), 4) if v else None,
        }
        for k, v in bins.items()
    }
    return {
        "n": n,
        "positive_rate": round(sum(labels) / max(n, 1), 4),
        "brier_score": round(brier, 4),
        "auroc": round(auroc, 4),
        "bins": bin_summary,
    }


def trigger_live_dashboard() -> None:
    script = Path(CONFIG.get("dashboard_tracker_script", ""))
    if script.is_file():
        try:
            subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass


def compute_cosine_lr(step_idx: int, total_steps: int, lr_max: float, lr_min: float, warmup_ratio: float) -> float:
    warmup_steps = max(int(round(total_steps * warmup_ratio)), 1)
    if step_idx <= warmup_steps:
        return lr_max * (step_idx / warmup_steps)
    progress = (step_idx - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def prepare(rebuild_suite: bool = False):
    if rebuild_suite:
        import build_stage1_formal_suite
        build_stage1_formal_suite.main()

    manifest = HERE / "stage1_formal_suite.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert all(r["source_dataset"] in ("roboturk", "os_ost_part3") for r in rows), "Non-Stage-1 row detected!"

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    train_parts = {r["participant_id"] for r in train_rows}
    val_parts = {r["participant_id"] for r in val_rows}
    test_parts = {r["participant_id"] for r in test_rows}
    assert not (train_parts & val_parts), "Train/Val participant overlap!"
    assert not (train_parts & test_parts), "Train/Test participant overlap!"

    train_vids = {r["source_video_sha256"] for r in train_rows}
    add_holdout = Path(CONFIG["additional_holdout_path"])
    if add_holdout.is_file():
        add_vids = {json.loads(l)["source_video_sha256"] for l in add_holdout.read_text().splitlines()}
        assert not (train_vids & add_vids), "Additional holdout video leaked into train!"

    for r in rows:
        assert len(r["images"]) == 4
        for im in r["images"]:
            if not Path(im["path"]).is_file():
                raise FileNotFoundError(im["path"])

    w, h = CONFIG["image_width"], CONFIG["image_height"]
    assert w % 28 == 0 and h % 28 == 0
    tokens_per_frame = (w // 28) * (h // 28)

    # Compute steps_per_epoch based on non-augmented next_task_selection train rows (capped at 240 for ~2.2 GPU-h budget)
    base_next_task_train = [
        r for r in train_rows
        if r["task_type"] == "next_task_selection" and r.get("goal_mode") != "specific_target_mismatch"
    ]
    steps_per_epoch = min(240, max(180, math.ceil(len(base_next_task_train) / CONFIG["batch_size"])))
    total_steps = steps_per_epoch * CONFIG["epochs"]

    summary = {
        "suite_version": "stage1_v2_rebalanced",
        "preflight_total_rows": len(rows),
        "activearena_rows": 0,
        "language": "en",
        "image_resolution": f"{w}x{h}",
        "visual_tokens_per_frame": tokens_per_frame,
        "visual_tokens_per_prefix": tokens_per_frame * 4,
        "gradient_checkpointing": CONFIG["gradient_checkpointing"],
        "batch_size": CONFIG["batch_size"],
        "micro_batch_size": CONFIG["micro_batch_size"],
        "batch_mix_4_2_4_4_2": CONFIG["batch_mix_4_2_4_4_2"],
        "epochs": CONFIG["epochs"],
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_rows": len(train_rows),
        "train_gmismatch_rows": sum(1 for r in train_rows if r.get("goal_mode") == "specific_target_mismatch"),
        "train_slot_pos_rows": sum(1 for r in train_rows if r.get("family") == "stage1_v2_slot_pos"),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
    }
    print(json.dumps(summary), flush=True)
    return rows, train_rows, val_rows, test_rows, summary


def build_stratified_v2_schedule(train_rows: list[dict], total_steps: int, batch_size: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    spec_pos_pool = [r for r in train_rows if r["goal_mode"] == "specific_target"]
    gmismatch_pool = [r for r in train_rows if r["goal_mode"] == "specific_target_mismatch"]
    rt_pool = [r for r in train_rows if r["source_dataset"] == "roboturk" and r["task_type"] == "next_task_selection" and r["goal_mode"] == "generic_goal"]
    os_pool = [r for r in train_rows if r["source_dataset"] == "os_ost_part3" and r["task_type"] == "next_task_selection" and r["goal_mode"] == "generic_goal"]
    act_pool = [r for r in train_rows if r["task_type"] == "activity_description"]
    assert spec_pos_pool and gmismatch_pool and rt_pool and os_pool and act_pool, "One of the 5 v2 training pools is empty!"

    def make_cycler(items: list[dict]):
        buf = []
        while True:
            if not buf:
                buf = list(items)
                rng.shuffle(buf)
            yield buf.pop()

    c_spec = make_cycler(spec_pos_pool)
    c_mis = make_cycler(gmismatch_pool)
    c_rt = make_cycler(rt_pool)
    c_os = make_cycler(os_pool)
    c_act = make_cycler(act_pool)

    mix = CONFIG["batch_mix_4_2_4_4_2"]
    batches = []
    for _ in range(total_steps):
        batch = (
            [next(c_spec) for _ in range(mix["specific_pos"])]
            + [next(c_mis) for _ in range(mix["gmismatch_neg"])]
            + [next(c_rt) for _ in range(mix["roboturk_generic"])]
            + [next(c_os) for _ in range(mix["os_generic"])]
            + [next(c_act) for _ in range(mix["activity_description"])]
        )
        rng.shuffle(batch)
        assert len(batch) == batch_size
        batches.append(batch)
    return batches


def fit_positive_ridge_temp_confidence(X_cal, y_cal, lp_cal, X_test, y_test_list, lp_test, test_domains: list[str]):
    """5-Fold CV L2-normalized Ridge Logistic Regression + Positive-Constrained (softplus) Temperature Scaling."""
    import torch
    from torch import nn

    h_mean = X_cal.mean(dim=0, keepdim=True)
    h_std = X_cal.std(dim=0, keepdim=True).clamp_min(1e-5)
    Xc_unit = nn.functional.normalize((X_cal - h_mean) / h_std, dim=-1)
    Xt_unit = nn.functional.normalize((X_test - h_mean) / h_std, dim=-1)

    n = Xc_unit.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    best_lam = 15.0
    best_cv_brier = 1e9
    best_oof_logits = None

    for lam in (1.0, 3.0, 10.0, 25.0, 60.0, 150.0):
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

    # Positive-constrained Platt/Temperature scaling: p = sigmoid(softplus(log_a) * z + b)
    log_a = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ~= 1.0
    bias_t = nn.Parameter(torch.tensor(0.0))
    t_opt = torch.optim.LBFGS([log_a, bias_t], lr=0.5, max_iter=50)

    def t_closure():
        t_opt.zero_grad()
        a_pos = nn.functional.softplus(log_a) + 1e-3
        loss = nn.functional.binary_cross_entropy_with_logits(a_pos * best_oof_logits + bias_t, y_cal)
        loss.backward()
        return loss

    t_opt.step(t_closure)

    # Fit final Ridge head on all validation rows
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

    # Positive-constrained calibrated token log-prob baseline
    lp_log_a = nn.Parameter(torch.tensor(0.5413))
    lp_b = nn.Parameter(torch.tensor(0.0))
    lp_opt = torch.optim.LBFGS([lp_log_a, lp_b], lr=0.5, max_iter=50)

    def lp_closure():
        lp_opt.zero_grad()
        a_pos = nn.functional.softplus(lp_log_a) + 1e-3
        loss = nn.functional.binary_cross_entropy_with_logits(a_pos * lp_cal + lp_b, y_cal)
        loss.backward()
        return loss

    lp_opt.step(lp_closure)

    with torch.no_grad():
        a_pos_val = float((nn.functional.softplus(log_a) + 1e-3).item())
        b_val = float(bias_t.item())
        raw_test_logits = final_head(Xt_unit).squeeze(-1)
        cal_test_probs = torch.sigmoid(a_pos_val * raw_test_logits + b_val).tolist()

        lp_a_val = float((nn.functional.softplus(lp_log_a) + 1e-3).item())
        lp_test_probs = torch.sigmoid(lp_a_val * lp_test + float(lp_b.item())).tolist()

        const_p = float(y_cal.mean().item())
        const_probs = [const_p for _ in y_test_list]

    by_domain = {}
    for dom in ("all", "os_ost_part3", "roboturk", "triplet"):
        idxs = [i for i, d in enumerate(test_domains) if dom == "all" or d == dom]
        if not idxs:
            continue
        yl = [y_test_list[i] for i in idxs]
        pl = [cal_test_probs[i] for i in idxs]
        lpl = [lp_test_probs[i] for i in idxs]
        cl = [const_probs[i] for i in idxs]
        by_domain[dom] = {
            "constant_baseline": compute_confidence_metrics(yl, cl),
            "calibrated_token_logprob": compute_confidence_metrics(yl, lpl),
            "ridge_temp_hidden_c": compute_confidence_metrics(yl, pl),
        }

    state = {
        "weight": final_head.weight.detach().cpu(),
        "bias": final_head.bias.detach().cpu(),
        "h_mean": h_mean.detach().cpu(),
        "h_std": h_std.detach().cpu(),
        "best_lam": best_lam,
        "temp_a_pos": a_pos_val,
        "temp_b": b_val,
    }
    summary = {
        "calibration_n": n,
        "test_n": len(y_test_list),
        "best_lam": best_lam,
        "cv_val_brier": round(best_cv_brier, 4),
        "temp_scale_a_pos": round(a_pos_val, 4),
        "temp_scale_b": round(b_val, 4),
        "by_domain": by_domain,
    }
    return summary, state


def main():
    if sys.argv[1:] == ["--cpu-preflight"]:
        prepare(rebuild_suite=False)
        return
    if not os.environ.get("ORX_RUN_ID"):
        raise RuntimeError("Use orx exp run to launch GPU work")

    started = time.monotonic()
    run_id = os.environ["ORX_RUN_ID"]
    out = Path(CONFIG["output_root"]) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    limits = CONFIG["budget"]
    receipt = {
        "run_id": run_id,
        "suite_version": "stage1_v2_rebalanced",
        "commit": os.environ.get("ORX_COMMIT"),
        "status": "running",
        "language": "en",
        "activearena_rows_used": 0,
        "image_resolution": f"{CONFIG['image_width']}x{CONFIG['image_height']}",
        "gradient_checkpointing": False,
        "epochs": CONFIG["epochs"],
        "first_gpu_seen_s": None,
        "peak_process_mib": 0,
        "train_steps": 0,
        "external_api_calls": 0,
    }
    stopped = threading.Event()

    def save():
        tmp = out / "receipt.tmp"
        tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(out / "receipt.json")

    def watch():
        while not stopped.wait(0.5):
            elapsed = time.monotonic() - started
            reason = "wall_seconds" if elapsed >= limits["hard_wall_seconds"] else None
            try:
                info = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    text=True, timeout=2,
                )
                for entry in info.splitlines():
                    pid, memory = (part.strip() for part in entry.split(","))
                    if pid == str(os.getpid()):
                        mib = int(memory)
                        if receipt["first_gpu_seen_s"] is None:
                            receipt["first_gpu_seen_s"] = elapsed
                        receipt["peak_process_mib"] = max(mib, receipt["peak_process_mib"])
                        if mib > limits["hard_process_mib"]:
                            reason = "process_vram"
            except (subprocess.SubprocessError, ValueError):
                pass
            if reason:
                receipt.update(
                    status="stopped_by_budget",
                    reason=reason,
                    wall_seconds=elapsed,
                    gpu_hours_upper_bound=elapsed / 3600,
                )
                save()
                print(json.dumps(receipt), flush=True)
                os._exit(124)

    save()
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()

    try:
        rows, train_rows, val_rows, test_rows, preflight_summary = prepare(rebuild_suite=False)
        (out / "stage1_formal_suite.jsonl").write_bytes((HERE / "stage1_formal_suite.jsonl").read_bytes())
        (out / "stage1_formal_summary.json").write_bytes((HERE / "stage1_formal_summary.json").read_bytes())
        receipt["dataset_summary"] = preflight_summary
        save()
        trigger_live_dashboard()

        import torch
        from torch import nn
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from safetensors.torch import load_file, save_file

        torch.manual_seed(CONFIG["seed"])
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free / 2**20 < limits["min_free_mib"]:
            raise RuntimeError(f"GPU free memory too low: {free / 2**20:.0f} MiB")
        torch.cuda.set_per_process_memory_fraction(limits["torch_allocator_mib"] * 2**20 / total)
        receipt["cuda_start_s"] = time.monotonic() - started

        processor = AutoProcessor.from_pretrained(CONFIG["model_path"], local_files_only=True)
        encoded_cache: dict[str, tuple[dict, torch.Tensor]] = {}
        verified_sha_cache: set[str] = set()
        img_w, img_h = CONFIG["image_width"], CONFIG["image_height"]

        def encode_row(row: dict) -> tuple[dict, torch.Tensor]:
            cid = row["id"]
            if cid in encoded_cache:
                return encoded_cache[cid]
            if len(encoded_cache) >= 384:
                del encoded_cache[next(iter(encoded_cache))]
            images = []
            for frame in row["images"]:
                path = Path(frame["path"])
                p_str = str(path)
                if p_str not in verified_sha_cache:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                        raise ValueError(f"Frame hash mismatch: {path}")
                    verified_sha_cache.add(p_str)
                with Image.open(path) as img:
                    images.append(img.convert("RGB").resize((img_w, img_h), Image.Resampling.LANCZOS))
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
            messages = [{"role": "user", "content": [*[{"type": "image"} for _ in images], {"type": "text", "text": prompt_text}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = dict(processor(images=images, text=[prompt], return_tensors="pt"))
            if encoded["input_ids"].shape[-1] > CONFIG["max_prompt_tokens"]:
                raise ValueError(f"Prompt exceeds budget: {encoded['input_ids'].shape[-1]}")
            target_ids = processor.tokenizer(
                row.get("target", "Inspect") + "<|im_end|>", add_special_tokens=False, return_tensors="pt"
            ).input_ids
            encoded_cache[cid] = (encoded, target_ids)
            return encoded, target_ids

        # Lock canonical 144-row test evaluation suite for exact apples-to-apples comparison
        canon_gen_file = Path(CONFIG["canonical_ref_dir"]) / "lora_r32_448x336_step378_test_generations.json"
        row_by_id = {r["id"]: r for r in rows}
        canonical_ids = [g["row_id"] for g in json.loads(canon_gen_file.read_text())]
        test_eval_rows = [row_by_id[rid] for rid in canonical_ids if rid in row_by_id]
        assert len(test_eval_rows) == 144, f"Expected 144 canonical test rows, got {len(test_eval_rows)}"

        # Expanded validation calibration pool (all val pairs + up to 120 stratified val single prefixes!)
        val_pair_rows = [r for r in val_rows if r.get("pair_id")]
        val_single_sub = select_stratified_subset(
            [r for r in val_rows if not r.get("pair_id")],
            CONFIG["val_eval_per_source"],
            CONFIG["seed"],
        )
        val_calib_rows = val_pair_rows + val_single_sub

        def send(inputs: dict) -> dict:
            return {
                key: (value.to(device="cuda", dtype=torch.bfloat16) if value.is_floating_point() else value.cuda())
                for key, value in inputs.items()
            }

        def subbatch_losses(active_model, batch_rows: list[dict]) -> torch.Tensor:
            encoded_pairs = [encode_row(r) for r in batch_rows]
            prompts = [enc["input_ids"][0] for enc, _ in encoded_pairs]
            answers = [tgt[0] for _, tgt in encoded_pairs]
            prompt_width = max(x.numel() for x in prompts)
            answer_width = max(x.numel() for x in answers)
            ids = torch.full((len(batch_rows), prompt_width + answer_width), processor.tokenizer.pad_token_id)
            attention = torch.zeros_like(ids)
            labels = torch.full((len(batch_rows), answer_width), -100)
            for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                start = prompt_width - prompt.numel()
                end = prompt_width + answer.numel()
                ids[i, start:prompt_width] = prompt
                ids[i, prompt_width:end] = answer
                attention[i, start:end] = 1
                labels[i, :answer.numel()] = answer
            inputs = {"input_ids": ids, "attention_mask": attention}
            for key in ("pixel_values", "image_grid_thw"):
                inputs[key] = torch.cat([enc[key] for enc, _ in encoded_pairs], dim=0)
            logits = active_model(**send(inputs), use_cache=False, logits_to_keep=answer_width + 1).logits[:, :-1]
            labels = labels.cuda()
            token_losses = nn.functional.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")
            return token_losses.sum(dim=1) / labels.ne(-100).sum(dim=1)

        def generate_and_extract_hidden(active_model, row: dict) -> dict:
            encoded, _ = encode_row(row)
            inputs = send(encoded)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen_ids = active_model.generate(**inputs, max_new_tokens=28, do_sample=False)
                new_ids = gen_ids[0, prompt_len:]
                sentence = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                full_attention = torch.ones_like(gen_ids)
                other = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
                n_new = max(int(new_ids.numel()), 1)
                out_fw = active_model(
                    input_ids=gen_ids,
                    attention_mask=full_attention,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=n_new + 1,
                    **other,
                )
                step_logits = out_fw.logits[0, :-1].float()
                log_probs = torch.log_softmax(step_logits, dim=-1)
                tok_lps = [float(log_probs[idx, int(tok_id)]) for idx, tok_id in enumerate(new_ids)]
                mean_lp = sum(tok_lps) / max(len(tok_lps), 1)
                last_layer_h = out_fw.hidden_states[-1][0]
                im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if int(new_ids[-1]) == im_end_id and new_ids.numel() >= 2:
                    h_last = last_layer_h[-2].float().detach().cpu()
                else:
                    h_last = last_layer_h[-1].float().detach().cpu()

            adj = adjudicate_generation(row, sentence)
            adj["mean_token_logprob"] = round(mean_lp, 4)
            adj["mean_token_prob"] = round(math.exp(max(min(mean_lp, 0.0), -20.0)), 4)
            return {
                "row_id": row["id"],
                "pair_id": row.get("pair_id"),
                "source_dataset": row["source_dataset"],
                "split": row.get("split"),
                "phase": row.get("phase"),
                "probe_type": row.get("probe_type"),
                "probe_scene": row.get("probe_scene"),
                "visibility": row.get("visibility"),
                "condition": row.get("condition"),
                "source_pool": row.get("source_pool"),
                "history_view": row.get("history_view", "full"),
                "goal_mode": row.get("goal_mode"),
                "target_noun": row.get("target_noun"),
                "goal": row["goal"],
                "reference_target": row.get("target"),
                **adj,
                "_h_last": h_last,
            }

        def evaluate_policy(active_model, policy_name: str, eval_rows: list[dict]) -> tuple[dict, list[dict]]:
            active_model.eval()
            records = [generate_and_extract_hidden(active_model, r) for r in eval_rows]

            pair_recs = [r for r in records if r.get("pair_id")]
            by_pair = defaultdict(dict)
            for r in pair_recs:
                if r.get("phase") in ("pre_reveal", "post_reveal", "post_reveal_masked"):
                    by_pair[r["pair_id"]][r["phase"]] = r

            # Exact complete triplets denominator (16 on canonical test, 8 on val!)
            complete_triplets = {
                pid: ph for pid, ph in by_pair.items()
                if {"pre_reveal", "post_reveal", "post_reveal_masked"} <= ph.keys()
            }
            n_triplets = len(complete_triplets)
            pair_both_ok = sum(
                ph["pre_reveal"]["strict_class_ok"] and ph["post_reveal"]["strict_class_ok"]
                for ph in complete_triplets.values()
            )
            cf_flip_ok = sum(
                ph["post_reveal"]["strict_class_ok"] and ph["post_reveal_masked"]["strict_class_ok"]
                for ph in complete_triplets.values()
            )
            cf_named_flip_ok = sum(
                ph["post_reveal"]["strict_exact"] and ph["post_reveal_masked"]["strict_class_ok"]
                for ph in complete_triplets.values()
            )
            post_named_ok = sum(ph["post_reveal"]["strict_exact"] for ph in complete_triplets.values())

            pre_recs = [ph["pre_reveal"] for ph in complete_triplets.values()]
            post_recs = [ph["post_reveal"] for ph in complete_triplets.values()]
            mask_recs = [ph["post_reveal_masked"] for ph in complete_triplets.values()]
            os_recs = [r for r in records if not r.get("pair_id") and r["source_dataset"] == "os_ost_part3"]
            rt_recs = [r for r in records if not r.get("pair_id") and r["source_dataset"] == "roboturk"]

            summary = {
                "policy": policy_name,
                "image_resolution": f"{img_w}x{img_h}",
                "total_evaluated": len(records),
                "evidence_pairs": {
                    "n_rows": len(pair_recs),
                    "n_complete_triplets": n_triplets,
                    "overall_class_correct": sum(r["strict_class_ok"] for r in pair_recs),
                    "overall_exact_correct": sum(r["strict_exact"] for r in pair_recs),
                    "pre_reveal_search_hits": f"{sum(r['strict_class_ok'] for r in pre_recs)}/{n_triplets}",
                    "post_reveal_acquire_hits": f"{sum(r['strict_class_ok'] for r in post_recs)}/{n_triplets}",
                    "post_reveal_named_hits": f"{post_named_ok}/{n_triplets}",
                    "masked_history_revert_to_search": f"{sum(r['strict_class_ok'] for r in mask_recs)}/{n_triplets}",
                    "pair_action_reversal": f"{pair_both_ok}/{n_triplets}",
                    "same_frame_counterfactual_flip": f"{cf_flip_ok}/{n_triplets}",
                    "same_frame_named_flip": f"{cf_named_flip_ok}/{n_triplets}",
                    "pair_action_reversal_count": pair_both_ok,
                    "same_frame_counterfactual_flip_count": cf_flip_ok,
                    "same_frame_named_flip_count": cf_named_flip_ok,
                },
                "os_ost_part3_heldout_seq_1_7": {
                    "n": len(os_recs),
                    "strict_class_correct": sum(r["strict_class_ok"] for r in os_recs),
                    "strict_class_acc": round(sum(r["strict_class_ok"] for r in os_recs) / max(len(os_recs), 1), 4),
                    "acceptable_multi_solution_correct": sum(r["acceptable_class_ok"] for r in os_recs),
                    "acceptable_multi_solution_acc": round(sum(r["acceptable_class_ok"] for r in os_recs) / max(len(os_recs), 1), 4),
                    "search_hits": f"{sum(r['strict_class_ok'] for r in os_recs if r['expected_class'] == 'search')}/{sum(1 for r in os_recs if r['expected_class'] == 'search')}",
                    "acquire_hits": f"{sum(r['strict_class_ok'] for r in os_recs if r['expected_class'] == 'acquire')}/{sum(1 for r in os_recs if r['expected_class'] == 'acquire')}",
                },
                "roboturk_heldout_unseen_operators": {
                    "n": len(rt_recs),
                    "strict_class_correct": sum(r["strict_class_ok"] for r in rt_recs),
                    "strict_class_acc": round(sum(r["strict_class_ok"] for r in rt_recs) / max(len(rt_recs), 1), 4),
                    "acceptable_multi_solution_correct": sum(r["acceptable_class_ok"] for r in rt_recs),
                    "acceptable_multi_solution_acc": round(sum(r["acceptable_class_ok"] for r in rt_recs) / max(len(rt_recs), 1), 4),
                    "search_hits": f"{sum(r['strict_class_ok'] for r in rt_recs if r['expected_class'] == 'search')}/{sum(1 for r in rt_recs if r['expected_class'] == 'search')}",
                    "acquire_hits": f"{sum(r['strict_class_ok'] for r in rt_recs if r['expected_class'] == 'acquire')}/{sum(1 for r in rt_recs if r['expected_class'] == 'acquire')}",
                },
            }
            print(json.dumps(summary), flush=True)
            return summary, records

        # =========================================================================
        # Stage-1 v2 Rebalanced 6-Epoch High-Resolution (448x336) Rank-32 LoRA
        # =========================================================================
        t_train = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        r32_cfg = CONFIG["lora_r32"]
        model = AutoModelForImageTextToText.from_pretrained(
            CONFIG["model_path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        model.requires_grad_(False)
        model.model.language_model.gradient_checkpointing_disable()

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

        adapters_r32: dict[str, LowRankLinear] = {}
        for name, module in list(model.named_modules()):
            if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in r32_cfg["projections"]:
                parent, _, leaf = name.rpartition(".")
                wrapper = LowRankLinear(module, r32_cfg["rank"], r32_cfg["alpha"])
                setattr(model.get_submodule(parent), leaf, wrapper)
                adapters_r32[name] = wrapper
        if len(adapters_r32) != r32_cfg["expected_adapters"]:
            raise ValueError(f"Expected {r32_cfg['expected_adapters']} adapters, found {len(adapters_r32)}")
        params_r32 = [par for ad in adapters_r32.values() for par in (ad.a, ad.b)]
        n_params_r32 = sum(par.numel() for par in params_r32)

        def save_r32_weights(path: Path):
            tensors = {}
            for name, ad in adapters_r32.items():
                tensors[f"{name}.a"] = ad.a.detach().cpu().contiguous()
                tensors[f"{name}.b"] = ad.b.detach().cpu().contiguous()
            save_file(tensors, str(path))

        def load_r32_weights(path: Path):
            tensors = load_file(str(path))
            with torch.no_grad():
                for name, ad in adapters_r32.items():
                    ad.a.copy_(tensors[f"{name}.a"])
                    ad.b.copy_(tensors[f"{name}.b"])

        steps_per_epoch = preflight_summary["steps_per_epoch"]
        total_steps = preflight_summary["total_steps"]
        lr_max = float(r32_cfg["learning_rate"])
        lr_min = float(r32_cfg.get("min_learning_rate", lr_max * 0.1))
        warmup_ratio = float(r32_cfg.get("warmup_ratio", 0.05))

        optimizer_r32 = torch.optim.AdamW(params_r32, lr=lr_max, weight_decay=0.0)
        batches_r32 = build_stratified_v2_schedule(
            train_rows, total_steps, CONFIG["batch_size"], CONFIG["seed"]
        )
        micro_bs = CONFIG["micro_batch_size"]
        model.train()

        r32_losses: list[float] = []
        r32_step_times_ms: list[float] = []
        epoch_evaluations: list[dict] = []
        best_val_score = -1.0
        best_epoch = 1
        best_val_records: list[dict] = []
        best_test_records: list[dict] = []

        with (out / "train_log.jsonl").open("w") as stream, (out / "epoch_evals.jsonl").open("w") as eval_stream:
            for step_idx in range(1, total_steps + 1):
                cur_lr = compute_cosine_lr(step_idx, total_steps, lr_max, lr_min, warmup_ratio)
                for pg in optimizer_r32.param_groups:
                    pg["lr"] = cur_lr

                batch = batches_r32[step_idx - 1]
                t_s = time.monotonic()
                optimizer_r32.zero_grad(set_to_none=True)
                n_chunks = math.ceil(len(batch) / micro_bs)
                step_loss_val = 0.0
                for c_idx in range(n_chunks):
                    sub = batch[c_idx * micro_bs : (c_idx + 1) * micro_bs]
                    weight = len(sub) / len(batch)
                    sub_loss = subbatch_losses(model, sub).mean() * weight
                    if not torch.isfinite(sub_loss):
                        raise ValueError(f"Nonfinite loss at step {step_idx}")
                    sub_loss.backward()
                    step_loss_val += float(sub_loss.detach())
                grad_norm = torch.nn.utils.clip_grad_norm_(params_r32, 1.0, error_if_nonfinite=True)
                optimizer_r32.step()
                torch.cuda.synchronize()
                dt_ms = (time.monotonic() - t_s) * 1000.0
                r32_step_times_ms.append(dt_ms)
                r32_losses.append(step_loss_val)
                receipt["train_steps"] = step_idx
                epoch_float = round(step_idx / steps_per_epoch, 3)

                log_row = {
                    "arm": "stage1_v2_r32_448x336_6ep",
                    "step": step_idx,
                    "epoch": epoch_float,
                    "loss": round(step_loss_val, 4),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": round(cur_lr, 7),
                    "step_ms": round(dt_ms, 1),
                    "alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                }
                stream.write(json.dumps(log_row) + "\n")
                stream.flush()

                if step_idx == 1 or step_idx % 25 == 0 or step_idx == total_steps:
                    print(json.dumps(log_row), flush=True)
                    save()
                    trigger_live_dashboard()

                # End-of-epoch checkpointing & full-val / canonical-test evaluation
                if step_idx % steps_per_epoch == 0 or step_idx == total_steps:
                    ep_num = step_idx // steps_per_epoch
                    ckpt_path = out / f"stage1_lora_r32_448x336_epoch{ep_num}.safetensors"
                    save_r32_weights(ckpt_path)

                    val_summary, val_recs = evaluate_policy(
                        model, f"stage1_v2_epoch{ep_num}_val", val_calib_rows
                    )
                    test_summary, test_recs = evaluate_policy(
                        model, f"stage1_v2_epoch{ep_num}_test", test_eval_rows
                    )

                    val_ev = val_summary["evidence_pairs"]
                    val_score = (
                        val_ev["overall_exact_correct"]
                        + 2.0 * val_ev["pair_action_reversal_count"]
                        + 3.0 * val_ev["same_frame_named_flip_count"]
                        + val_summary["os_ost_part3_heldout_seq_1_7"]["strict_class_correct"]
                        + val_summary["roboturk_heldout_unseen_operators"]["strict_class_correct"]
                    )
                    is_best = val_score >= best_val_score
                    if is_best:
                        best_val_score = val_score
                        best_epoch = ep_num
                        best_val_records = val_recs
                        best_test_records = test_recs
                        save_r32_weights(out / "stage1_lora_r32_448x336_best.safetensors")

                    ep_record = {
                        "epoch": ep_num,
                        "step": step_idx,
                        "lr": round(cur_lr, 7),
                        "recent_16_loss": round(sum(r32_losses[-16:]) / min(len(r32_losses), 16), 4),
                        "val_composite_score": val_score,
                        "is_best_val": is_best,
                        "val": val_summary,
                        "test": test_summary,
                    }
                    epoch_evaluations.append(ep_record)
                    eval_stream.write(json.dumps(ep_record) + "\n")
                    eval_stream.flush()

                    clean_test = [{k: v for k, v in r.items() if k != "_h_last"} for r in test_recs]
                    (out / f"epoch{ep_num}_test_generations.json").write_text(
                        json.dumps(clean_test, indent=2, ensure_ascii=False) + "\n"
                    )

                    receipt["epoch_evaluations"] = epoch_evaluations
                    receipt["best_validation_epoch"] = best_epoch
                    save()
                    trigger_live_dashboard()
                    model.train()

        # =========================================================================
        # Reload Best Validation Checkpoint for Holdout + 2x2 Control + Confidence
        # =========================================================================
        load_r32_weights(out / "stage1_lora_r32_448x336_best.safetensors")
        model.eval()

        # 1) 4 Brand-New Held-Out OS Videos (`additional_holdout.jsonl`: 9 triplets + 2 goal_cross scenes)
        add_rows = [json.loads(l) for l in Path(CONFIG["additional_holdout_path"]).read_text().splitlines()]
        add_recs = [generate_and_extract_hidden(model, r) for r in add_rows]
        by_add_pair = defaultdict(dict)
        gc_by_scene = defaultdict(list)
        for r in add_recs:
            if r.get("probe_type") == "new_video_triplet":
                by_add_pair[r["pair_id"]][r["phase"]] = r
            elif r.get("probe_type") == "goal_cross":
                gc_by_scene[r["probe_scene"]].append(r)
        add_summary = {
            "new_video_pre_post_9": sum(v["pre_reveal"]["strict_class_ok"] and v["post_reveal"]["strict_class_ok"] for v in by_add_pair.values()),
            "new_video_named_flip_9": sum(v["post_reveal"]["strict_exact"] and v["post_reveal_masked"]["strict_class_ok"] for v in by_add_pair.values()),
            "goal_cross_both_ok_2": sum(all(x["strict_exact"] for x in grp) for grp in gc_by_scene.values()),
            "goal_cross_details": [
                {"scene": r["probe_scene"], "noun": r["target_noun"], "exp": r["expected_class"], "pred": r["predicted_class"], "exact": r["strict_exact"], "text": r["text"]}
                for r in add_recs if r.get("probe_type") == "goal_cross"
            ],
        }
        (out / "additional_holdout_generations.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "_h_last"} for r in add_recs], indent=2, ensure_ascii=False) + "\n"
        )

        # 2) Full 25-Triplet 2x2 Factorial Jump-Cut vs Evidence Control (stratified by visibility audit)
        vis_map = {}
        with Path(CONFIG["visibility_audit_csv"]).open() as f_csv:
            for vrow in csv.DictReader(f_csv):
                vis_map[vrow["row_id"]] = vrow["visibility"]

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
                        "source_dataset": full["source_dataset"],
                        "source_pool": pool_tag,
                        "visibility": vis,
                        "condition": cond,
                        "goal": full["goal"],
                        "goal_mode": "specific_target",
                        "target_noun": full.get("target_noun"),
                        "expected_action_class": exp_cls,
                        "images": imgs,
                    })
        open_2x2_recs = [generate_and_extract_hidden(model, pr) for pr in open_2x2_probes]

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
        (out / "open_2x2_generations.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "_h_last"} for r in open_2x2_recs], indent=2, ensure_ascii=False) + "\n"
        )

        # 3) Positive-Constrained (softplus) 5-Fold CV Ridge + Temperature-Scaled Confidence Head
        X_cal = torch.stack([r["_h_last"] for r in best_val_records]).float()
        y_cal = torch.tensor([1.0 if r["accepted"] else 0.0 for r in best_val_records]).float()
        lp_cal = torch.tensor([r["mean_token_logprob"] for r in best_val_records]).float()

        X_test = torch.stack([r["_h_last"] for r in best_test_records]).float()
        y_test_list = [1 if r["accepted"] else 0 for r in best_test_records]
        lp_test = torch.tensor([r["mean_token_logprob"] for r in best_test_records]).float()
        test_domains = ["triplet" if r.get("pair_id") else r["source_dataset"] for r in best_test_records]

        conf_summary, conf_state = fit_positive_ridge_temp_confidence(
            X_cal, y_cal, lp_cal, X_test, y_test_list, lp_test, test_domains
        )
        conf_state["image_resolution"] = f"{img_w}x{img_h}"
        conf_state["best_epoch"] = best_epoch
        torch.save(conf_state, out / "stage1_v2_confidence_head_ridge_temp.pt")

        r32_peak_alloc_mib = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        r32_peak_res_mib = round(torch.cuda.max_memory_reserved() / 2**20, 2)

        receipt["stage1_v2_summary"] = {
            "trainable_parameters": n_params_r32,
            "adapter_count": len(adapters_r32),
            "image_resolution": f"{img_w}x{img_h}",
            "epochs": CONFIG["epochs"],
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "mean_step_ms": round(sum(r32_step_times_ms) / max(len(r32_step_times_ms), 1), 1),
            "first_16_loss": round(sum(r32_losses[:16]) / 16, 4),
            "last_16_loss": round(sum(r32_losses[-16:]) / 16, 4),
            "best_validation_epoch": best_epoch,
            "best_epoch_test_evaluation": epoch_evaluations[best_epoch - 1]["test"],
            "additional_holdout": add_summary,
            "open_2x2_control": open_2x2_summary,
            "torch_peak_allocated_mib": r32_peak_alloc_mib,
            "torch_peak_reserved_mib": r32_peak_res_mib,
            "train_and_eval_gpu_hours": round((time.monotonic() - t_train) / 3600, 4),
        }
        receipt["stage1_confidence_readout"] = conf_summary
        receipt.update(
            status="completed",
            torch_peak_allocated_mib=r32_peak_alloc_mib,
            torch_peak_reserved_mib=r32_peak_res_mib,
        )
    except BaseException as error:
        receipt.update(status="failed", error=str(error))
        traceback.print_exc()
        raise
    finally:
        stopped.set()
        thread.join(timeout=3)
        wall = time.monotonic() - started
        receipt.update(
            wall_seconds=round(wall, 2),
            gpu_hours_upper_bound=round(wall / 3600, 4),
            gpu_hours_since_context=round((wall - receipt["cuda_start_s"]) / 3600, 4) if "cuda_start_s" in receipt else None,
        )
        save()
        trigger_live_dashboard()
        print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
