#!/usr/bin/env python3
"""Decisive 2x2 Factorial Control (Jump-Cut vs. Evidence-Removal) + ActiveArena Color-Swap Counterfactual.

Tests whether the VLM tracks SEMANTIC EVIDENCE in causal visual history rather than mere frame jump-cuts:
- Part A (ActiveArena Controlled, 6 held-out episodes x 5 conditions = 30 prefixes, 336x252):
    1. full:               [f1, f2, f3_reveal, f4]           (Continuous + Evidence Present -> Place true color)
    2. jump_keep_evidence: [p1, f3_reveal, p3, f4]           (SAME p3->f4 Jump-Cut + Evidence Preserved -> Place true color)
    3. smooth_no_evidence: [f4, f4, f4, f4]                  (ZERO Jump-Cut + Evidence Removed -> Inspect)
    4. masked:             [p1, p2, p3, f4]                  (Jump-Cut + Evidence Removed -> Inspect)
    5. color_swap:         [f1, f2, f3_other_color, f4]      (Swapped revealing frame -> Place SWAPPED color!)
- Part B (Open-Video 6-Epoch Best Checkpoint, 25 triplets [16 canonical + 9 additional] x 5 conditions = 125 prefixes, 448x336):
    Stratified by `current_frame_audit.csv` into `ambiguous_final_frame` (11 triplets) vs `clear_final_frame` (14 triplets):
    1. full:               [f1, f2, f3, f4]
    2. early_jump_keep:    [p1, f2, f3, f4]
    3. late_jump_keep:     [p1, f3, p3, f4]                  (SAME p3->f4 Jump-Cut as masked, but keeps f3!)
    4. smooth_no_evidence: [f4, f4, f4, f4]                  (ZERO Jump-Cut!)
    5. masked:             [p1, p2, p3, f4]

Zero training, pure greedy inference, hard GPU budget guard (<= 600s, <= 8 GiB VRAM).
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path("/home/ubuntu/ygwang/IP-VLA/asa-iteration/reports/rss2027_autoresearch")
OUT_DIR = ROOT / "results/stage1_closeout_20261121"
AA_RUN = ROOT / "results/evidence_confidence_action_v1/e49634f5-72f2-41cf-9baf-29a0a1fcf9fe"
OPEN_RUN = ROOT / "results/stage1_full_6ep_r32_highres448_v1/725c7958-3dbf-4e3d-9b14-70ce5c9c02f3"
PILOT = Path("/home/ubuntu/ygwang/IP-VLA/rss2027-local-pilot/research/rss2027_local")
sys.path.insert(0, str(PILOT))
import run as pilot_run

COLORS = ("green", "red", "blue", "purple", "orange", "yellow")


def detect_color(text: str) -> str | None:
    tl = text.lower()
    found = [c for c in COLORS if re.search(rf"\b{c}\b", tl)]
    return found[0] if len(found) == 1 else (found[0] if found else None)


def classify_aa(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ("place", "put", "set", "drop", "move the gray block to", "move the block to")):
        return "place"
    if any(w in tl for w in ("rotate", "turn", "flip", "inspect", "check", "look", "view", "reveal", "examine")):
        return "inspect"
    return pilot_run.classify(text)


def load_visibility_audit() -> dict[str, str]:
    vis = {}
    csv_path = OUT_DIR / "current_frame_audit.csv"
    if csv_path.is_file():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                vis[row["row_id"]] = row["visibility"]
    return vis


def build_probes() -> tuple[list[dict], list[dict]]:
    # Part A: ActiveArena 6 held-out episodes
    aa_rows = [json.loads(l) for l in (AA_RUN / "evidence_suite.jsonl").read_text().splitlines()]
    aa_post = [r for r in aa_rows if r["domain"] == "activearena" and r["split"] == "test" and r["evidence_present"]]
    assert len(aa_post) == 6, f"Expected 6 ActiveArena test episodes, got {len(aa_post)}"

    aa_probes = []
    for idx, r in enumerate(aa_post):
        other = aa_post[(idx + 1) % len(aa_post)]
        f = r["images"]          # [f1, f2, f3_reveal, f4]
        p = r["masked_images"]   # [p1, p2, p3, f4]
        true_color = r["hidden_color"]
        swap_color = other["hidden_color"]
        conditions = [
            ("full",               [f[0], f[1], f[2], f[3]],              "place",   true_color),
            ("jump_keep_evidence", [p[0], f[2], p[2], f[3]],              "place",   true_color),
            ("smooth_no_evidence", [f[3], f[3], f[3], f[3]],              "inspect", None),
            ("masked",             [p[0], p[1], p[2], f[3]],              "inspect", None),
            ("color_swap",         [f[0], f[1], other["images"][2], f[3]], "place",  swap_color),
        ]
        for cond, imgs, exp_cls, exp_color in conditions:
            aa_probes.append({
                "id": f"{r['id']}__{cond}",
                "episode_id": r["id"],
                "pair_id": r["pair_id"],
                "condition": cond,
                "goal": r["goal"],
                "true_color": true_color,
                "expected_class": exp_cls,
                "expected_color": exp_color,
                "images": imgs,
            })

    # Part B: Open-Video 16 canonical + 9 additional holdout triplets
    vis_map = load_visibility_audit()
    open_probes = []

    def add_open_triplets(rows: list[dict], source_tag: str) -> None:
        by_pair = defaultdict(dict)
        for r in rows:
            if r.get("pair_id") and r.get("phase") in ("pre_reveal", "post_reveal", "post_reveal_masked"):
                by_pair[r["pair_id"]][r["phase"]] = r
        for pid, grp in sorted(by_pair.items()):
            if not {"pre_reveal", "post_reveal", "post_reveal_masked"} <= grp.keys():
                continue
            pre = grp["pre_reveal"]
            full = grp["post_reveal"]
            f = full["images"]
            p = pre["images"]
            vis = vis_map.get(full["id"], "unknown")
            conditions = [
                ("full",               [f[0], f[1], f[2], f[3]], "acquire"),
                ("early_jump_keep",    [p[0], f[1], f[2], f[3]], "acquire"),
                ("late_jump_keep",     [p[0], f[2], p[2], f[3]], "acquire"),
                ("smooth_no_evidence", [f[3], f[3], f[3], f[3]], "search"),
                ("masked",             [p[1], p[2], p[3], f[3]], "search"),
            ]
            for cond, imgs, exp_cls in conditions:
                open_probes.append({
                    "id": f"{full['id']}__{cond}",
                    "row_id": full["id"],
                    "pair_id": pid,
                    "source_pool": source_tag,
                    "source_dataset": full["source_dataset"],
                    "video": full["source_video_sha256"],
                    "visibility": vis,
                    "condition": cond,
                    "goal": full["goal"],
                    "goal_mode": "specific_target",
                    "target_noun": full.get("target_noun"),
                    "expected_action_class": exp_cls,
                    "images": imgs,
                })

    # Load 16 canonical triplets from parent 144-row frozen test pool
    suite_rows = [json.loads(l) for l in (OPEN_RUN / "stage1_formal_suite.jsonl").read_text().splitlines()]
    row_by_id = {r["id"]: r for r in suite_rows}
    parent_gen = ROOT / "results/stage1_bs16_nogc_rank32_highres448_v1/ea40ef8b-e862-4827-b435-5d0334d17559/lora_r32_448x336_step378_test_generations.json"
    canon_ids = [g["row_id"] for g in json.loads(parent_gen.read_text())]
    add_open_triplets([row_by_id[rid] for rid in canon_ids if rid in row_by_id], "canonical_16")

    # Load 9 additional holdout triplets
    add_rows = [json.loads(l) for l in (OUT_DIR / "additional_holdout.jsonl").read_text().splitlines()]
    add_open_triplets([r for r in add_rows if r.get("probe_type") == "new_video_triplet"], "additional_9")

    return aa_probes, open_probes


def main() -> None:
    aa_probes, open_probes = build_probes()
    preflight = {
        "aa_probes": len(aa_probes),
        "open_probes": len(open_probes),
        "open_triplets_canonical": len({p["pair_id"] for p in open_probes if p["source_pool"] == "canonical_16"}),
        "open_triplets_additional": len({p["pair_id"] for p in open_probes if p["source_pool"] == "additional_9"}),
        "open_ambiguous_triplets": len({p["pair_id"] for p in open_probes if p["visibility"] != "clear"}),
        "open_clear_triplets": len({p["pair_id"] for p in open_probes if p["visibility"] == "clear"}),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if sys.argv[1:] == ["--cpu-preflight"]:
        for p in aa_probes + open_probes:
            assert len(p["images"]) == 4
            for im in p["images"]:
                assert Path(im["path"]).is_file(), f"Missing frame: {im['path']}"
        print("CPU preflight passed: all frames verified.", flush=True)
        return

    started = time.monotonic()
    receipt = {"status": "running", "peak_process_mib": 0, "preflight": preflight}
    stopping = threading.Event()

    def watch():
        while not stopping.wait(0.5):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    text=True, timeout=2,
                )
                for line in out.splitlines():
                    pid, mem = (s.strip() for s in line.split(","))
                    if pid == str(os.getpid()):
                        receipt["peak_process_mib"] = max(receipt["peak_process_mib"], int(mem))
            except Exception:
                pass
            if time.monotonic() - started >= 600 or receipt["peak_process_mib"] > 9216:
                receipt["status"] = "stopped_by_budget"
                (OUT_DIR / "factorial_2x2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                os._exit(124)

    t = threading.Thread(target=watch, daemon=True)
    t.start()

    try:
        import torch
        from torch import nn
        from PIL import Image
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        assert free > 8192 * 2**20, f"Insufficient free VRAM: {free / 2**20:.0f} MiB"
        torch.cuda.set_per_process_memory_fraction(8192 * 2**20 / total)

        model_path = "/home/ubuntu/ygwang/IP-VLA/asa-iteration/checkpoints/Qwen3-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        class Adapter(nn.Module):
            def __init__(self, original, a, b, scale):
                super().__init__()
                self.base = original
                self.register_buffer("a", a.to(original.weight))
                self.register_buffer("b", b.to(original.weight))
                self.scale = scale

            def forward(self, x):
                return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.a), self.b) * self.scale

        def load_fresh_model(ckpt_path: Path | None, scale: float = 2.0):
            m = AutoModelForImageTextToText.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map={"": 0},
            )
            m.eval().requires_grad_(False)
            if ckpt_path is not None:
                weights = load_file(str(ckpt_path))
                names = [k[:-2] for k in weights if k.endswith(".a")]
                for n in names:
                    parent, _, leaf = n.rpartition(".")
                    w = Adapter(m.get_submodule(n), weights[n + ".a"], weights[n + ".b"], scale)
                    setattr(m.get_submodule(parent), leaf, w)
            return m

        def infer_one(model, imgs_meta: list[dict], goal: str, w: int, h: int) -> str:
            pil_imgs = []
            for im in imgs_meta:
                with Image.open(im["path"]) as img:
                    pil_imgs.append(img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
            prompt = (
                f"Goal: {goal}\n"
                "The views are ordered oldest to newest. "
                "State one feasible immediate next task as a short imperative sentence."
            )
            msg = [{"role": "user", "content": [*[{"type": "image"} for _ in pil_imgs], {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inps = processor(images=pil_imgs, text=[text], return_tensors="pt")
            inps = {k: v.to("cuda", dtype=model.dtype if v.is_floating_point() else v.dtype) for k, v in inps.items()}
            with torch.inference_mode():
                ids = model.generate(**inps, max_new_tokens=28, do_sample=False)
                return processor.tokenizer.decode(ids[0, inps["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # --- Part A: ActiveArena Controlled 2x2 + Color Swap (336x252) ---
        aa_model = load_fresh_model(AA_RUN / "evidence_paired_lora.safetensors", scale=2.0)
        aa_results = []
        for p in aa_probes:
            sent = infer_one(aa_model, p["images"], p["goal"], 336, 252)
            pred_cls = classify_aa(sent)
            pred_color = detect_color(sent)
            cls_ok = pred_cls == p["expected_class"]
            color_ok = (pred_color == p["expected_color"]) if p["expected_color"] else True
            aa_results.append({
                "id": p["id"], "episode_id": p["episode_id"], "condition": p["condition"],
                "true_color": p["true_color"], "expected_class": p["expected_class"],
                "expected_color": p["expected_color"], "text": sent,
                "predicted_class": pred_cls, "predicted_color": pred_color,
                "exact_ok": cls_ok and color_ok, "class_ok": cls_ok, "color_ok": color_ok,
            })
        del aa_model
        torch.cuda.empty_cache()

        # --- Part B: Open-Video 6-Epoch Best Checkpoint 2x2 (448x336) ---
        open_model = load_fresh_model(OPEN_RUN / "stage1_lora_r32_448x336_best.safetensors", scale=2.0)
        open_results = []
        for p in open_probes:
            sent = infer_one(open_model, p["images"], p["goal"], 448, 336)
            adj = pilot_run.adjudicate_generation(p, sent)
            open_results.append({
                "id": p["id"], "pair_id": p["pair_id"], "source_pool": p["source_pool"],
                "source_dataset": p["source_dataset"], "video": p["video"],
                "visibility": p["visibility"], "condition": p["condition"],
                "target_noun": p["target_noun"], **adj,
            })
        del open_model
        torch.cuda.empty_cache()

        # Summarize Part A
        aa_summary = {}
        for cond in ("full", "jump_keep_evidence", "smooth_no_evidence", "masked", "color_swap"):
            sub = [r for r in aa_results if r["condition"] == cond]
            aa_summary[cond] = {
                "n": len(sub),
                "exact_ok": sum(r["exact_ok"] for r in sub),
                "class_ok": sum(r["class_ok"] for r in sub),
                "place_count": sum(r["predicted_class"] == "place" for r in sub),
                "inspect_count": sum(r["predicted_class"] == "inspect" for r in sub),
            }

        # Summarize Part B (Overall + Stratified by Final-Frame Visibility)
        def summarize_open(sub_rows: list[dict]) -> dict:
            out_s = {}
            for cond in ("full", "early_jump_keep", "late_jump_keep", "smooth_no_evidence", "masked"):
                c_rows = [r for r in sub_rows if r["condition"] == cond]
                out_s[cond] = {
                    "n": len(c_rows),
                    "acquire_count": sum(r["predicted_class"] == "acquire" for r in c_rows),
                    "acquire_named_count": sum(r["predicted_class"] == "acquire" and r["noun_ok"] for r in c_rows),
                    "search_count": sum(r["predicted_class"] == "search" for r in c_rows),
                }
            return out_s

        open_summary = {
            "all_25": summarize_open(open_results),
            "canonical_16": summarize_open([r for r in open_results if r["source_pool"] == "canonical_16"]),
            "additional_9": summarize_open([r for r in open_results if r["source_pool"] == "additional_9"]),
            "ambiguous_final_frame_11": summarize_open([r for r in open_results if r["visibility"] != "clear"]),
            "clear_final_frame_14": summarize_open([r for r in open_results if r["visibility"] == "clear"]),
        }

        payload = {
            "aa_summary": aa_summary,
            "open_summary": open_summary,
            "aa_results": aa_results,
            "open_results": open_results,
        }
        (OUT_DIR / "factorial_2x2_control_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )

        # Write crisp human-readable report
        md = [
            "# 2×2 跳切 vs. 证据移除反事实对照实验报告（含 ActiveArena 颜色替换检验）",
            "",
            "本实验在**全局目标 $G$ 与末帧 $f_4$ 字节级完全不变**的前提下，拆解“画面时间跳切（Jump-Cut）”与“关键视觉证据有无（Semantic Evidence）”两个正交因子。",
            "",
            "## 一、 ActiveArena 受控 2×2 因子对照 + 跨局颜色替换反事实（6 局留出，末帧 0 像素可见贴纸）",
            "",
            "| 对照条件（相同目标 $G$ + 相同末帧 $f_4$） | 历史是否有跳切？ | 关键颜色证据是否在历史中？ | 期望行为 | 精确命中 (`/6`) | 输出 `place` (`/6`) | 输出 `inspect` (`/6`) |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| `full`（连续完整历史 `[f1, f2, f3, f4]`） | 否 | **有（真色）** | `place` + 真色 | **`{aa_summary['full']['exact_ok']}/6`** | `{aa_summary['full']['place_count']}/6` | `{aa_summary['full']['inspect_count']}/6` |",
            f"| **`jump_keep_evidence`（同款 `p3->f4` 大跳切 + 保留 `f3`）** | **是（~8.5s 跳切）** | **有（真色）** | `place` + 真色 | **`{aa_summary['jump_keep_evidence']['exact_ok']}/6`** | `{aa_summary['jump_keep_evidence']['place_count']}/6` | `{aa_summary['jump_keep_evidence']['inspect_count']}/6` |",
            f"| **`smooth_no_evidence`（零跳切 `[f4, f4, f4, f4]`）** | **否（0.0s 零跳切）** | **无** | `inspect` | **`{aa_summary['smooth_no_evidence']['exact_ok']}/6`** | `{aa_summary['smooth_no_evidence']['place_count']}/6` | `{aa_summary['smooth_no_evidence']['inspect_count']}/6` |",
            f"| `masked`（跳切 + 移除证据 `[p1, p2, p3, f4]`） | 是（~8.5s 跳切） | 无 | `inspect` | **`{aa_summary['masked']['exact_ok']}/6`** | `{aa_summary['masked']['place_count']}/6` | `{aa_summary['masked']['inspect_count']}/6` |",
            f"| **`color_swap`（将第 3 帧替换为另一局的异色贴纸帧）** | 轻微 | **有（异色）** | `place` + **异色** | **`{aa_summary['color_swap']['exact_ok']}/6`** | `{aa_summary['color_swap']['place_count']}/6` | `{aa_summary['color_swap']['inspect_count']}/6` |",
            "",
            "## 二、 开放视频 6-Epoch 验证优选模型（`725c7958`）2×2 对照（25 组完整三元组，按末帧独立审阅分层）",
            "",
            "| 切片 / 条件 | `full` 拿取(命名) | `early_jump_keep` 拿取(命名) / 退回搜索 | `late_jump_keep` (`p3->f4` 跳切+留 `f3`) 拿取(命名) / 退回搜索 | `smooth_no_evidence` (`[f4]*4` 零跳切) 退回搜索 | `masked` (`p1..p3+f4`) 退回搜索 |",
            "| --- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for label, key in (
            ("**全部 25 组**（16 固定 + 9 新增）", "all_25"),
            ("末帧**含混/不可定位**（11 组：真需历史证据）", "ambiguous_final_frame_11"),
            ("末帧**清晰可定位**（14 组：末帧已含目标）", "clear_final_frame_14"),
            ("固定 16 组（4 段录像）", "canonical_16"),
            ("新增 9 组（4 段新 OS 录像）", "additional_9"),
        ):
            s = open_summary[key]
            n = s["full"]["n"]
            md.append(
                f"| {label} | `{s['full']['acquire_count']}/{n}` (`{s['full']['acquire_named_count']}/{n}`) | "
                f"`{s['early_jump_keep']['acquire_count']}/{n}` (`{s['early_jump_keep']['acquire_named_count']}/{n}`) / `{s['early_jump_keep']['search_count']}/{n}` | "
                f"`{s['late_jump_keep']['acquire_count']}/{n}` (`{s['late_jump_keep']['acquire_named_count']}/{n}`) / `{s['late_jump_keep']['search_count']}/{n}` | "
                f"`{s['smooth_no_evidence']['search_count']}/{n}` | `{s['masked']['search_count']}/{n}` |"
            )
        (OUT_DIR / "FACTORIAL_2X2_REPORT.md").write_text("\n".join(md) + "\n")
        receipt["status"] = "completed"
        receipt["wall_seconds"] = round(time.monotonic() - started, 2)
        receipt["gpu_hours_upper_bound"] = round(receipt["wall_seconds"] / 3600, 4)
        (OUT_DIR / "factorial_2x2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"receipt": receipt, "aa_summary": aa_summary, "open_summary": open_summary}, indent=2), flush=True)
    finally:
        stopping.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()
