"""Summarize the completed, paired native/best multi-task scene experiment."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage1_generalization_s1217_gpu0"
OUTPUT = ROOT / "docs/assets/stage1_generalization"
CONDITIONS = ("normal", "camera", "lighting", "combined")
POLICIES = ("native", "best")


def dump_csv(name, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    spec = json.loads((ROOT / "experiments/diagnostic_cases/stage1_generalization_s1217/manifest.json").read_text())
    audit = json.loads((RUN / "scene_audit/report.json").read_text())
    assert audit["passed"] and audit["cases"] == 96
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene_metrics = []
    for case in spec["cases"]:
        baseline_path = RUN / "scene_audit" / (case["layout"]+"_normal") / "observation_0000.npz"
        with np.load(baseline_path, allow_pickle=False) as baseline, np.load(
            RUN / "scene_audit" / case["case_id"] / "observation_0000.npz", allow_pickle=False
        ) as frame:
            scene_metrics.append({"case_id": case["case_id"], "our_training_split": case["our_training_split"],
                                  "yaw_degrees": case["yaw_degrees"], "pitch_degrees": case["pitch_degrees"],
                                  "light_scale": case["light_scale"], "manual_seed": case["manual_seed"],
                                  "agent_rgb_mean": float(frame["agent_rgb"].mean()),
                                  "agent_rgb_mae_vs_normal": float(np.abs(frame["agent_rgb"].astype(float)-baseline["agent_rgb"]).mean()),
                                  "wrist_rgb_mae_vs_normal": float(np.abs(frame["wrist_rgb"].astype(float)-baseline["wrist_rgb"]).mean()),
                                  "agent_valid_depth_fraction": float((np.isfinite(frame["agent_depth"]) & (frame["agent_depth"] > 0)).mean())})
    dump_csv("scene_conditions.csv", scene_metrics)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    cards = []
    for task in spec["tasks"]:
        canvas = Image.new("RGB", (1024, 576), "white")
        draw = ImageDraw.Draw(canvas)
        for i, condition in enumerate(CONDITIONS):
            case_id = f"S_{task['task_index']:02d}_30_{condition}"
            x, y = i % 2 * 512, i // 2 * 288
            draw.text((x+8, y+5), condition, fill="black", font=font)
            with np.load(RUN / "scene_audit" / case_id / "observation_0000.npz", allow_pickle=False) as frame:
                for j, camera in enumerate(("agent", "wrist")):
                    canvas.paste(Image.fromarray(frame[camera+"_rgb"]), (x+256*j, y+30))
        filename = f"task_{task['task_index']:02d}.jpg"
        canvas.save(OUTPUT / filename, quality=90)
        cards.append(f"<section><h2>{task['our_training_split']} · {html.escape(task['source'])}</h2>"
                     f"<img src='{filename}' alt='四种场景条件，两路相机'></section>")
    report_path = RUN / "evaluation/report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    complete = bool(report and report["complete"])
    rows, scores, effects, demos = [], [], [], []
    if complete:
        assert report["modes"] == list(POLICIES) and len(report["episodes"]) == 192
        cases = {c["case_id"]: c for c in spec["cases"]}
        outcomes = {}
        for row in report["episodes"]:
            case = cases[row["case_id"]]
            outcomes[row["policy"], row["case_id"]] = row
            rows.append({"policy": row["policy"], "case_id": row["case_id"],
                         "task_index": case["task_index"], "source": case["source"],
                         "our_training_split": case["our_training_split"],
                         "condition": case["condition"], "episode": case["episode"],
                         "manual_seed": row["manual_seed"], "success": row["success"],
                         "control_steps": row["policy_control_steps"],
                         "max_steps": case["max_policy_steps"],
                         "penalized_horizon_fraction": (row["policy_control_steps"]/case["max_policy_steps"] if row["success"] else 1.),
                         "wall_seconds": row["wall_seconds"],
                         "peak_allocated_gib": row["peak_allocated_gib"]})
        assert len(outcomes) == 192
        for split in ("train", "validation", "test"):
            task_ids = [t["task_index"] for t in spec["tasks"] if t["our_training_split"] == split]
            assert len(task_ids) == 4
            for condition in CONDITIONS:
                for policy in POLICIES:
                    subset = [r for r in rows if r["policy"] == policy and r["condition"] == condition and r["our_training_split"] == split]
                    assert len(subset) == 8
                    scores.append({"our_training_split": split, "condition": condition, "policy": policy,
                                   "successes": sum(r["success"] for r in subset), "episodes": len(subset),
                                   "tasks": 4, "layouts": 8,
                                   "mean_penalized_horizon_fraction": float(np.mean([r["penalized_horizon_fraction"] for r in subset]))})
                values = np.asarray([[int(outcomes["best", f"S_{t:02d}_{reset:02d}_{condition}"]["success"])
                                      - int(outcomes["native", f"S_{t:02d}_{reset:02d}_{condition}"]["success"])
                                      for reset in (30, 40)] for t in task_ids], dtype=float)
                rng = np.random.default_rng(1317)
                task_draws = rng.integers(0, 4, (10000, 4, 1))
                reset_draws = rng.integers(0, 2, (10000, 4, 2))
                low, high = np.quantile(values[task_draws, reset_draws].mean(axis=(1, 2)), [.025, .975])
                effects.append({"our_training_split": split, "condition": condition,
                                "best_minus_native": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
                                "manual_seed": 1317, "bootstrap": "10000 hierarchical task-then-reset paired draws"})
        dump_csv("episodes.csv", rows); dump_csv("scores.csv", scores); dump_csv("paired_effects.csv", effects)
        from build_stage1_pair_assets import demo

        # Select by a declared outcome stratum, never by visual appeal.
        # Include shared failures and native-only wins as well as best-only wins.
        rng = np.random.default_rng(1317)
        for stratum, native_success, best_success in (
            ("both_success", True, True), ("native_only", True, False),
            ("best_only", False, True), ("both_failure", False, False),
        ):
            candidates = sorted(c["case_id"] for c in spec["cases"]
                                if c["our_training_split"] == "test"
                                and outcomes["native", c["case_id"]]["success"] == native_success
                                and outcomes["best", c["case_id"]]["success"] == best_success)
            if not candidates:
                demos.append({"outcome_stratum": stratum, "case_id": "", "file": "", "status": "no test case in this stratum"})
                continue
            case_id = str(rng.choice(candidates))
            filename = demo(case_id, [outcomes[p, case_id] for p in POLICIES], run=RUN / "evaluation", output=OUTPUT)
            demos.append({"outcome_stratum": stratum, "case_id": case_id, "file": filename, "status": "generated; manual seed 1317"})
        dump_csv("demos.csv", demos)
    evidence = {"complete": complete, "planned_episodes": 192, "completed_episodes": len(report["episodes"]) if report else 0,
                "scene_audit_passed": True, "tasks": spec["tasks"], "scores": scores, "effects": effects, "demos": demos,
                "limits": spec["limits"], "report": str(report_path.relative_to(ROOT))}
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    table = "".join(f"<tr><td>{r['our_training_split']}</td><td>{r['condition']}</td><td>{r['policy']}</td><td>{r['successes']}/{r['episodes']}</td></tr>" for r in scores)
    videos = "".join(f"<h3>{d['outcome_stratum']} · {d['case_id']}</h3><video controls preload='metadata' width='100%' src='{d['file']}'></video>"
                     for d in demos if d["file"])
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>阶段一多任务场景泛化</title>
<style>body{{max-width:1050px;margin:32px auto;padding:16px;font:17px/1.7 sans-serif;color:#183047}}img{{width:100%}}h2{{font-size:18px;overflow-wrap:anywhere}}td,th{{padding:8px;border:1px solid #ccd7df}}table{{border-collapse:collapse}}section{{margin:32px 0}}</style>
<h1>12 个任务、24 个布局、96 个场景条件</h1><p>只比较 MolmoAct2 原生和阶段一 best；计划共 192 次 rollout。模型评估完成：{complete}。本页在审计后及评估结束后更新，中途实时进度见运行目录 report.json。</p>
<p>camera：世界坐标中偏航 ±20°、俯仰 ∓10°；lighting：环境灯光乘 0.55；combined：二者叠加。物理状态和任务语言保持一致，RGB、深度及相机标定由模拟器生成。</p>
<p>train 的任务及初态已用于我们训练；validation 参与 best 选择；test 的 4 个任务未参与我们训练或 best 选择。原生预训练可能见过所有 LIBERO 任务，因此不能称为对底座全新。每任务仅 2 个布局，是第一轮广度验证。</p>
<table><tr><th>划分</th><th>场景条件</th><th>模型</th><th>任务成功</th></tr>{table}</table>
<p><a href='evidence.json'>证据与状态</a> · <a href='scene_conditions.csv'>96 个场景的实际图像变化与有效深度</a> · <a href='../stage1_native_best/index.html'>已有几何/历史结果及 demo</a></p>
{videos}
{''.join(cards)}<p>所有图片来自实际环境。这里扩展的是任务、已有物体和观察条件，不是新生成的物体类别。</p></html>""")
    if complete:
        marker = "阶段一多任务场景比较完成（manual seed 1217）"
        log = ROOT / "docs/experiment_log.md"
        if marker not in log.read_text():
            lines = [f"\n\n#### {marker}\n\n", "两模型共 192 次 rollout，12 个任务、24 个物理布局。原始结果及分划分表见 `docs/assets/stage1_generalization/`。各条件分母均为 8；顺序为原场景、相机变化、光照变化、组合变化。\n\n"]
            for split in ("train", "validation", "test"):
                for policy in POLICIES:
                    values = [next(r["successes"] for r in scores if r["our_training_split"] == split and r["policy"] == policy and r["condition"] == c) for c in CONDITIONS]
                    lines.append(f"- {split} / {policy}：{values} / 每条件 8。\n")
            lines.append("\n这是完整系统比较；单任务只有 2 个布局，不独立归因于几何或历史，也不证明新物体类别泛化。\n")
            with log.open("a") as stream:
                stream.write("".join(lines))
    print(json.dumps({"complete": complete, "output": str(OUTPUT.relative_to(ROOT)), "scores": scores}))


if __name__ == "__main__":
    main()
