"""Build native-versus-best tables, trajectory diagnostics and synchronized demos.

Reads completed policy logs only; run using .venv-sim/bin/python. No GPU use.
"""

import csv
import html
import json
import os
from itertools import product
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = ROOT / "docs/assets/stage1_native_best"
G_RUN = ROOT / "runs/stage1_geometry_test_s217"
H_RUN = ROOT / "runs/stage1_native_best_history_s217_gpu0/evaluation"
POLICIES = ("native", "best")
LABELS = {"native": "MolmoAct2 native", "best": "Stage 1 best (1500)",
          "stage2_best": "Stage 2 best (1500)"}
ANALYSIS_SEED = 1317


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    if not rows:
        return
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect(values, layouts):
    # Resample physical layouts, retaining geometry task strata.
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(ANALYSIS_SEED)
    strata = {}
    for i, layout in enumerate(layouts):
        strata.setdefault(layout.rsplit("_", 1)[0], []).append(i)
    draws = np.concatenate(
        [rng.choice(indices, (10000, len(indices))) for indices in strata.values()],
        axis=1,
    )
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    nonzero = values[values != 0]
    observed = abs(float(nonzero.sum()))
    # Small-sample check: a bootstrap interval alone can overstate evidence.
    exact_p = sum(abs(float(np.dot(signs, nonzero))) >= observed-1e-12
                  for signs in product((-1, 1), repeat=len(nonzero))) / 2**len(nonzero)
    return {"difference": float(values.mean()), "ci_low": float(low),
            "ci_high": float(high), "independent_layouts": len(values),
            "paired_sign_flip_p_two_sided": exact_p}


def directory(run, row):
    return run / row["policy"] / row["case_id"] / row["episode"]


def trajectory(run, row):
    records = [json.loads(line) for line in (directory(run, row) / "rollout.jsonl").read_text().splitlines()]
    observations = [r["observation"] for r in records if "observation" in r]
    actions = [r for r in records if "requested_actions" in r]
    states = []
    for packet in observations:
        with np.load(ROOT / packet["path"], allow_pickle=False) as frame:
            states.append(frame["states"][:3])
    applied = np.asarray([a["value"] for p in observations for a in p["applied_actions"]])
    return observations, actions, np.asarray(states), applied


def demo(case_id, pair, *, run=G_RUN, output=OUTPUT, runs=None):
    """Every saved frame spans five 20 Hz steps; freeze a finished policy."""
    traces = [trajectory(root, r)[0] for root,r in zip(runs or [run]*len(pair),pair,strict=True)]
    final_step = max(t[-1]["step"] for t in traces)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    dest = output / f"{case_id}.mp4"
    with imageio.get_writer(dest, fps=4, codec="libx264", quality=7,
                            ffmpeg_params=["-threads", "2"]) as writer:
        for step in range(0, final_step + 1, 5):
            canvas = Image.new("RGB", (512*len(pair), 384), "#101822")
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), f"{case_id} | simulation step {step} | {step/20:.2f} s", font=font)
            for i, (row, packets) in enumerate(zip(pair, traces)):
                packet = max((p for p in packets if p["step"] <= step), key=lambda p: p["step"])
                with np.load(ROOT / packet["path"], allow_pickle=False) as frame:
                    for j, camera in enumerate(("agent", "wrist")):
                        canvas.paste(Image.fromarray(frame[f"{camera}_rgb"]), (i*512+j*256, 72))
                stopped = step >= packets[-1]["step"]
                status = ("SUCCESS" if row["success"] else "TIMEOUT") if stopped else "executing"
                draw.text((i*512+12, 42), f"{LABELS[row['policy']]} | {status}", font=font)
                draw.text((i*512+12, 340), "agent view          wrist view" + (" | frozen" if stopped else ""), font=font)
            writer.append_data(np.asarray(canvas))
    return dest.name


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    g_report = read(G_RUN / "report.json")
    assert g_report["complete"]
    g_rows = [r for r in g_report["episodes"] if r["policy"] in POLICIES]
    assert len(g_rows) == 80
    raw, scores, effects, demos = [], [], [], []
    cache = {}
    for row in g_rows:
        _observations, actions, xyz, applied = trajectory(G_RUN, row)
        cache[row["policy"], row["case_id"]] = row
        raw.append({"policy": row["policy"], "case_id": row["case_id"], "layout": row["layout"],
                    "task": "drawer" if row["layout"].startswith("G_0") else "bowl_on_plate",
                    "condition": row["condition"], "success": row["success"],
                    "control_steps": row["policy_control_steps"],
                    "penalized_steps": row["policy_control_steps"] if row["success"] else 300,
                    "sampled_eef_path_m_lower_bound": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                    "translation_control_saturation_fraction": float((np.abs(applied[:, :3]) >= 0.999).mean()),
                    "policy_latency_median_s": float(np.median([a["policy_seconds"] for a in actions])),
                    "peak_allocated_gib": row["peak_allocated_gib"],
                    "manual_seed": row["manual_seed"]})
    for condition in ("normal", "shifted"):
        for policy in POLICIES:
            rows = [r for r in raw if r["policy"] == policy and r["condition"] == condition]
            scores.append({"suite": "camera", "condition": condition, "policy": policy,
                           "correct": sum(r["success"] for r in rows), "episodes": len(rows),
                           "independent_layouts": len(rows),
                           "mean_penalized_steps": float(np.mean([r["penalized_steps"] for r in rows]))})
        layouts = sorted({r["layout"] for r in g_rows})
        delta = [int(cache["best", f"{layout}_{condition}"]["success"])
                 - int(cache["native", f"{layout}_{condition}"]["success"]) for layout in layouts]
        effects.append({"suite": "camera", "condition": condition, **effect(delta, layouts)})
    # Show every discordant outcome and one deterministically selected joint success.
    case_ids = sorted({r["case_id"] for r in g_rows})
    selected = [c for c in case_ids if cache["native", c]["success"] != cache["best", c]["success"]]
    joint = [c for c in case_ids if all(cache[p, c]["success"] for p in POLICIES)]
    selected.append(str(np.random.default_rng(ANALYSIS_SEED).choice(joint)))
    for case_id in selected:
        pair = [cache[p, case_id] for p in POLICIES]
        path = demo(case_id, pair)
        demos.append({"case_id": case_id, "native_success": pair[0]["success"],
                      "best_success": pair[1]["success"], "file": path,
                      "selection": "all discordant cases; one seeded joint success"})
    write_csv("geometry_episodes.csv", raw)
    by_task = []
    for task in ("drawer", "bowl_on_plate"):
        for condition in ("normal", "shifted"):
            for policy in POLICIES:
                subset = [r for r in raw if r["task"] == task and r["condition"] == condition and r["policy"] == policy]
                by_task.append({"task": task, "condition": condition, "policy": policy,
                                "successes": sum(r["success"] for r in subset), "episodes": len(subset)})
    write_csv("geometry_per_task.csv", by_task)
    write_csv("demos.csv", demos)

    h_complete = (H_RUN / "report.json").exists() and read(H_RUN / "report.json")["complete"]
    h_rows, h_raw = [], []
    if h_complete:
        h_report = read(H_RUN / "report.json")
        assert h_report["modes"] == list(POLICIES)
        h_rows = h_report["episodes"]
        assert len(h_rows) == 48
        h_cache = {}
        for row in h_rows:
            assert row["policy_calls"] == 1 and row["policy_control_steps"] == 5
            fc = row["first_choice"]
            _, actions, _, _ = trajectory(H_RUN, row)
            h_cache[row["policy"], row["layout"], row["condition"], row["target_index"]] = (row, actions[0])
            h_raw.append({"policy": row["policy"], "case_id": row["case_id"], "layout": row["layout"],
                          "condition": row["condition"], "target_index": row["target_index"],
                          "correct": fc["correct"], "chosen_index": fc["chosen_index"],
                          "axis_displacement_m": fc["axis_displacement_m"],
                          "target_aligned_displacement_m": fc["axis_displacement_m"] * (2*row["target_index"]-1),
                          "policy_input_steps": str(actions[0]["policy_input_observation_steps"]),
                          "manual_seed": row["manual_seed"]})
        layouts = sorted({r["layout"] for r in h_rows})
        dependence = []
        for policy in POLICIES:
            for layout in layouts:
                actions = [np.asarray(h_cache[policy, layout, "hidden", t][1]["requested_actions"]) for t in (0, 1)]
                delta = float(np.max(np.abs(actions[0]-actions[1])))
                if policy == "native" and delta != 0:
                    raise ValueError("native hidden actions differ despite identical inputs and noise")
                dependence.append({"policy": policy, "layout": layout, "hidden_world_action_max_abs_difference": delta})
        write_csv("history_action_dependence.csv", dependence)
        paired = {}
        for condition in ("visible", "hidden"):
            for policy in POLICIES:
                rows = [r for r in h_raw if r["policy"] == policy and r["condition"] == condition]
                scores.append({"suite": "history_direction", "condition": condition, "policy": policy,
                               "correct": sum(r["correct"] for r in rows), "episodes": len(rows),
                               "independent_layouts": len(layouts), "mean_penalized_steps": "not applicable"})
            paired[condition] = [
                np.mean([int(h_cache["best", l, condition, t][0]["first_choice"]["correct"])
                         - int(h_cache["native", l, condition, t][0]["first_choice"]["correct"]) for t in (0, 1)])
                for l in layouts
            ]
            effects.append({"suite": "history_direction", "condition": condition, **effect(paired[condition], layouts)})
        effects.append({"suite": "history_direction", "condition": "hidden_minus_visible_gain",
                        **effect(np.asarray(paired["hidden"])-paired["visible"], layouts)})
        write_csv("history_episodes.csv", h_raw)
        sensitivity = []
        for threshold in (0.001, 0.002, 0.005):
            for condition in ("visible", "hidden"):
                for policy in POLICIES:
                    rows = [r for r in h_raw if r["policy"] == policy and r["condition"] == condition]
                    sensitivity.append({"threshold_m": threshold, "primary_threshold": threshold == .002,
                                        "policy": policy, "condition": condition, "episodes": len(rows),
                                        "correct": sum(r["target_aligned_displacement_m"] > threshold for r in rows),
                                        "scope": "post-hoc sensitivity; primary threshold stays 2 mm"})
        write_csv("history_threshold_sensitivity.csv", sensitivity)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, condition in zip(axes, ("visible", "hidden")):
            for policy, marker in zip(POLICIES, ("o", "s")):
                rows = sorted([r for r in h_raw if r["policy"] == policy and r["condition"] == condition],
                              key=lambda r: (r["layout"], r["target_index"]))
                ax.plot(range(12), [1000*r["target_aligned_displacement_m"] for r in rows], marker=marker, label=LABELS[policy])
            ax.axhline(2, color="gray", ls="--", label="Correct direction threshold")
            ax.axhline(0, color="black", lw=.5)
            ax.set_title(condition); ax.set_xticks(range(12))
            ax.set_xticklabels([r["layout"]+f"/{r['target_index']}" for r in rows], rotation=70)
            ax.set_ylabel("Displacement toward target (mm)"); ax.grid(alpha=.2)
        axes[1].legend(fontsize=8); fig.suptitle("One five-step decision; 6 layouts / 12 balanced worlds")
        fig.tight_layout(); fig.savefig(OUTPUT / "history_direction.png", dpi=180)
        fig.savefig(OUTPUT / "history_direction.pdf"); plt.close(fig)

    write_csv("scores.csv", scores)
    write_csv("paired_effects.csv", effects)
    evidence = {"models": list(POLICIES), "checkpoint": "runs/stage1_s17_gpu7/best.pt", "checkpoint_update": 1500,
                "analysis_manual_seed": ANALYSIS_SEED, "bootstrap_replicates": 10000,
                "geometry_source": str(G_RUN.relative_to(ROOT)), "geometry_existing_episodes_reanalyzed": 80,
                "history_source": str(H_RUN.relative_to(ROOT)), "history_complete": h_complete,
                "new_history_episodes": len(h_rows), "scores": scores, "effects": effects,
                "limits": ["Only native versus best: no separate causal attribution to geometry, history, or fine-tuning.",
                           "History is a five-step direction diagnostic, not task completion or active exploration.",
                           "Geometry: two tasks; history: one task, six admissible layouts. No unseen object/category claim.",
                           "Exploratory multiple endpoints; bootstrap intervals are not simultaneous significance tests.",
                           "Sampled path is a 5-step lower bound; latency is from different runs/hardware and not a speed benchmark.",
                           "No joint failure exists in the observed geometry pairs; do not fabricate one."]}
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    score_html = "".join(f"<tr><td>{r['suite']}</td><td>{r['condition']}</td><td>{LABELS[r['policy']]}</td>"
                         f"<td>{r['correct']}/{r['episodes']}</td><td>{r['independent_layouts']}</td></tr>" for r in scores)
    videos = "".join(f"<article><h3>{d['case_id']}</h3><p>原生成功：{d['native_success']}；best 成功：{d['best_success']}</p>"
                     f"<video controls preload='metadata' src='{d['file']}'></video></article>" for d in demos)
    links = " · ".join(f"<a href='{p.name}'>{html.escape(p.name)}</a>" for p in sorted(OUTPUT.iterdir()) if p.suffix in {".csv", ".json", ".pdf"})
    history_figure = "<img src='history_direction.png' alt='逐布局目标方向位移'>" if h_complete else "<p>新增历史测试仍在执行，尚未填写结果。</p>"
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>阶段一：best 与 MolmoAct2 原生</title><style>body{{max-width:1100px;margin:32px auto;padding:16px;font:17px/1.7 sans-serif;color:#183047}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7df;padding:8px}}video,img{{width:100%}}article{{margin:28px 0}}a{{color:#06729a}}.note{{background:#edf3f7;padding:18px}}</style>
<h1>阶段一到底学到了什么？</h1><p>只比较 MolmoAct2-LIBERO 原生与阶段一 best（update 1500）。相同任务、物理初态、动作预算和 manual seed；原生使用自己的当前帧接口，best 使用训练时的几何与历史接口。</p>
<p class='note'>这是两个完整系统的比较。可以判断 best 是否改善表现，不能把差异单独归因于几何模块或历史模块。相机变化测试复用已有日志；历史测试为本次 GPU 0 新执行。</p>
<h2>实际结果</h2><table><tr><th>测试</th><th>条件</th><th>模型</th><th>成功/正确</th><th>独立布局</th></tr>{score_html}</table>
<p>camera 计完整任务成功；history_direction 只计视觉恢复前首次 5 步是否朝正确目标移动。分母里的两种目标世界共用布局，统计时以布局为单位。</p>
<h2>历史如何影响第一次选择</h2>{history_figure}<p>目标分列 ±6 cm。隐藏条件下当前 RGB、depth、本体状态和指令相同，历史画面不同；此前执行动作相同。4 个物理状态不匹配布局在读取模型结果前剔除，保留 6 个布局。灰屏是人工压力测试，不能代表自然遮挡泛化。</p>
<p>样本量只有 6 个独立布局；小样本精确成对检验及 bootstrap 区间同时保留在 paired_effects.csv，不能仅凭 bootstrap 区间不跨零宣称显著。2 mm 是固定主阈值，1/5 mm 只是事后敏感性分析。主指标未达标但方向正确，也仍计失败。</p>
<h2>相机变化后的成对 demo</h2><p>包括全部结果不一致案例，加一个 manual seed 1317 选择的共同成功案例。按同一仿真步对齐，每 5 步一帧、20 Hz 仿真、4 fps 视频；完成后冻结并标记。视频不代表推理实时速度。</p>{videos}
<h2>可核查资产</h2><p>{links}</p><p><a href='../validation/stage1_history_eligible_scenes.jpg'>历史证据场景总览</a> · <a href='../../experiment_log.md'>统一实验日志</a></p>
<p>只有 2 个操作任务及 1 个历史任务，没有新物体类别、跨场景或主动获取信息的实证。置信区间及逐案例原始数值见 CSV；此处不把未检出的差异写成等价。</p></html>""")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "history_complete": h_complete,
                      "scores": scores, "effects": effects}, indent=2))


if __name__ == "__main__":
    main()
