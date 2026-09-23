"""Build review assets from existing logs and a design-only evaluation protocol.

No simulator, model, GPU, training, or new policy evaluation is invoked.
Run with .venv-sim/bin/python scripts/build_validation_assets.py.
"""

import csv
import html
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

OUTPUT = ROOT / "docs/assets/validation"
POLICIES = ["native", "best", "best_no_geometry"]
LABELS = {
    "native": "Native",
    "best": "Stage 1 best",
    "best_no_geometry": "Stage 1: geometry off",
    "best_current": "Stage 1: current only",
    "last": "Stage 1 last",
}


def read(relative):
    return json.loads((ROOT / relative).read_text())


def write_table(filename, rows, fields=None):
    fields = fields or list(rows[0])
    with (OUTPUT / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    head = "".join(f"<th>{html.escape(key)}</th>" for key in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in fields)
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(
        (ROOT / "experiments/evaluation_protocol.yaml").read_text()
    )
    assert (
        protocol["status"] == "design_only" and not protocol["execute_new_experiments"]
    )
    for suite in protocol["suites"]:
        expected = (
            suite["task_families"]
            * suite["layouts_per_family"]
            * len(suite["conditions"])
            * suite["target_worlds"]
        )
        if expected != suite["episodes_per_model"]:
            raise ValueError(f"episode arithmetic mismatch: {suite['id']}")
    total = sum(s["episodes_per_model"] for s in protocol["suites"])
    assert total == protocol["budget"]["closed_loop_episodes_per_model"] == 1584
    assert 3 * total == protocol["budget"]["initial_three_models_total"]
    assert 5 * total == protocol["budget"]["total_with_two_controls_on_all_suites"]

    g_path = "runs/stage1_geometry_test_s217/report.json"
    a_path = "runs/stage1_geometry_test_s217/analysis.json"
    h_path = "runs/stage1_history_single_bowl_dev_s117/report.json"
    p_path = "runs/stage1_paired_precheck_s117/report.json"
    g, a, h, p = map(read, (g_path, a_path, h_path, p_path))
    if not g["complete"] or len(g["episodes"]) != 120 or not p["complete"]:
        raise ValueError("existing completed scopes are missing")
    assert (
        "H" in h["completed_kinds"]
    )  # The development manifest also contains unused G cases.
    grouped = defaultdict(list)
    for row in g["episodes"]:
        grouped[row["policy"], row["condition"]].append(row)
    geometry = []
    for policy in POLICIES:
        for condition in ("normal", "shifted"):
            rows = grouped[policy, condition]
            assert len(rows) == 20 and len({r["layout"] for r in rows}) == 20
            geometry.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "successes": sum(r["success"] for r in rows),
                    "episodes": len(rows),
                    "success_rate": sum(r["success"] for r in rows) / len(rows),
                    "independent_layouts": 20,
                    "task_families": 2,
                    "source": g_path,
                    "scope": "completed historical diagnostic; not new confirmatory test",
                }
            )
    write_table("observed_geometry.csv", geometry)
    write_table("observed_geometry_cases.csv", g["episodes"], list(g["episodes"][0]))
    by_task = []
    for policy in POLICIES:
        for task in ("G_0", "G_1"):
            for condition in ("normal", "shifted"):
                rows = [
                    r
                    for r in g["episodes"]
                    if r["policy"] == policy
                    and r["layout"].startswith(task + "_")
                    and r["condition"] == condition
                ]
                by_task.append(
                    {
                        "policy": policy,
                        "task": task,
                        "condition": condition,
                        "successes": sum(r["success"] for r in rows),
                        "episodes": len(rows),
                    }
                )
    write_table("observed_geometry_per_task.csv", by_task)
    effects = []
    for comparison, effect in a["groups"]["G"]["paired_effects"].items():
        effects.append(
            {
                "comparison": comparison,
                "difference_pp": 100 * effect["paired_mean_difference"],
                "lower_95_pp": 100 * effect["interval_95"][0],
                "upper_95_pp": 100 * effect["interval_95"][1],
                "independent_layouts": effect["independent_layouts"],
                "manual_seed": effect["manual_seed"],
                "source": a_path,
            }
        )
    write_table("observed_paired_effects.csv", effects)
    history = []
    for policy in ("native", "best", "best_current"):
        for condition in ("visible", "hidden"):
            rows = [
                r
                for r in h["episodes"]
                if r["policy"] == policy and r["condition"] == condition
            ]
            assert len(rows) == 2 and {r["target_index"] for r in rows} == {0, 1}
            history.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "first_prefix_correct": sum(
                        r["first_choice"]["correct"] for r in rows
                    ),
                    "target_worlds": 2,
                    "independent_layouts": 1,
                    "final_success_after_vision_returns": sum(
                        r["success"] for r in rows
                    ),
                    "scope": "development only; first prefix is the memory endpoint",
                    "source": h_path,
                }
            )
    write_table("observed_history_development.csv", history)
    ordinary = []
    for policy in ("native", "best", "last"):
        rows = [r for r in p["episodes"] if r["policy"] == policy]
        ordinary.append(
            {
                "policy": policy,
                "successes": sum(r["success"] for r in rows),
                "episodes": len(rows),
                "scope": "engineering check on training starts",
                "source": p_path,
            }
        )
    write_table("observed_ordinary_precheck.csv", ordinary)
    methods = [
        {
            "id": m["id"],
            "method": m["label"],
            "status": m["status"],
            "purpose": m["purpose"],
        }
        for m in protocol["methods"]
    ]
    write_table("methods_and_controls.csv", methods)
    planned = []
    for suite in protocol["suites"]:
        for method in ("M0", "M1", "M2", "C2_action", "C2_detach"):
            planned.append(
                {
                    "suite": suite["id"],
                    "method": method,
                    "status": "not_run",
                    "planned_episodes": suite["episodes_per_model"],
                    "primary_metric": suite["primary_metric"],
                    "observed_value": "",
                    "interval_95": "",
                    "source_run": "",
                }
            )
    write_table("planned_results.csv", planned)
    demos = [
        {
            "id": "geometry",
            "purpose": "Projection changes and contact errors",
            "pair": "same state, different camera",
            "required_views": "normal/shifted; agent + wrist",
            "endpoint": "contact and goal",
            "status": "storyboard only",
        },
        {
            "id": "history",
            "purpose": "History changes the first decision",
            "pair": "identical current input, opposite real evidence",
            "required_views": "real past, current blackout, first five steps",
            "endpoint": "choice before reobservation",
            "status": "storyboard only",
        },
        {
            "id": "consequence",
            "purpose": "Actions predict their own observed consequences",
            "pair": "same initial state, three actual action branches",
            "required_views": "observed endpoints and latent error; no invented RGB future",
            "endpoint": "matching future branch",
            "status": "storyboard only",
        },
        {
            "id": "language",
            "purpose": "Same image and different valid goal",
            "pair": "canonical/paraphrase/changed target",
            "required_views": "instruction + chosen object",
            "endpoint": "goal conditioned on language",
            "status": "storyboard only",
        },
        {
            "id": "generalization",
            "purpose": "Transfer to held-out object/scene families",
            "pair": "separate layout/object/scene/composition strata",
            "required_views": "asset identity and scene family",
            "endpoint": "success and failure stage",
            "status": "storyboard only",
        },
        {
            "id": "active_perception",
            "purpose": "Acquire evidence only when needed and use it",
            "pair": "visible/hidden, two hidden worlds",
            "required_views": "policy input and separately marked evaluator truth",
            "endpoint": "reveal then correct evidence-dependent action",
            "status": "storyboard only",
        },
    ]
    write_table("demo_storyboards.csv", demos)
    exclusions = [
        {
            "case_id": "",
            "scene_family": "",
            "status": "template",
            "reason": "",
            "audit_before_model_results": "",
            "replacement_case_id": "",
            "source": "",
        }
    ]
    write_table("exclusion_template.csv", exclusions)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = ["#596579", "#207b9a", "#b08b39"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for index, policy in enumerate(POLICIES):
        vals = [
            100
            * next(
                r["success_rate"]
                for r in geometry
                if r["policy"] == policy and r["condition"] == c
            )
            for c in ("normal", "shifted")
        ]
        xs = [x + (index - 1) * 0.24 for x in (0, 1)]
        axes[0].bar(xs, vals, width=0.22, color=colors[index], label=LABELS[policy])
        for x, value in zip(xs, vals, strict=True):
            axes[0].text(
                x, value + 1.2, f"{round(value / 5)}/20", ha="center", fontsize=9
            )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Original camera", "Shifted camera"],
        ylim=(0, 115),
        ylabel="Success (%)",
        title="Completed G diagnostic: two task families",
    )
    axes[0].legend(loc="lower left", fontsize=8)
    for index, policy in enumerate(("native", "best", "best_current")):
        count = next(
            r["first_prefix_correct"]
            for r in history
            if r["policy"] == policy and r["condition"] == "hidden"
        )
        axes[1].bar(index, count, color=colors[index], width=0.6)
        axes[1].text(index, count + 0.06, f"{count}/2", ha="center")
    axes[1].set(
        xticks=[0, 1, 2],
        xticklabels=["Native", "Stage 1", "Current only"],
        ylim=(0, 2.5),
        yticks=[0, 1, 2],
        ylabel="Correct first direction (two target worlds)",
        title="H development: ONE independent layout",
    )
    fig.suptitle(
        "Observed evidence: geometry gain unsupported; history signal preliminary",
        fontsize=12,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"observed_evidence.{suffix}", dpi=180)
    plt.close(fig)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "new_experiments_run": 0,
        "protocol_status": "design_only",
        "source_reports": [g_path, a_path, h_path, p_path],
        "observed_geometry": geometry,
        "observed_paired_effects": effects,
        "observed_history_development": history,
        "new_planned_episodes_per_model": total,
        "new_planned_initial_three_models_episodes": 3 * total,
        "limits": [
            "Historical G cases have already been inspected.",
            "H has only one development layout.",
            "No stage-two final capabilities evaluated.",
            "No new scene assets downloaded or instantiated.",
        ],
    }
    (OUTPUT / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    scope_rows = [
        {
            "测试": s["id"] + " " + s["name"],
            "每模型回合": s["episodes_per_model"],
            "主要指标": s["primary_metric"],
            "状态": "设计，未运行",
        }
        for s in protocol["suites"]
    ]
    observed_rows = [
        {
            "模型": LABELS[m],
            "原视角": f"{grouped[m, 'normal'] and sum(r['success'] for r in grouped[m, 'normal'])}/20",
            "扰动视角": f"{sum(r['success'] for r in grouped[m, 'shifted'])}/20",
        }
        for m in POLICIES
    ]
    links = "".join(
        f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
        for path in sorted(OUTPUT.glob("*.csv"))
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>几何与历史 VLA：验证资产</title><style>
body{{font-family:system-ui,"Noto Sans CJK SC",sans-serif;max-width:1120px;margin:auto;padding:32px;color:#192b3d;line-height:1.7;background:#f8fafc}}
h1{{font-size:30px}}h2{{margin-top:36px}}.notice{{border-left:5px solid #c79329;background:#fff8e7;padding:16px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #d9e2eb;padding:10px}}th{{background:#e8eff5}}a{{color:#12657d}}img{{width:100%;background:white}}details{{background:white;padding:14px;margin:12px 0}}code{{background:#e8eff5;padding:2px 5px}}@media print{{body{{background:white;padding:0}}details{{display:block}}}}
</style><body><h1>两阶段训练究竟学到了什么？</h1>
<p>验证资产与评测设计 · {summary["generated_utc"]} · 数据来自已有原始记录</p>
<div class="notice"><strong>当前只做设计和已有结果整理，没有启动新实验。</strong> 新场景、48个物体身份、扩大测试和对照训练均待实现。阶段二仍在训练，当前没有它的最终能力成绩。</div>
<p><a href="../../experiment_log.md">统一实验文档（第2.11节）</a> · <a href="../../../experiments/evaluation_protocol.yaml">机器可读协议</a> · <a href="evidence.json">数字与来源</a></p>
<h2>已经知道的结果</h2><img src="observed_evidence.png" alt="几何测试没有观察到独立收益；历史仅单布局初步信号">
{table(observed_rows, ["模型", "原视角", "扰动视角"])}
<p>G共120回合、20个布局、2个任务族。扰动条件中阶段一相对原生差值−5个百分点，配对95%区间[−20,+10]；相对去几何差值0，区间[−15,+15]。不能据此确认收益，也不能证明等价。去几何是推理干预，不是重新训练的消融。</p>
<p>H灰屏首动作：原生1/2、阶段一2/2、仅当前输入1/2；仅一个开发布局、两个目标世界。恢复视觉后的最终成功不等于记住了历史。普通操作预检三种权重均4/4，但使用训练起点，只支持工程检查。</p>
<h2>设计中的正式验证</h2>{table(scope_rows, ["测试", "每模型回合", "主要指标", "状态"])}
<p>另有P：600个离线窗口×4种动作条件；B：48个初态×3条实际动作分支。它们分别检验预测敏感性与真实动作后果对应关系。</p>
<details open><summary><strong>对照与归因</strong></summary>{table(methods, ["id", "method", "status", "purpose"])}</details>
<details open><summary><strong>泛化与判定</strong></summary><p>按新布局、新物体、新场景、新组合分别报告；同一资产及变体不跨开发/测试划分。48个物体身份和六类场景目前是目标设计，底座预训练接触情况未知。M2需超过等预算动作续训对照才能把收益归于预测监督；还需在信息获取任务上通过“揭示→读取证据→正确行动”整条链，才能讨论主动感知迁移。</p></details>
<details><summary><strong>对比demo分镜（待录制）</strong></summary>{table(demos, ["id", "purpose", "pair", "required_views", "endpoint", "status"])}<p>同模拟步、同起点、同动作噪声；既展示成功也展示基线胜出和共同失败。隐藏真值单独标记为评测者视角。</p></details>
<h2>可下载表格</h2><ul>{links}<li><a href="observed_evidence.pdf">可用于汇报的矢量图 PDF</a></li></ul>
<p>更新已有结果资产：<code>.venv-sim/bin/python scripts/build_validation_assets.py</code>。该命令不运行策略或模拟器。新测试入口尚未实现；不要把此页面当作全套实验已完成。</p>
</body></html>"""
    (OUTPUT / "index.html").write_text(page)
    print(
        json.dumps(
            {
                "assets": str(OUTPUT.relative_to(ROOT)),
                "new_experiments": 0,
                "protocol_arithmetic_valid": True,
                "planned_per_model": total,
                "csv_files": len(list(OUTPUT.glob("*.csv"))),
            }
        )
    )


if __name__ == "__main__":
    main()
