"""Create an auditable overview of the expanded native/best experiment."""

import csv
import fcntl
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage1_expanded_s3217"
DESIGN = ROOT / "experiments/diagnostic_cases/stage1_expanded_s3217"
OUTPUT = ROOT / "docs/assets/stage1_expanded"


def read(path):
    return json.loads(path.read_text())


def csv_file(name, rows):
    if not rows:
        return
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def endpoint(row):
    return bool(row["first_choice"] and row["first_choice"]["correct"]) if row["kind"] == "H" else bool(row["success"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summarize()


def summarize():
    protocol = read(DESIGN / "protocol.json")
    meta, outcomes, locations = {}, {}, {}
    audit_rows, eligibility, statuses = [], [], []
    for gpu in protocol["gpus"]:
        p = RUN / f"gpu{gpu}/status.json"
        statuses.append(read(p) if p.exists() else {"physical_gpu":gpu,"complete":False,"phases":{}})
    complete = all(s["complete"] and not s.get("error") for s in statuses)
    for shard in protocol["shards"]:
        spec = read(ROOT / shard["manifest"])
        meta.update({c["case_id"]:c for c in spec["cases"]})
        phase = RUN / f"gpu{shard['physical_gpu']}" / shard["kind"]
        audit_path = phase / "audit/report.json"
        if audit_path.exists():
            audit = read(audit_path)
            eligibility.append({"gpu":shard["physical_gpu"],"kind":shard["kind"],
                                "candidate_cases":len(spec["cases"]),"audited_candidate_cases":audit["candidate_cases"],
                                "eligible_cases":audit["cases"],"passed":audit["passed"]})
            for a in audit["audits"]:
                case = next(c for c in spec["cases"] if c["layout"] == a["layout"])
                audit_rows.append({"gpu":shard["physical_gpu"], "kind":shard["kind"],"layout":a["layout"],
                                   "task_index":case["task_index"],"target_object":case.get("target_object",""),
                                   "passed":a["passed"],"reasons":"; ".join(a["exclusion_reasons"])})
        report_path = phase / "evaluation/report.json"
        if report_path.exists():
            report = read(report_path)
            assert report["modes"] == ["native","best"] and report["checkpoint_updates"] == 1500
            for row in report["episodes"]:
                key = row["policy"],row["case_id"]
                assert key not in outcomes
                if row["kind"] == "H":
                    assert row["policy_control_steps"] == 5 and row["policy_calls"] == 1
                outcomes[key] = row
                locations[row["case_id"]] = phase / "evaluation"
    # Never compare an early complete native cohort to an incomplete best cohort.
    layouts = defaultdict(list)
    for case in meta.values():
        layouts[case["layout"]].append(case)
    paired_cases = [c for cases in layouts.values()
                    if all((p,c["case_id"]) in outcomes for c in cases for p in ("native","best"))
                    for c in cases]
    raw, paired = [], []
    for case in paired_cases:
        for policy in ("native","best"):
            row = outcomes[policy,case["case_id"]]
            fc = row["first_choice"]
            raw.append({"kind":case["kind"],"case_id":case["case_id"],"layout":case["layout"],
                        "task_index":case["task_index"],"our_training_split":case["our_training_split"],
                        "target_object":case.get("target_object",""),"condition":case["condition"],
                        "policy":policy,"correct_or_successful":endpoint(row),
                        "control_steps":row["policy_control_steps"],"wall_seconds":row["wall_seconds"],
                        "axis_displacement_m":fc["axis_displacement_m"] if fc else "",
                        "target_index":case.get("target_index",""),"manual_seed":row["manual_seed"]})
        paired.append({**case,"delta":int(endpoint(outcomes["best",case["case_id"]]))-int(endpoint(outcomes["native",case["case_id"]]))})
    scores = []
    groups = defaultdict(list)
    for row in raw:
        groups[row["kind"],row["our_training_split"],row["condition"],row["policy"]].append(row)
    for (kind,split,condition,policy),rows in sorted(groups.items()):
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_index"]].append(r["correct_or_successful"])
        scores.append({"kind":kind,"our_training_split":split,"condition":condition,"policy":policy,
                       "correct_or_successful":sum(r["correct_or_successful"] for r in rows),"episodes":len(rows),
                       "independent_layouts":len({r["layout"] for r in rows}),"tasks":len(by_task),
                       "macro_task_rate":float(np.mean([np.mean(v) for v in by_task.values()])),
                       "scope_complete":complete})
    effects = []
    if complete:
        for kind,split,condition in sorted({(r["kind"],r["our_training_split"],r["condition"]) for r in paired}):
            subset = [r for r in paired if (r["kind"],r["our_training_split"],r["condition"]) == (kind,split,condition)]
            task_layouts = defaultdict(lambda:defaultdict(list))
            for r in subset:
                task_layouts[r["task_index"]][r["layout"]].append(r["delta"])
            values = [np.asarray([np.mean(v) for v in layouts.values()]) for layouts in task_layouts.values()]
            rng = np.random.default_rng(3317)
            task_draws = rng.integers(0,len(values),(10000,len(values)))
            means = np.empty_like(task_draws,dtype=float)
            for task_index,array in enumerate(values):
                mask = task_draws == task_index
                means[mask] = rng.choice(array,(int(mask.sum()),len(array)),replace=True).mean(axis=1)
            low,high = np.quantile(means.mean(axis=1),[.025,.975])
            effects.append({"kind":kind,"our_training_split":split,"condition":condition,
                            "best_minus_native_macro_task":float(np.mean([v.mean() for v in values])),
                            "ci_low":float(low),"ci_high":float(high),"tasks":len(values),
                            "independent_layouts":sum(len(v) for v in values),"analysis_manual_seed":3317,
                            "interpretation":"exploratory hierarchical interval, not a multiplicity-corrected significance claim"})
    csv_file("paired_episodes.csv",raw);csv_file("scores.csv",scores);csv_file("paired_effects.csv",effects)
    dependency = []
    history_layouts = {c["layout"] for c in paired_cases if c["kind"] == "H"}
    for layout in sorted(history_layouts):
        hidden = sorted([c for c in paired_cases if c["layout"] == layout and c["condition"] == "hidden"],
                        key=lambda c:c["target_index"])
        assert len(hidden) == 2
        for policy in ("native","best"):
            controls = []
            for case in hidden:
                path = locations[case["case_id"]] / policy / case["case_id"] / case["episode"] / "rollout.jsonl"
                with path.open() as stream:
                    action = next(json.loads(line) for line in stream if '"requested_actions"' in line)
                assert action["manual_seed"] == case["manual_seed"]+40
                assert action["policy_input_observation_steps"] == ([40] if policy == "native" else [30,35,40])
                controls.append(np.asarray(action["requested_actions"]))
            change = float(np.max(np.abs(controls[0]-controls[1])))
            if policy == "native" and change != 0:
                raise ValueError("native hidden actions differ for identical current inputs and noise")
            dependency.append({"layout":layout,"target_object":hidden[0]["target_object"],"policy":policy,
                               "hidden_world_action_max_abs_difference":change})
    csv_file("history_action_dependence.csv",dependency)
    task_rows = []
    task_groups = defaultdict(list)
    sources = {t["task_index"]:t["source"] for t in protocol["tasks"]}
    for row in raw:
        task_groups[row["kind"],row["task_index"],row["condition"],row["policy"]].append(row)
    for (kind,task,condition,policy),rows in sorted(task_groups.items()):
        task_rows.append({"kind":kind,"task_index":task,"source":sources[task],
                          "our_training_split":rows[0]["our_training_split"],
                          "target_object":rows[0]["target_object"],"condition":condition,"policy":policy,
                          "correct_or_successful":sum(r["correct_or_successful"] for r in rows),
                          "episodes":len(rows),"scope_complete":complete})
    csv_file("task_rates.csv",task_rows)
    history_complete = all(s["phases"].get("H") in {"complete","not_evaluated_no_valid_paired_layout"} for s in statuses)
    if history_complete and dependency:
        import os
        os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        objects = sorted({r["target_object"] for r in raw if r["kind"] == "H"})
        fig,ax = plt.subplots(figsize=(10,4))
        for i,policy in enumerate(("native","best")):
            positions = np.arange(len(objects))+(i-.5)*.35
            selected = [[r for r in raw if r["kind"] == "H" and r["condition"] == "hidden"
                         and r["target_object"] == obj and r["policy"] == policy] for obj in objects]
            rates = [np.mean([r["correct_or_successful"] for r in rows]) for rows in selected]
            bars = ax.bar(positions,rates,width=.35,label=policy)
            for bar,rows in zip(bars,selected):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,
                        f"{sum(r['correct_or_successful'] for r in rows)}/{len(rows)}",ha="center")
        ax.set_xticks(np.arange(len(objects)),[o.removesuffix('_1').replace('_',' ') for o in objects])
        ax.set_ylim(0,1.2);ax.set_ylabel("First-five-step correct direction (>2 mm)")
        ax.set_title(f"Hidden current view: {len(objects)}/6 object tasks have valid paired scenes")
        ax.legend();fig.tight_layout();fig.savefig(OUTPUT/'history_by_object.png',dpi=170)
        fig.savefig(OUTPUT/'history_by_object.pdf');plt.close(fig)
    csv_file("scene_exclusions.csv",audit_rows);csv_file("eligibility.csv",eligibility)
    demos = []
    if complete:
        from build_stage1_pair_assets import demo
        rng = np.random.default_rng(3317)
        for kind in ("S","P"):
            for a,b in ((True,True),(True,False),(False,True),(False,False)):
                candidates = sorted(c["case_id"] for c in paired_cases if c["kind"] == kind
                    and endpoint(outcomes["native",c["case_id"]]) == a and endpoint(outcomes["best",c["case_id"]]) == b)
                if not candidates:
                    continue
                selected = str(rng.choice(candidates))
                filename = f"{selected}.mp4"
                if not (OUTPUT / filename).exists():
                    demo(selected,[outcomes[p,selected] for p in ("native","best")],run=locations[selected],output=OUTPUT)
                demos.append({"kind":kind,"native_success":a,"best_success":b,"case_id":selected,"file":filename})
        csv_file("demos.csv",demos)
    evidence = {"complete":complete,"manual_seed":3217,"analysis_manual_seed":3317,"models":["native","best"],
                "candidate_cases":808,"maximum_policy_episodes":1616,"completed_policy_episodes":len(outcomes),
                "balanced_completed_cases":len(paired_cases),"worker_status":statuses,
                "eligibility":eligibility,"scores":scores,"effects":effects,"demos":demos,
                "limits":protocol["interpretation"]}
    temporary = OUTPUT / "evidence.partial"
    temporary.write_text(json.dumps(evidence,indent=2)+"\n")
    temporary.replace(OUTPUT / "evidence.json")
    score_html = "".join(f"<tr><td>{r['kind']}</td><td>{r['our_training_split']}</td><td>{r['condition']}</td><td>{r['policy']}</td><td>{r['correct_or_successful']}/{r['episodes']}</td><td>{r['tasks']}</td></tr>" for r in scores)
    links = " · ".join(f"<a href='{p.name}'>{p.name}</a>" for p in sorted(OUTPUT.glob('*.csv')))
    videos = "".join(f"<h3>{d['case_id']}：原生 {d['native_success']} / best {d['best_success']}</h3><video controls preload='metadata' src='{d['file']}'></video>" for d in demos)
    history_figure = "<img width='100%' src='history_by_object.png' alt='有效跨物体历史场景的首次选择'>" if history_complete and dependency else ""
    history_videos = ""
    if (OUTPUT / "history_demos.json").exists():
        history_videos = "<h2>历史证据与首次选择</h2><p>按 manual seed 3317 每类物体选一个有效布局；展示真实过去画面、当前灰屏和动作执行后的评估画面。采用讲解停帧，不代表实时推理速度。</p>" + "".join(
            f"<h3>{d['object']} · {d['layout']}</h3><video controls preload='metadata' src='{d['file']}'></video>"
            for d in read(OUTPUT / "history_demos.json"))
    states = "".join(f"<li>分片 {s['physical_gpu']} / 执行 GPU {s.get('execution_gpu',s['physical_gpu'])}：{html.escape(str(s['phases']))}{html.escape(str(s.get('error','')))}</li>" for s in statuses)
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>阶段一扩大泛化实验</title>
<style>body{{max-width:1120px;margin:28px auto;padding:16px;font:17px/1.7 sans-serif;color:#183047}}td,th{{border:1px solid #ccd7df;padding:8px}}table{{border-collapse:collapse}}video{{width:100%}}</style>
<h1>40 任务与跨物体机制诊断</h1><p>只比较阶段一 best（1500）与 MolmoAct2 原生。808 个候选条件，最多 1,616 次 rollout。完成：{complete}；已记录策略回合：{len(outcomes)}。表格只纳入两模型及所有条件均已完成的布局，其他样本仍待评估。</p>
<p>S：40 个任务的视角/灯光变化；P：6 类物体真实偏移 ±4 cm 后的完整任务；H：6 类物体的历史证据与首次 5 步方向，阈值 2 mm。H 的方向正确不等于完成任务。</p><ul>{states}</ul>
<p>场景剔除先于策略结果；按物理接触、非目标状态一致性、标定和信息隔离决定。被剔除的任务必须显式报告，不能算作模型失败或从分母中无声消失。H 当前灰屏是机制压力测试。</p>
{history_figure}{history_videos}
<table><tr><th>测试</th><th>本项目划分</th><th>条件</th><th>模型</th><th>成功/正确</th><th>任务数</th></tr>{score_html}</table>
<p>{links} · <a href='evidence.json'>完整证据</a> · <a href='../stage1_generalization/index.html'>先前192次结果</a> · <a href='../../experiment_log.md'>统一日志</a></p>{videos}
<p>不同回合保留不同种子及原始记录。本批是在看过上一轮结果后开展的探索性扩展。初态对本次评测是新的，不一定对训练是新的；仅4个源任务属于原始test划分。所有物体仍是现有LIBERO网格，底座可能见过，不能宣称全新物体类别泛化。</p></html>""")
    if complete:
        marker = "扩大阶段一泛化实验完成（manual seed 3217）"
        log = ROOT / "docs/experiment_log.md"
        if marker not in log.read_text():
            lines = [f"\n\n#### {marker}\n\n",f"共完成 {len(outcomes)} 次策略评估；候选上限1616，实际分母受模型无关场景审计剔除影响。逐项结果、所有剔除及对比视频见 `docs/assets/stage1_expanded/`。\n\n"]
            lines += [f"- {r['kind']}/{r['our_training_split']}/{r['condition']}/{r['policy']}：{r['correct_or_successful']}/{r['episodes']}，{r['tasks']}个任务。\n" for r in scores]
            lines.append("\nH仅是首次方向；S/P为任务成功。统计按任务和布局成对，不能独立归因于模块，尚无新物体类别证据。\n")
            with log.open("a") as stream:
                stream.write("".join(lines))
    print(json.dumps({"complete":complete,"episodes":len(outcomes),"balanced_cases":len(paired_cases),"output":str(OUTPUT.relative_to(ROOT))}))
    if complete:
        write_system_assessment(protocol, eligibility, paired, outcomes, raw, demos)


def write_system_assessment(protocol, eligibility, paired, outcomes, raw, demos):
    """Issue the completion signal only after coverage and all final assets pass."""
    expected = set()
    for row in eligibility:
        if row["passed"]:
            manifest = read(RUN / f"gpu{row['gpu']}" / row["kind"] / "audit/eligible/manifest.json")
            expected.update((model, c["case_id"]) for c in manifest["cases"] for model in ("native", "best"))
    if set(outcomes) != expected or len(paired)*2 != len(expected):
        raise ValueError("final assessment lacks complete eligible paired coverage")
    if any(not r["paired_initial_observation_equal"] for (p,_),r in outcomes.items() if p == "best"):
        raise ValueError("final assessment contains an unpaired initial observation")
    for demo in demos:
        if not (OUTPUT / demo["file"]).is_file():
            raise ValueError("comparison video missing")

    def summarize_group(kind, condition=None, split=None):
        cases = [c for c in paired if c["kind"] == kind
                 and (condition is None or c["condition"] == condition)
                 and (split is None or c["our_training_split"] == split)]
        by_task = defaultdict(lambda: defaultdict(list))
        counts = {p: sum(endpoint(outcomes[p,c["case_id"]]) for c in cases) for p in ("native", "best")}
        for c in cases:
            by_task[c["task_index"]][c["layout"]].append(c["delta"])
        values = [np.array([np.mean(v) for v in layouts.values()]) for layouts in by_task.values()]
        rng = np.random.default_rng(3317)
        draws = rng.integers(0, len(values), (10000, len(values)))
        means = np.empty(draws.shape)
        for i, array in enumerate(values):
            mask = draws == i
            means[mask] = rng.choice(array, (int(mask.sum()), len(array)), replace=True).mean(axis=1)
        low, high = np.quantile(means.mean(axis=1), [.025, .975])
        return {"kind":kind,"condition":condition or "all","split":split or "all",
                "episodes_per_model":len(cases),"tasks":len(values),
                "layouts":sum(len(v) for v in values),"native_successes":counts["native"],
                "best_successes":counts["best"],
                "native_rate":counts["native"]/len(cases),"best_rate":counts["best"]/len(cases),
                "paired_macro_task_delta":float(np.mean([v.mean() for v in values])),
                "exploratory_ci95":[float(low),float(high)]}

    groups = [summarize_group("S", c) for c in (None,"normal","camera","lighting","combined")]
    groups += [summarize_group("S", split=s) for s in ("train","validation","test")]
    groups += [summarize_group("P", c) for c in (None,"normal","minus_y","plus_y")]
    groups += [summarize_group("H", c) for c in ("visible","hidden")]
    resources = []
    for policy in ("native","best"):
        rows = [r for (p,_),r in outcomes.items() if p == policy]
        resources.append({"policy":policy,"episodes":len(rows),
            "recorded_rollout_hours":sum(r["wall_seconds"] for r in rows)/3600,
            "median_episode_policy_median_seconds":float(np.median([r["policy_seconds_median"] for r in rows if r["policy_seconds_median"] is not None])),
            "max_allocated_gib":max(r["peak_allocated_gib"] for r in rows),
            "max_reserved_gib":max(r["peak_reserved_gib"] for r in rows)})
    transitions = []
    for condition in ("normal","camera","lighting","combined"):
        for task in protocol["tasks"]:
            cases = [c for c in paired if c["kind"] == "S" and c["condition"] == condition
                     and c["task_index"] == task["task_index"]]
            bins = {(a,b):0 for a in (False,True) for b in (False,True)}
            for c in cases:
                bins[endpoint(outcomes["native",c["case_id"]]),endpoint(outcomes["best",c["case_id"]])] += 1
            transitions.append({"task_index":task["task_index"],"source":task["source"],
                "split":cases[0]["our_training_split"],"condition":condition,
                "both_success":bins[True,True],"native_only":bins[True,False],
                "best_only":bins[False,True],"both_fail":bins[False,False],
                "best_minus_native":(bins[False,True]-bins[True,False])/len(cases)})
    repairs = [str(p.relative_to(ROOT)) for p in RUN.glob("gpu*/S/evaluation/best/*/*/initial_render_alignment.json")]
    assessment = {"status":"COMPLETE", "signal":"一阶段训练的系统评估：已完成",
        "finished_utc":datetime.now(timezone.utc).isoformat(),"manual_seed":3217,"analysis_manual_seed":3317,
        "checkpoint":"runs/stage1_s17_gpu7/best.pt","checkpoint_updates":1500,
        "completed_policy_episodes":len(outcomes),"eligible_policy_episodes":len(expected),
        "groups":groups,"resources":resources,"initial_render_alignment_records":repairs,
        "interpretation":{
            "history":"局部支持：使用历史证据作短时方向决策；不等于完整任务成功或主动获取信息。",
            "geometry":"未独立验证：当前两模型比较混合了训练、新增输入和接口差异。",
            "retention":"未预设不劣界限，不能仅凭差异小或区间跨零判定原技能不退化。",
            "generalization":"查看源任务test单独结果；train与validation不替代test。LIBERO物体与任务可能已被底座见过。",
            "prediction_and_active_perception":"未检验：阶段一不含未来预测监督，阶段二权重未参加本次评测。",
            "statistics":"探索性按任务、布局两层成对bootstrap，10000次；没有多重比较校正，只有4个test源任务。",
            "runtime":"每回合策略中位耗时的中位数；非所有调用合并P50。累计回合墙钟不含加载、排队、审计和失败尝试，不能当作总GPU计费。"}}
    csv_file("system_scores.csv",groups)
    csv_file("runtime.csv",resources)
    csv_file("failure_transitions.csv",transitions)
    (OUTPUT / "system_assessment.json").write_text(json.dumps(assessment,indent=2,ensure_ascii=False)+"\n")

    import matplotlib.pyplot as plt
    conditions = groups[1:5]
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(4)
    for i, p in enumerate(("native","best")):
        bars = ax.bar(x+(i-.5)*.36,[g[p+"_rate"]*100 for g in conditions],width=.36,label=p)
        for bar,g in zip(bars,conditions):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                    f"{g[p+'_successes']}/{g['episodes_per_model']}",ha="center",fontsize=9)
    ax.set_xticks(x,[g["condition"] for g in conditions]);ax.set_ylim(0,112)
    ax.set_ylabel("Full-task success (%)");ax.set_title("Stage 1 best vs native: all 40 LIBERO source tasks")
    ax.legend();fig.tight_layout();fig.savefig(OUTPUT/'task_success.png',dpi=180)
    fig.savefig(OUTPUT/'task_success.pdf');plt.close(fig)
    task_order = sorted(protocol["tasks"],key=lambda t:(
        next(r["split"] for r in transitions if r["task_index"] == t["task_index"]),t["task_index"]))
    labels = []
    matrix = []
    for task in task_order:
        rows = [r for r in transitions if r["task_index"] == task["task_index"]]
        matrix.append([next(r["best_minus_native"] for r in rows if r["condition"] == c)
                       for c in ("normal","camera","lighting","combined")])
        labels.append(f"{rows[0]['split']}: {Path(task['source']).stem.removesuffix('_demo')}")
    fig,ax=plt.subplots(figsize=(14,13))
    plot=ax.imshow(np.asarray(matrix)*100,cmap="RdBu",vmin=-100,vmax=100,aspect="auto")
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    ax.set_xticks(range(4),["normal","camera","lighting","combined"])
    ax.set_title("Best minus native: full-task success (4 paired layouts per cell)")
    fig.colorbar(plot,ax=ax,label="Percentage points");fig.tight_layout()
    fig.savefig(OUTPUT/'task_differences.png',dpi=180);fig.savefig(OUTPUT/'task_differences.pdf');plt.close(fig)

    table = ["| 评测 / 本项目划分 | 原生 | 阶段一 best | 宏平均差值及探索性95%区间（百分点） |",
             "| --- | ---: | ---: | ---: |"]
    for g in groups:
        n=g["episodes_per_model"];lo,hi=g["exploratory_ci95"]
        table.append(f"| {g['kind']}/{g['condition']}/{g['split']} | {g['native_successes']}/{n}（{100*g['native_rate']:.1f}%） | {g['best_successes']}/{n}（{100*g['best_rate']:.1f}%） | {100*g['paired_macro_task_delta']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] |")
    overall, normal, heldout = groups[0], groups[1], groups[7]
    findings = (f"40任务合计：原生 {overall['native_successes']}/{overall['episodes_per_model']}，best {overall['best_successes']}/{overall['episodes_per_model']}，"
        f"差值 {100*(overall['best_rate']-overall['native_rate']):+.2f} 个百分点。"
        f"正常场景差值 {100*(normal['best_rate']-normal['native_rate']):+.2f} 个百分点；"
        f"本项目4个test源任务合计：原生 {heldout['native_successes']}/{heldout['episodes_per_model']}，best {heldout['best_successes']}/{heldout['episodes_per_model']}。"
        "结合各扰动条件及配对区间解读，不能用全部任务均值取代test结果。")
    section = "\n".join([
        "<!-- stage1-system-assessment:start -->", "#### 一阶段训练的系统评估：已完成", "",
        f"完成时间（UTC）：{assessment['finished_utc']}。有效评测 **{len(outcomes)}/{len(expected)}**，仅比较原生与阶段一 best update1500。manual seed3217，统计seed3317。",
        "", findings, "", *table, "",
        "S/P为完整任务成功率；H为首次5步方向超过2mm的正确率，不能合并为一个总成功率。S覆盖40任务×4布局×4条件；P只有4/6类物体通过场景审计，H只有4/6类、13布局。完整剔除列表保留。",
        "", "普通任务与扰动成绩回答系统表现是否改变；历史实验回答是否能利用已有历史。两者都不能独立证明几何模块贡献，也不能证明主动探索。普通场景保持没有预注册的不劣界限；test仅4个本项目未训练源任务，底座可能已见过。以上区间为探索性，不作多重校正后的显著性主张。",
        "", f"GPU5分片中断于172/256回合，剩余84回合按用户授权迁到GPU0，保留全部已完成回合、手动种子与原始中断记录。原错误为腕部RGB仅3个颜色分量相差1/255。现只允许每视角≤0.01%颜色分量、幅值≤1/255的舍入差；其他数组及完整物理状态必须完全相同，再以原生初始帧作为双方共同输入，保存原始渲染与修正记录。本次实际发生 {len(repairs)} 次初帧归一。未修改奖励、成功判据、权重或场景。跨设备后续渲染/数值差异仍是限制。",
        "", "资源统计（回合耗时不含模型加载、排队、审计及中断尝试；共享主机不支持严格吞吐比较）：", "",
        "| 模型 | 回合墙钟累计/h | 每回合策略中位耗时的中位数/s | allocated / reserved 峰值/GiB |",
        "| --- | ---: | ---: | ---: |",
        *[f"| {r['policy']} | {r['recorded_rollout_hours']:.3f} | {r['median_episode_policy_median_seconds']:.3f} | {r['max_allocated_gib']:.2f} / {r['max_reserved_gib']:.2f} |" for r in resources],
        "", "[完整表格、图与对比demo](assets/stage1_expanded/index.html) · [机器可读系统评估](assets/stage1_expanded/system_assessment.json)。此前12任务两者84/96的阴性结果继续保留，不合并不同种子的重复任务冒充独立任务。",
        "<!-- stage1-system-assessment:end -->", ""])
    log=ROOT/'docs/experiment_log.md';content=log.read_text()
    start="<!-- stage1-system-assessment:start -->";end="<!-- stage1-system-assessment:end -->"
    if start in content:
        before, rest=content.split(start,1);_,after=rest.split(end,1)
        content=before+section+after
    else:
        content=content.replace("### 3.4",section+"\n### 3.4",1)
    content=content.replace("正在扩大阶段一双模型泛化评测", "阶段一双模型扩大评测已完成，见3.3节系统评估")
    content=content.replace("| 阶段一完整闭环评测 | 未完成；下述 12 回合开发预检完成 | 开发起点通过不足以确认全任务原技能保持 |",
                            "| 阶段一完整闭环评测 | 本轮40任务及跨物体诊断已完成，见下方系统评估 | 正常、扰动与源任务test分开解释；未作原技能不劣判定 |")
    content=content.replace("四卡联合训练已启动；尚无训练完成后的预测机制/闭环评测", "四卡联合训练已完成；尚无阶段二预测机制/闭环评测")
    lines=content.splitlines()
    for i,line in enumerate(lines):
        if line.startswith("| H1：原技能保持 |"):
            lines[i]=(f"| H1：原技能保持 | **普通操作可用，未建立统计不劣结论** | 扩大S正常条件原生{normal['native_successes']}/{normal['episodes_per_model']}、best{normal['best_successes']}/{normal['episodes_per_model']}；未预设不劣界限，结合3.3节配对区间解释 |")
        if line.startswith("**现阶段结论：工程可行性已有支持"):
            lines[i]="**现阶段结论：接口训练可行、历史条件决策有局部证据；完整操作的正常与扰动表现见3.3节系统评估，几何独立贡献、动作后果学习及主动获取信息仍需各自对照。** 动作loss或总体均值不能替代这些机制验证。"
    content='\n'.join(lines)+'\n'
    log.write_text(content)
    page=OUTPUT/'index.html'
    overview=f"<h2>一阶段训练的系统评估：已完成</h2><p>{findings}</p><p>有效回合全部完成。S/P为完整任务成功，H为首次方向。<a href='system_assessment.json'>系统评估与限制</a> · <a href='system_scores.csv'>汇总表</a> · <a href='runtime.csv'>资源实测</a> · <a href='failure_transitions.csv'>逐任务成败变化</a></p><img width='100%' src='task_success.png' alt='40任务完整成功率'><img width='100%' src='task_differences.png' alt='逐任务成功率差值'>"
    page.write_text(page.read_text().replace("<h1>40 任务与跨物体机制诊断</h1>","<h1>40 任务与跨物体机制诊断</h1>"+overview))
    # Written last: existence signals both rollouts and reviewable assets completed.
    temporary=RUN/'completion_signal.partial'
    temporary.write_text(json.dumps({k:assessment[k] for k in (
        "status","signal","finished_utc","completed_policy_episodes","eligible_policy_episodes")},ensure_ascii=False,indent=2)+"\n")
    temporary.replace(RUN/'completion_signal.json')


if __name__ == "__main__":
    main()
