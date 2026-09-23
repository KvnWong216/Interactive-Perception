"""Paired three-system scores with stage-two versus stage-one primary differences."""

import csv
import fcntl
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT / "runs/stage2_assessment_s3217_gpu1"
OUTPUT=ROOT / "docs/assets/stage2_assessment"


def read(path):
    return json.loads(path.read_text())


def csv_file(name,rows):
    if rows:
        with (OUTPUT / name).open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def endpoint(r):
    return bool(r["first_choice"] and r["first_choice"]["correct"]) if r["kind"]=="H" else bool(r["success"])


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT / "summary.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        summarize()


def summarize():
    if not (RUN / "report.json").exists():
        return
    report=read(RUN / "report.json");protocol=read(RUN / "protocol.json")
    outcomes={("stage2_best",r["case_id"]):r for r in report["episodes"]}
    if len(outcomes)!=len(report["episodes"]):
        raise ValueError("duplicate stage-two episode")
    references={c["reference_output"] for c in protocol["cases"]}
    for reference in sorted(references):
        baseline=read(ROOT / reference / "report.json")
        if baseline["checkpoint_updates"]!=1500 or baseline["modes"]!=["native","best"]:
            raise ValueError("stage-one baseline changed")
        for r in baseline["episodes"]:
            outcomes[r["policy"],r["case_id"]]=r
    models=("native","best","stage2_best")
    layouts=defaultdict(list)
    for c in protocol["cases"]:layouts[c["layout"]].append(c)
    cases=[c for cohort in layouts.values()
           if all((p,c["case_id"]) in outcomes for c in cohort for p in models) for c in cohort]
    complete=report["complete"] and len(cases)==len(protocol["cases"])
    groups=defaultdict(list)
    raw=[]
    for c in cases:
        for p in models:
            r=outcomes[p,c["case_id"]]
            if r["manual_seed"]!=c["manual_seed"] or (p!="native" and not r["paired_initial_observation_equal"]):
                raise ValueError("stage comparison not paired")
            raw.append({"policy":p,"case_id":c["case_id"],"kind":c["kind"],
                "split":c["our_training_split"],"condition":c["condition"],"task_index":c["task_index"],
                "layout":c["layout"],"endpoint":endpoint(r),"steps":r["policy_control_steps"]})
        for split in ("all",c["our_training_split"]):
            for condition in ("all",c["condition"]):
                # H all would conflate visible and hidden mechanism conditions.
                if c["kind"]!="H" or condition!="all":
                    groups[c["kind"],split,condition].append(c)
    scores=[];effects=[]
    for (kind,split,condition),cohort in sorted(groups.items()):
        for p in models:
            tasks=defaultdict(list)
            for c in cohort:tasks[c["task_index"]].append(endpoint(outcomes[p,c["case_id"]]))
            scores.append({"kind":kind,"split":split,"condition":condition,"policy":p,
                "successes":sum(sum(v) for v in tasks.values()),"episodes":len(cohort),"tasks":len(tasks),
                "macro_task_rate":float(np.mean([np.mean(v) for v in tasks.values()])),"complete":complete})
        if complete:
            task_layout=defaultdict(lambda:defaultdict(list))
            for c in cohort:
                delta=int(endpoint(outcomes["stage2_best",c["case_id"]]))-int(endpoint(outcomes["best",c["case_id"]]))
                task_layout[c["task_index"]][c["layout"]].append(delta)
            arrays=[np.array([np.mean(v) for v in layouts.values()]) for layouts in task_layout.values()]
            rng=np.random.default_rng(3317);draws=rng.integers(0,len(arrays),(10000,len(arrays)));means=np.empty(draws.shape)
            for i,a in enumerate(arrays):
                mask=draws==i;means[mask]=rng.choice(a,(int(mask.sum()),len(a)),replace=True).mean(axis=1)
            lo,hi=np.quantile(means.mean(axis=1),[.025,.975])
            effects.append({"kind":kind,"split":split,"condition":condition,"tasks":len(arrays),
                "stage2_minus_stage1":float(np.mean([a.mean() for a in arrays])),"ci_low":float(lo),"ci_high":float(hi)})
    prediction=[]
    diagnostic=read(RUN / "prediction_diagnostic.json") if (RUN / "prediction_diagnostic.json").exists() else {}
    if diagnostic.get("complete"):
        pg=defaultdict(lambda:defaultdict(list))
        for row in diagnostic["rows"]:
            for variant,score in row["scores"].items():
                for region,value in score.items():
                    if value is not None:pg[row["split"],row["task"],variant,region][row["episode_id"]].append(value)
        for (split,task,variant,region),episodes in sorted(pg.items()):
            prediction.append({"split":split,"task":task,"variant":variant,"region":region,
                "windows":sum(len(v) for v in episodes.values()),"trajectories":len(episodes),
                "loss":float(np.mean([np.mean(v) for v in episodes.values()]))})
    complete=complete and diagnostic.get("complete",False)
    csv_file("scores.csv",scores);csv_file("paired_episodes.csv",raw);csv_file("paired_effects.csv",effects)
    csv_file("prediction_by_task.csv",prediction)
    demos=[]
    if complete:
        from build_stage1_pair_assets import demo
        rng=np.random.default_rng(3317)
        for kind in ("S","P"):
            for a,b in ((True,True),(True,False),(False,True),(False,False)):
                eligible=sorted((c for c in cases if c["kind"]==kind and endpoint(outcomes["best",c["case_id"]])==a
                                 and endpoint(outcomes["stage2_best",c["case_id"]])==b),key=lambda c:c["case_id"])
                if not eligible:continue
                c=eligible[int(rng.integers(len(eligible)))];filename=c["case_id"]+".mp4"
                if not (OUTPUT / filename).exists():
                    demo(c["case_id"],[outcomes[p,c["case_id"]] for p in models],output=OUTPUT,
                         runs=[ROOT / c["reference_output"],ROOT / c["reference_output"],RUN])
                demos.append({"case_id":c["case_id"],"stage1_success":a,"stage2_success":b,"file":filename})
        csv_file("demos.csv",demos)
    summary={"complete":complete,"episodes_completed":len(report["episodes"]),"episodes_planned":len(protocol["cases"]),
        "balanced_cases":len(cases),"scores":scores,"effects":effects,"prediction_by_task":prediction,
        "demos":demos,"limits":protocol["limits"],"updated_utc":datetime.now(timezone.utc).isoformat()}
    temporary=OUTPUT / "evidence.partial";temporary.write_text(json.dumps(summary,indent=2)+"\n");temporary.replace(OUTPUT / "evidence.json")
    rows="".join(f"<tr><td>{r['kind']}/{r['split']}/{r['condition']}</td><td>{r['policy']}</td><td>{r['successes']}/{r['episodes']}</td></tr>" for r in scores)
    videos="".join(f"<h3>{d['case_id']}</h3><video controls preload='metadata' src='{d['file']}'></video>" for d in demos)
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>阶段二系统评估</title>
<style>body{{max-width:1200px;margin:24px auto;font:17px/1.7 sans-serif}}td,th{{border:1px solid #ccc;padding:6px}}table{{border-collapse:collapse}}video{{width:100%}}</style>
<h1>阶段二系统评估</h1><p>阶段二best update1500；新增回合 {len(report['episodes'])}/{len(protocol['cases'])}；完整评估完成：{complete}。</p>
<p>同场景、同种子，比较原生、阶段一best、阶段二best。只统计三模型及布局所有条件已完成的数据。S/P为任务成功，H为首次方向。</p>
<p>阶段二与阶段一差异同时包含额外训练、学习率变化和预测监督，不能直接解释为预测损失的独立因果贡献。预测动作替换是在固定真实未来上的离线诊断，不能替代模拟器反事实执行。</p>
<p><a href='scores.csv'>任务表</a> · <a href='paired_effects.csv'>配对区间（全量完成后）</a> · <a href='prediction_by_task.csv'>预测诊断（完成后）</a> · <a href='evidence.json'>证据</a> · <a href='../../experiment_log.md'>统一日志</a></p>
<p>{html.escape(str(report.get('error','')))}</p><table><tr><th>实验/划分/条件</th><th>模型</th><th>成功或正确/回合</th></tr>{rows}</table>{videos}</html>""")
    if complete:
        log=ROOT / "docs/experiment_log.md";marker="<!-- stage2-system-assessment-complete -->"
        if marker not in log.read_text():
            def count(kind,split,condition,policy):
                row=next(r for r in scores if (r['kind'],r['split'],r['condition'],r['policy'])==(kind,split,condition,policy))
                return f"{row['successes']}/{row['episodes']}"
            comparison=(f"**完整任务结果。** S全任务合计阶段一 {count('S','all','all','best')}、阶段二 {count('S','all','all','stage2_best')}；"
                f"正常条件阶段一 {count('S','all','normal','best')}、阶段二 {count('S','all','normal','stage2_best')}；"
                f"源任务test阶段一 {count('S','test','all','best')}、阶段二 {count('S','test','all','stage2_best')}。"
                "这些结果描述系统训练前后的差异，需结合按任务和布局成对区间及额外训练对照，不能仅凭全任务均值断言方向正确。")
            prediction_lines=[]
            for split in ('validation','test'):
                selected={variant:float(np.mean([r['loss'] for r in prediction if r['split']==split and r['region']=='changed' and r['variant']==variant]))
                          for variant in ('actual','copy_current','same_task_other_episode','zero_control','reverse_time')}
                prediction_lines.append(f"- {split}变化patch上的任务宏平均误差：真实动作 {selected['actual']:.6f}；复制当前 {selected['copy_current']:.6f}；同任务其他轨迹动作 {selected['same_task_other_episode']:.6f}；零控制 {selected['zero_control']:.6f}；时序反转 {selected['reverse_time']:.6f}。")
                prediction_lines.append("  解释："+("真实动作点估计优于复制当前，" if selected['actual']<selected['copy_current'] else "真实动作点估计未优于复制当前，")+
                    ("替换其他轨迹动作后误差上升。" if selected['actual']<selected['same_task_other_episode'] else "替换其他轨迹动作后误差没有上升。")+
                    "这些点估计不构成显著性或反事实动力学正确性的证明。")
            lines=[f"\n\n{marker}\n#### 阶段二系统评估完成\n\n固定best update1500，新增{len(cases)}次评测，与原生和阶段一形成三方成对比较。结果为探索性，不能将阶段二效果独立归因预测损失。\n\n",
                   comparison+"\n\n**未来预测机制。**\n\n"+'\n'.join(prediction_lines)+"\n\n",
                   "| 测试/划分/条件 | 模型 | 成功或正确/回合 |\n| --- | --- | ---: |\n"]
            lines += [f"| {r['kind']}/{r['split']}/{r['condition']} | {r['policy']} | {r['successes']}/{r['episodes']} |\n" for r in scores]
            lines.append("\n[完整结果、预测诊断及三方demo](assets/stage2_assessment/index.html)。H不等于任务成功，4个源任务test不能支持广泛新类别泛化。是否扩训须结合真实动作预测对复制当前特征的优势、错误动作干预、闭环test和正常技能保持共同判断。\n")
            with log.open('a') as f:f.write(''.join(lines))
        (RUN / "completion_signal.json").write_text(json.dumps({"signal":"二阶段训练的系统评估：已完成",
            "complete":True,"episodes":len(cases),"finished_utc":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"complete":complete,"stage2_episodes":len(report["episodes"]),"balanced_cases":len(cases)}),flush=True)


if __name__ == "__main__":
    main()
