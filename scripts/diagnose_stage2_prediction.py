"""Frozen held-out windows test persistence and action dependence of the predictor."""

import json
import time
from collections import defaultdict

import numpy as np
import torch
from evaluate_stage1 import ROOT, save_report
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.data import Trajectory, TrajectoryDataset


def make_plan(dataset):
    groups=defaultdict(list)
    for entry in dataset.entries:
        if entry["split"] in {"validation","test"}:
            groups[entry["split"],entry["task"]].append(entry)
    rng=np.random.default_rng(4217)
    plan=[]
    for (split,task),entries in sorted(groups.items()):
        entries=sorted(entries,key=lambda e:e["episode_id"])
        selected=sorted(rng.choice(len(entries),min(12,len(entries)),replace=False).tolist())
        task_rows=[]
        for selected_index in selected:
            entry=entries[selected_index]
            with np.load(dataset.resolve(entry),allow_pickle=False) as arrays:
                actions=arrays["actions"]
            # Fixed 10-step horizon, real target at t+10, no terminal padding.
            windows=np.arange(0,len(actions)-9,dataset.config.execute_steps)
            if len(windows)<2:
                raise ValueError("trajectory too short for two full prediction windows")
            for half,part in enumerate(np.array_split(windows,2)):
                step=int(rng.choice(part))
                task_rows.append({"episode_id":entry["episode_id"],"task":task,"split":split,
                    "step":step,"half":half,"actual_actions":actions[step:step+10].tolist()})
        for index,row in enumerate(task_rows):
            donors=[r for r in task_rows if r["episode_id"]!=row["episode_id"] and r["half"]==row["half"]]
            if not donors:
                raise ValueError("need a different-trajectory same-task action donor")
            donor=donors[int(rng.integers(len(donors)))]
            row.update(donor_episode=donor["episode_id"],donor_step=donor["step"],
                       donor_actions=donor["actual_actions"],manual_seed=4217+len(plan)+index)
        plan.extend(task_rows)
    return plan


def diagnose(backend,predictor,output):
    path=output / "prediction_diagnostic.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        return
    dataset=TrajectoryDataset(ROOT / "data/prepared/stage1_full/manifest.json",backend.config)
    plan=make_plan(dataset)
    plan_path=output / "prediction_windows.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("prediction diagnostic window selection changed")
    save_report(plan_path,plan)
    report=json.loads(path.read_text()) if path.exists() else {
        "complete":False,"manual_seed":4217,"windows_planned":len(plan),"rows":[],
        "horizon":10,"changed_patch_threshold_normalized_rms":0.1,
        "interpretation":"Offline action substitutions keep the observed true future fixed; sensitivity is necessary evidence, not proof of counterfactual simulator accuracy."}
    done={(r["episode_id"],r["step"]) for r in report["rows"]}
    entries={e["episode_id"]:e for e in dataset.entries}
    started=time.monotonic()
    trajectory=None;episode=None
    backend.reset();backend.train(False);predictor.eval()
    for row in plan:
        if (row["episode_id"],row["step"]) in done:
            continue
        if episode!=row["episode_id"]:
            entry=entries[row["episode_id"]]
            trajectory=Trajectory(dataset.resolve(entry),entry,total_steps=backend.config.total_steps)
            episode=row["episode_id"]
        step=row["step"];context=trajectory.context(step,backend.config)
        actual=np.asarray(row["actual_actions"],dtype=np.float32)
        if not np.array_equal(actual,trajectory.arrays["actions"][step:step+10]):
            raise ValueError("frozen diagnostic actions differ from real trajectory")
        set_manual_seed(row["manual_seed"])
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            shared=backend.encode(context)
            # Explicit seed immediately before flow noise: same specification as paired action-loss probes.
            set_manual_seed(row["manual_seed"])
            action_loss=float(backend.flow_loss(shared,actual))
            target,positions,valid=backend.target(trajectory.observation(step+10),context.task)
            current,current_positions,current_valid=backend.target(context.current,context.task)
            if target.shape!=current.shape or not torch.equal(positions,current_positions):
                raise ValueError("persistence baseline patch positions differ")
            truth=F.layer_norm(target.float(),(target.shape[-1],))
            persistence=F.layer_norm(current.float(),(current.shape[-1],))
            valid=valid & current_valid
            changed=valid & ((truth-persistence).square().mean(-1).sqrt()>0.1)
            def score(prediction, truth=truth, valid=valid, changed=changed):
                error=F.smooth_l1_loss(prediction.float(),truth,reduction="none").mean(-1)
                return {"all_valid":float(error[valid].mean()),
                        "changed":float(error[changed].mean()) if changed.any() else None}
            scores={"copy_current":score(persistence),"zero_feature":score(torch.zeros_like(truth))}
            variants={"actual":actual,"zero_control":np.zeros_like(actual),
                      "reverse_time":actual[::-1].copy(),
                      "same_task_other_episode":np.asarray(row["donor_actions"],dtype=np.float32)}
            predictions={}
            for name,actions in variants.items():
                normalized=backend.normalize_actions(actions)[None]
                prediction=predictor(shared.hidden,torch.ones_like(shared.ids,dtype=torch.bool),
                    normalized,torch.ones(normalized.shape[:2],device=normalized.device,dtype=torch.bool),
                    positions,torch.tensor([10],device=normalized.device))
                scores[name]=score(prediction)
                predictions[name]=prediction.float()
            dependence={name:float((pred-predictions["actual"])[valid].square().mean().sqrt())
                        for name,pred in predictions.items() if name!="actual"}
        result={k:v for k,v in row.items() if k not in {"actual_actions","donor_actions"}}
        result.update(action_loss=action_loss,scores=scores,valid_patches=int(valid.sum()),
                      changed_patches=int(changed.sum()),prediction_rms_change=dependence,
                      actual_donor_action_max_difference=float(np.abs(actual-variants["same_task_other_episode"]).max()))
        report["rows"].append(result);report["recorded_seconds_this_session"]=time.monotonic()-started
        save_report(path,report)
        print(json.dumps({"prediction_windows_completed":len(report["rows"]),"total":len(plan)}),flush=True)
    report["complete"]=len(report["rows"])==len(plan)
    save_report(path,report)
    backend.reset();torch.cuda.empty_cache()
