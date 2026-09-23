"""Evaluate fixed stage-two best on the frozen, audited stage-one cases."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
from evaluate_stage1 import ROOT, rollout, save_report
from gpu_devices import verify_cuda_target

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.training import load_checkpoint, make_predictor

RUN = ROOT / "runs/stage2_assessment_s3217_gpu1"
BASE = ROOT / "runs/stage1_expanded_s3217"
CHECKPOINT = "runs/stage2_s17_gpu4567/best.pt"


def frozen_protocol():
    cases, audits = [], []
    for kind in ("H","P","S"):
        for gpu in (0,4,5,6,7):
            phase = BASE / f"gpu{gpu}" / kind
            audit = json.loads((phase / "audit/report.json").read_text())
            audits.append({"kind":kind,"logical_shard":gpu,"eligible_cases":audit["cases"]})
            if not audit["passed"]:
                continue
            manifest = phase / "audit/eligible/manifest.json"
            spec = json.loads(manifest.read_text())
            if audit["manifest"] != str(manifest.relative_to(ROOT)) or audit["cases"] != len(spec["cases"]):
                raise ValueError("scene audit and eligible cases disagree")
            for case in spec["cases"]:
                baseline = phase / "evaluation"
                native = json.loads((baseline / "report.json").read_text())
                if not any(r["policy"] == "native" and r["case_id"] == case["case_id"] for r in native["episodes"]):
                    raise ValueError("native reference episode is not complete")
                cases.append({**case,"case_file":str((manifest.parent / (case["case_id"]+".json")).relative_to(ROOT)),
                              "reference_output":str(baseline.relative_to(ROOT))})
    return {"manual_seed":3217,"analysis_manual_seed":3317,"physical_gpu":1,
            "checkpoint":CHECKPOINT,"checkpoint_updates":1500,
            "selection":"fixed training-validation joint objective; no evaluation-based reselection",
            "comparators":["native","stage1_best","stage2_best"],
            "reference_run":str(BASE.relative_to(ROOT)),"cases":cases,"audits":audits,
            "prediction_diagnostic":{"manual_seed":4217,"max_trajectories_per_task":12,
                "windows_per_trajectory":2,"splits":["validation","test"],"horizon":10,
                "changed_patch_threshold_normalized_rms":0.1},
            "new_policy_episodes":len(cases),"output_limit_decimal_bytes":60_000_000_000,
            "repository_limit_decimal_bytes":1_000_000_000_000,
            "limits":["Stage2 versus Stage1 includes extra optimization and changed LR; not a prediction-loss ablation.",
                      "Frozen test cases already observed in Stage1; this is exploratory, not a fresh confirmatory benchmark.",
                      "Different execution GPU from some baseline episodes; identical initial policy inputs required."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN / "worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol=frozen_protocol()
        path=RUN / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("frozen stage-two evaluation protocol changed")
        save_report(path,protocol)
        report_path=RUN / "report.json"
        if report_path.exists() and not args.resume:
            raise ValueError("existing evaluation requires explicit --resume")
        report=json.loads(report_path.read_text()) if report_path.exists() else {
            "manual_seed":3217,"checkpoint":CHECKPOINT,"checkpoint_updates":1500,
            "policy":"stage2_best","episodes":[],"complete":False,
            "started_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid()}
        if report["complete"]:
            return
        report.pop("error",None)
        def save():
            report["updated_utc"]=datetime.now(timezone.utc).isoformat()
            save_report(report_path,report)
        try:
            used=int(subprocess.check_output(["du","-sb",str(ROOT)],text=True).split()[0])
            if used+protocol["output_limit_decimal_bytes"] > protocol["repository_limit_decimal_bytes"]:
                raise RuntimeError("stage-two assessment would exceed 1 TB reserved budget")
            while True:
                pids=subprocess.check_output(["nvidia-smi","-i","1","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                free=int(subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if not pids and free>=18432:
                    break
                report["phase"]="waiting_for_idle_gpu1";save();time.sleep(30)
            mapping=verify_cuda_target(1);report["device"]=mapping
            config=load_config(ROOT / "experiments/stage2_policy.yaml")
            torch.set_num_threads(4);set_manual_seed(config.manual_seed)
            report["phase"]="loading_checkpoint";save()
            backend=NativeVLABackend.from_pretrained(config,device="cuda:0",local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO")
            for parameter in backend.model.parameters():
                if parameter.requires_grad:
                    parameter.data=parameter.data.float()
            predictor=make_predictor(backend)
            payload=load_checkpoint(ROOT / CHECKPOINT,backend,predictor)
            state=backend.model.state_dict()
            if payload["updates"]!=1500 or payload["predictor"] is None:
                raise ValueError("wrong stage-two checkpoint")
            if any(not torch.equal(state[k].cpu(),v) for k,v in payload["adapters"].items()):
                raise ValueError("stage-two policy did not restore exactly")
            if any(not torch.equal(predictor.state_dict()[k].cpu(),v) for k,v in payload["predictor"].items()):
                raise ValueError("stage-two predictor did not restore exactly")
            report["checkpoint_audit"]={"exact":True,"policy_tensors":len(payload["adapters"]),
                "predictor_tensors":len(payload["predictor"]),"initialization":payload["initialization"]}
            del payload
            backend.train(False);predictor.eval();save()
            completed={r["case_id"] for r in report["episodes"]}
            for kind in ("H","P","S"):
                report["phase"]=kind;save()
                for case in protocol["cases"]:
                    if case["kind"]!=kind or case["case_id"] in completed:
                        continue
                    used=int(subprocess.check_output(["du","-sb",str(RUN)],text=True).split()[0])
                    if used+200_000_000>protocol["output_limit_decimal_bytes"]:
                        raise RuntimeError("stage-two output budget exhausted")
                    destination=RUN / "stage2_best" / case["case_id"] / case["episode"]
                    if destination.exists():
                        parent=RUN / "interrupted" / case["case_id"];parent.mkdir(parents=True,exist_ok=True)
                        attempt=1
                        while (parent / f"{case['episode']}.attempt{attempt}").exists():attempt+=1
                        destination.rename(parent / f"{case['episode']}.attempt{attempt}")
                    result=rollout(backend,"stage2_best",case["case_id"],case["episode"],RUN,
                        case["manual_seed"],case["max_policy_steps"],diagnostic_case=ROOT / case["case_file"],
                        physical_gpu=1,reference_output=ROOT / case["reference_output"])
                    result.update(case_id=case["case_id"],kind=kind,condition=case["condition"],
                        layout=case["layout"],task_index=case["task_index"],
                        our_training_split=case["our_training_split"],device=mapping)
                    report["episodes"].append(result);save();print(json.dumps(result),flush=True)
                    if len(report["episodes"])%16==0:
                        subprocess.run([str(ROOT / ".venv-sim/bin/python"),str(ROOT / "scripts/summarize_stage2_assessment.py")],cwd=ROOT,check=True)
                if kind=="H":
                    report["phase"]="prediction_diagnostic";save()
                    from diagnose_stage2_prediction import diagnose
                    diagnose(backend,predictor,RUN)
                subprocess.run([str(ROOT / ".venv-sim/bin/python"),str(ROOT / "scripts/summarize_stage2_assessment.py")],cwd=ROOT,check=True)
            report["complete"]=True;report["phase"]="complete"
            report["finished_utc"]=datetime.now(timezone.utc).isoformat();save()
            subprocess.run([str(ROOT / ".venv-sim/bin/python"),str(ROOT / "scripts/summarize_stage2_assessment.py")],cwd=ROOT,check=True)
        except Exception as error:
            report["error"]=f"{type(error).__name__}: {error}";save();raise


if __name__ == "__main__":
    main()
