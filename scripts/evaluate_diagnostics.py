"""Run audited geometry/history cases with native and the fixed stage-one best."""

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone

import torch
from evaluate_stage1 import ROOT, inside, rollout, save_report
from gpu_devices import verify_cuda_target

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-device-migration", action="store_true")
    parser.add_argument("--pause-after-native", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--storage-root")
    parser.add_argument("--storage-limit-gb", type=float)
    parser.add_argument(
        "--modes", nargs="+", default=["native", "best"],
        choices=["native", "best", "best_no_geometry", "best_current"],
    )
    parser.add_argument("--kinds", nargs="+", choices=["G", "H", "S", "P"], default=["G", "H"])
    args = parser.parse_args()
    manifest, output = inside(args.manifest), inside(args.output)
    storage_root = inside(args.storage_root) if args.storage_root else None
    if (storage_root is None) != (args.storage_limit_gb is None):
        raise ValueError("storage root and decimal-GB limit must be provided together")
    if storage_root is not None and (not output.is_relative_to(storage_root) or args.storage_limit_gb <= 0):
        raise ValueError("invalid storage budget")
    spec = json.loads(manifest.read_text())
    audit = json.loads(inside(args.scene_audit).read_text())
    if (
        audit.get("passed") is not True
        or audit["manifest"] != str(manifest.relative_to(ROOT))
        or audit["cases"] != len(spec["cases"])
    ):
        raise ValueError("matching scene/causality audit must pass first")
    if args.modes[0] != "native" or len(set(args.modes)) != len(args.modes):
        raise ValueError("native must run first and modes must be unique")
    mapping = verify_cuda_target(args.physical_gpu)
    config = load_config(ROOT / "experiments/stage1_policy.yaml")
    output.mkdir(parents=True, exist_ok=args.resume)
    report = {
        "kind": "geometry/history controlled diagnostic",
        "split": spec["split"],
        "manual_seed": spec["manual_seed"],
        "device": mapping,
        "manifest": str(manifest.relative_to(ROOT)),
        "cases": spec["cases"],
        "checkpoint": "runs/stage1_s17_gpu7/best.pt",
        "checkpoint_updates": 1500,
        "modes": args.modes,
        "interventions": "Complete systems compared using their intended inputs. Optional channel interventions are not retrained ablations.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": [],
        "complete": False,
        "storage_budget": {"root": str(storage_root.relative_to(ROOT)), "decimal_gb": args.storage_limit_gb}
        if storage_root is not None else None,
    }
    if args.resume:
        previous = json.loads((output / "report.json").read_text())
        for key in (
            "manual_seed",
            "manifest",
            "cases",
            "checkpoint",
            "modes",
        ):
            if previous[key] != report[key]:
                raise ValueError(f"resume changed diagnostic setting: {key}")
        if previous["complete"]:
            raise ValueError("diagnostic run is already complete")
        if previous["device"] != mapping:
            if not args.allow_device_migration:
                raise ValueError("resume changed device without explicit migration")
            previous.setdefault("device_history", []).append({
                "from": previous["device"], "to": mapping,
                "completed_episodes": len(previous["episodes"]),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            previous["device"] = mapping
        report = previous
        if report.get("storage_budget") != ({"root": str(storage_root.relative_to(ROOT)), "decimal_gb": args.storage_limit_gb}
                                              if storage_root is not None else None):
            raise ValueError("resume changed storage budget")
    report["paused_after_native"] = False
    save_report(output / "report.json", report)
    torch.set_num_threads(4)
    set_manual_seed(config.manual_seed)
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda:0", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    for parameter in backend.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    loaded = False
    for mode in report["modes"]:
        if mode != "native" and args.pause_after_native:
            visible = [
                r
                for r in report["episodes"]
                if r["policy"] == "native"
                and r["kind"] == "H"
                and r["condition"] == "visible"
            ]
            report["history_visible_feasibility"] = {
                "episodes": len(visible),
                "successful": sum(r["success"] for r in visible),
                "correct_first_choices": sum(
                    bool(r["first_choice"] and r["first_choice"]["correct"])
                    for r in visible
                ),
                "passed": bool(visible)
                and all(
                    r["success"] and r["first_choice"] and r["first_choice"]["correct"]
                    for r in visible
                ),
            }
            report["paused_after_native"] = True
            save_report(output / "report.json", report)
            print(json.dumps(report["history_visible_feasibility"]), flush=True)
            return
        cases = [
            c
            for c in spec["cases"]
            if c["kind"] in args.kinds
            and not (
                (mode == "best_no_geometry" and c["kind"] != "G")
                or (mode == "best_current" and c["kind"] != "H")
            )
        ]
        completed = {r["case_id"] for r in report["episodes"] if r["policy"] == mode}
        if all(c["case_id"] in completed for c in cases):
            continue
        if mode != "native" and not loaded:
            payload = load_checkpoint(ROOT / report["checkpoint"], backend)
            state = backend.model.state_dict()
            if payload["updates"] != 1500 or any(
                not torch.equal(state[n].detach().cpu(), t)
                for n, t in payload["adapters"].items()
            ):
                raise RuntimeError("fixed best checkpoint did not restore exactly")
            del payload
            loaded = True
        backend.config = (
            replace(config, geometry=False) if mode == "best_no_geometry" else config
        )
        for case in cases:
            if case["case_id"] in completed:
                continue
            if storage_root is not None:
                used = int(subprocess.check_output(["du", "-sb", str(storage_root)], text=True).split()[0])
                # Reserve at least 200 MB for one rollout before beginning it.
                if used + 200_000_000 > args.storage_limit_gb * 1_000_000_000:
                    raise RuntimeError("evaluation output budget reached; results retained")
            destination = output / mode / case["case_id"] / case["episode"]
            if args.resume and destination.exists():
                # Keep partial rollouts as evidence; restart this case from its seed.
                parent = output / "interrupted" / mode / case["case_id"]
                parent.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while (parent / f"{case['episode']}.attempt{attempt}").exists():
                    attempt += 1
                archived = parent / f"{case['episode']}.attempt{attempt}"
                destination.rename(archived)
                report.setdefault("interrupted_rollouts", []).append(str(archived.relative_to(ROOT)))
                save_report(output / "report.json", report)
            result = rollout(
                backend,
                mode,
                case["case_id"],
                case["episode"],
                output,
                case["manual_seed"],
                case["max_policy_steps"],
                diagnostic_case=manifest.parent / f"{case['case_id']}.json",
                physical_gpu=args.physical_gpu,
            )
            result.update(
                device=mapping,
                case_id=case["case_id"],
                layout=case["layout"],
                kind=case["kind"],
                condition=case["condition"],
                target_index=case.get("target_index"),
            )
            report["episodes"].append(result)
            save_report(output / "report.json", report)
            print(json.dumps(result), flush=True)
    report["completed_kinds"] = sorted(
        set(report.get("completed_kinds", [])) | set(args.kinds)
    )
    report["complete"] = set(report["completed_kinds"]) == {
        c["kind"] for c in spec["cases"]
    }
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(output / "report.json", report)


if __name__ == "__main__":
    main()
