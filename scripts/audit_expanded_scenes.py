"""Build candidate fixtures in parallel and exclude invalid complete paired layouts.

No policy is loaded. Every candidate and exclusion remains available for review.
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from audit_diagnostic_scenes import arrays_equal

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def construct(case, manifest, output, gpu):
    destination = output / case["case_id"]
    cache = destination / "takeover.json"
    if cache.exists():
        if read(destination / "scene_audit.json")["case"] != case:
            raise ValueError("cached scene differs from its declaration")
        return read(cache)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), PYOPENGL_PLATFORM="egl",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with (output / f"{case['case_id']}.stderr.log").open("a") as errors:
        result = subprocess.run([str(ROOT / ".venv-sim/bin/python"), "-u",
            str(ROOT / "scripts/libero_diagnostic_server.py"), "--case",
            str(manifest.parent / f"{case['case_id']}.json"), "--output", str(destination),
            "--physical-gpu", str(gpu)], input='{"close":true}\n', text=True,
            stdout=subprocess.PIPE, stderr=errors, env=env, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(f"simulator exit {result.returncode}; see {case['case_id']}.stderr.log")
    packets = [json.loads(line.removeprefix("IP_OBSERVATION:")) for line in result.stdout.splitlines()
               if line.startswith("IP_OBSERVATION:")]
    if len(packets) != 1:
        raise ValueError("expected one initial observation")
    write(cache, packets[0])
    return packets[0]


def audit_layout(cases, packets, output):
    kind = cases[0]["kind"]
    reports = [read(output / c["case_id"] / "scene_audit.json") for c in cases]
    ps = [packets[c["case_id"]] for c in cases]
    checks = {
        "case_metadata_exact": all(r["case"] == c for r,c in zip(reports,cases)),
        "task_equal": len({p["task"] for p in ps}) == 1,
        "manual_seed_equal": len({c["manual_seed"] for c in cases}) == 1,
        "not_already_solved": all(not p["evaluation_success"] for p in ps),
        "initial_feedback_empty": all(not p["applied_actions"] for p in ps),
        "calibration_passed": all(v["rays"] >= 6 and v["p90_depth_error_m"] <= .015
                                   for r in reports for v in r["calibration"].values()),
    }
    states = np.asarray([r["physical_state"] for r in reports])
    private = [r["private_evaluation"] for r in reports]
    excluded = private[0].get("ignored_state_indices", [])
    if kind in {"H", "P"}:
        checks["only_target_state_excluded"] = len(excluded) == 13 and all(
            p["ignored_state_indices"] == excluded and p["target"] == c["target_object"]
            for p,c in zip(private,cases))
        states = np.delete(states, excluded, axis=1)
        checks["target_not_deeply_penetrating"] = all(p["target_contacts"]["min_contact_distance_m"] >= -.002 for p in private)
    checks["non_target_physics_equal"] = bool(np.allclose(states, states[0], atol=1e-10, rtol=0))
    if kind == "H":
        checks["balanced_worlds_and_visibility"] = len(cases) == 4 and {
            (c["target_index"], c["condition"]) for c in cases} == {(i,v) for i in (0,1) for v in ("visible","hidden")}
        checks["history_is_real_and_aligned"] = all(p["step"] == 40
            and [o["step"] for o in p["initial_history"]["observations"]] == [30,35]
            and [a["step"] for a in p["initial_history"]["applied_actions"]] == list(range(30,40)) for p in ps)
        hidden = [p for p,c in zip(ps,cases) if c["condition"] == "hidden"]
        a,b = hidden
        checks["hidden_inputs_identical"] = arrays_equal(ROOT / a["path"], ROOT / b["path"])
        checks["actual_past_controls_identical"] = a["initial_history"]["applied_actions"] == b["initial_history"]["applied_actions"]
        checks["past_visual_evidence_differs"] = not arrays_equal(
            ROOT / a["initial_history"]["observations"][-1]["path"],
            ROOT / b["initial_history"]["observations"][-1]["path"], ["agent_rgb","wrist_rgb"])
        checks["target_drift_within_one_cm"] = all(np.linalg.norm(
            np.asarray(p["target_final_position"])-p["target_positions"][c["target_index"]]) <= .01
            for p,c in zip(private,cases))
        # Recheck exact visible/hidden history, not just the hidden pair.
        checks["same_history_when_only_visibility_changes"] = True
        for target in (0,1):
            pair = [p for p,c in zip(ps,cases) if c["target_index"] == target]
            checks["same_history_when_only_visibility_changes"] &= all(arrays_equal(ROOT / x["path"], ROOT / y["path"])
                for x,y in zip(pair[0]["initial_history"]["observations"], pair[1]["initial_history"]["observations"]))
    else:
        checks["initial_step_zero"] = all(p["step"] == 0 for p in ps)
        normal_index = next(i for i,c in enumerate(cases) if c["condition"] == "normal")
        baseline = ROOT / ps[normal_index]["path"]
        checks["robot_and_intrinsics_unchanged"] = all(arrays_equal(baseline, ROOT / p["path"],
            ["states","agent_K","wrist_K","wrist_T_world"]) for p in ps)
        if kind == "S":
            checks["four_conditions"] = {c["condition"] for c in cases} == {"normal","camera","lighting","combined"}
            checks["camera_changes_match_plan"] = all(
                arrays_equal(baseline, ROOT / p["path"], ["agent_T_world"]) != bool(c["yaw_degrees"] or c["pitch_degrees"])
                for p,c in zip(ps,cases))
            checks["lighting_preserves_depth"] = all(arrays_equal(baseline, ROOT / p["path"], ["agent_depth","wrist_depth"])
                for p,c in zip(ps,cases) if not c["yaw_degrees"] and not c["pitch_degrees"])
        else:
            checks["three_conditions"] = {c["condition"] for c in cases} == {"normal","minus_y","plus_y"}
            checks["target_translation_matches_plan"] = all(np.allclose(np.asarray(p["target_after"])-p["target_before"],
                c["target_offset_m"], atol=1e-10, rtol=0) for p,c in zip(private,cases))
            # LIBERO scenes use different world/table origins. Validate a local
            # displacement envelope, never an assumed absolute table height.
            checks["target_local_placement_within_budget"] = all(
                np.max(np.abs(np.asarray(p["target_after"])[:2]-p["target_before"][:2])) <= .04+1e-10
                and abs(p["target_after"][2]-p["target_before"][2]) <= 1e-10 for p in private)
            checks["all_camera_poses_unchanged"] = all(arrays_equal(baseline, ROOT / p["path"], ["agent_T_world","wrist_T_world"]) for p in ps)
        checks["changed_scene_has_changed_pixels"] = all(not arrays_equal(baseline, ROOT / p["path"], ["agent_rgb","wrist_rgb"])
            for p,c in zip(ps,cases) if c["condition"] != "normal")
    return {"layout": cases[0]["layout"], "task_index": cases[0]["task_index"], "kind": kind,
            "passed": bool(all(checks.values())), "checks": {k:bool(v) for k,v in checks.items()},
            "non_target_state_max_error": float(np.max(np.abs(states-states[0]))),
            "exclusion_reasons": [k for k,v in checks.items() if not v]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest, output = Path(args.manifest).resolve(), Path(args.output).resolve()
    if not manifest.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or not 1 <= args.workers <= 2:
        raise ValueError("invalid repository paths or worker budget")
    spec = read(manifest)
    if spec["physical_gpu"] != args.physical_gpu:
        raise ValueError("GPU differs from the frozen assignment")
    output.mkdir(parents=True, exist_ok=True)
    packets, failures = {}, {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(construct,c,manifest,output,args.physical_gpu):c for c in spec["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            try:
                packets[case["case_id"]] = future.result()
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
                failures[case["case_id"]] = str(error)
            write(output / "progress.json", {"cases_created": len(packets), "case_failures": failures,
                  "candidate_cases": len(spec["cases"]), "physical_gpu": args.physical_gpu})
            print(json.dumps({"case": case["case_id"], "created": case["case_id"] in packets}), flush=True)
    audits, accepted = [], []
    for layout in sorted({c["layout"] for c in spec["cases"]}):
        cases = [c for c in spec["cases"] if c["layout"] == layout]
        failed = [c["case_id"] for c in cases if c["case_id"] in failures]
        if failed:
            result = {"layout": layout, "task_index": cases[0]["task_index"], "kind": cases[0]["kind"],
                      "passed": False, "exclusion_reasons": ["fixture_construction_failed"], "failed_cases": failed}
        else:
            result = audit_layout(cases,packets,output)
        audits.append(result)
        if result["passed"]:
            accepted.extend(cases)
    selected = output / "eligible"
    selected.mkdir(exist_ok=True)
    selected_spec = {**spec, "cases": accepted, "status": "model-blind audit complete",
                     "candidate_manifest": str(manifest.relative_to(ROOT))}
    for case in accepted:
        write(selected / f"{case['case_id']}.json", case)
    write(selected / "manifest.json", selected_spec)
    report = {"manual_seed": spec["manual_seed"], "manifest": str((selected / "manifest.json").relative_to(ROOT)),
              "cases": len(accepted), "candidate_cases": len(spec["cases"]), "audits": audits,
              "case_failures": failures, "passed": bool(accepted), "selection_used_policy_scores": False,
              "limits": "Geometric/causal validity only; do not claim reachable or solvable merely because these checks pass."}
    write(output / "report.json", report)
    print(json.dumps({"eligible_cases":len(accepted), "excluded_layouts":sum(not a["passed"] for a in audits)}), flush=True)


if __name__ == "__main__":
    main()
