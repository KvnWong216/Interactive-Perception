"""Audit existing model-blind history fixtures and retain complete valid layouts.

CPU only. Never reads policy outcomes; excluded layouts remain in the report.
"""

import json
from pathlib import Path

import numpy as np
from audit_diagnostic_scenes import arrays_equal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/diagnostic_cases/test_single_bowl_H/manifest.json"
FIXTURES = ROOT / "runs/stage1_history_test_scene_audit_s217"
OUTPUT = ROOT / "experiments/diagnostic_cases/stage1_native_best_history_s217"


def read(path):
    return json.loads(path.read_text())


def main():
    spec = read(SOURCE)
    audits, accepted = [], []
    for layout in sorted({c["layout"] for c in spec["cases"]}):
        cases = [c for c in spec["cases"] if c["layout"] == layout]
        assert {(c["target_index"], c["condition"]) for c in cases} == {
            (0, "visible"), (0, "hidden"), (1, "visible"), (1, "hidden")
        }
        assert len(cases) == 4
        scenes = [read(FIXTURES / c["case_id"] / "scene_audit.json") for c in cases]
        packets = [read(FIXTURES / c["case_id"] / "takeover.json") for c in cases]
        excluded = scenes[0]["private_evaluation"]["ignored_state_indices"]
        checks = {
            "fixture_case_exact": all(s["case"] == c for s, c in zip(scenes, cases)),
            "only_target_state_excluded": all(
                s["private_evaluation"]["ignored_state_indices"] == excluded
                and s["private_evaluation"]["target"] == "akita_black_bowl_1"
                for s in scenes
            ) and len(excluded) == 13,
            "same_manual_seed": len({c["manual_seed"] for c in cases}) == 1,
            "historical_steps": all(
                p["step"] == 40
                and [o["step"] for o in p["initial_history"]["observations"]] == [30, 35]
                and [a["step"] for a in p["initial_history"]["applied_actions"]]
                == list(range(30, 40)) for p in packets
            ),
        }
        states = np.delete(np.asarray([s["physical_state"] for s in scenes]), excluded, axis=1)
        checks["non_target_physics_equal"] = bool(np.allclose(states, states[0], atol=1e-10, rtol=0))
        a, b = [p for c, p in zip(cases, packets) if c["condition"] == "hidden"]
        checks["hidden_current_all_arrays_equal"] = arrays_equal(ROOT / a["path"], ROOT / b["path"])
        checks["task_and_controls_equal"] = (
            a["task"] == b["task"] and a["initial_history"]["applied_actions"]
            == b["initial_history"]["applied_actions"]
        )
        checks["past_visual_evidence_differs"] = not arrays_equal(
            ROOT / a["initial_history"]["observations"][-1]["path"],
            ROOT / b["initial_history"]["observations"][-1]["path"], ["agent_rgb"],
        )
        passed = all(checks.values())
        audits.append({"layout": layout, "passed": passed, "checks": checks,
                       "non_target_max_abs_state_difference": float(np.max(np.abs(states-states[0]))),
                       "exclusion_reasons": [k for k, v in checks.items() if not v]})
        if passed:
            accepted.extend(cases)
    if not accepted:
        raise ValueError("no valid paired layouts")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected = {**spec, "cases": accepted, "modes": ["native", "best"],
                "status": "audited_before_policy_evaluation",
                "training_exposure": "Source task and original demo resets belong to stage-one train; shifted bowl and blackout worlds are mechanism probes, not held-out source tasks.",
                "selection": "All layouts passing exact information-isolation checks; no policy scores used.",
                "source_manifest": str(SOURCE.relative_to(ROOT))}
    for c in accepted:
        (OUTPUT / f"{c['case_id']}.json").write_text(json.dumps(c, indent=2) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    report = {"manual_seed": spec["manual_seed"], "passed": True,
              "manifest": str((OUTPUT / "manifest.json").relative_to(ROOT)),
              "cases": len(accepted), "candidate_layouts": len(audits),
              "accepted_layouts": [r["layout"] for r in audits if r["passed"]],
              "source_fixtures": str(FIXTURES.relative_to(ROOT)), "audits": audits,
              "limits": "Model-blind array audit, not a visual-feasibility or policy-success guarantee. One task only."}
    (OUTPUT / "scene_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
