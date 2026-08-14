# Interactive Perception

We study when a frozen vision-language-action policy should gather information
before committing to a task action.

## Method

The controller maintains beliefs over four states: visible, recoverable by a
viewpoint change, recoverable by physical interaction, and absent. Repeated
policy samples are grouped by semantic intent. Split conformal prediction
calibrates the intent set; a risk router chooses `ACT`, `MOVE_CLOSER`,
`ROTATE`, `REMOVE_OCCLUDER`, or `NOT_FOUND`.

Simulator state and segmentation are evaluation-only. Calibration scenes are
disjoint from test scenes.

## Gates

| Gate | Result | Decision |
|---|---:|---|
| Stock π0.5/LIBERO control | 5/5 | GO |
| T01 direct middle-layer opening | 15/30; 95% lower bound 0.339 | NOT-GO (required 0.90) |
| Four-intent conformal calibration | 40/40 coverage; mean set size 1.0 | GO for intent prediction only |
| Task-facing wrist camera, visible control | 0/5 | NOT-GO |
| `MOVE_CLOSER`, task-facing camera | 0/30 | NOT-GO |
| `ROTATE`, task-facing camera | 0/30 | NOT-GO |

Changing only the wrist-camera extrinsics breaks a visible control. Therefore
the information-action failures do not establish a missing π0.5 skill; they
establish that this camera protocol is out of distribution. The next valid
ability test must preserve stock camera extrinsics and place the scene inside
the native wrist field of view.

T01 contact diagnostics show a more specific failure. Failed seeds reach the
handle region (1.9 cm), make sustained two-finger contact, and produce large
contact forces (323–464 N), but generate almost no opening-axis joint force
(less than 0.001 N). Their near-handle motion points away from the opening axis
on 55–70% of steps. The dominant failure is pull direction, not failure to
reach the handle.

## Evidence

- [Calibration artifact](results/calibration/semantic_intent_g4_v3.json)
- [Information-action ability test](results/capability/information_actions_task_facing_30seed.json)
- [T01 contact mechanics](results/diagnostics/t01_contact_mechanics_force_3seed.json)
- [MOVE_CLOSER demo](results/demos/information_actions_v3/IE02_resolution_only_seed000.mp4)
- [ROTATE demo](results/demos/information_actions_v3/IE03_orientation_only_seed000.mp4)
- [Pipeline](docs/PIPELINE_V04.md)
- [Human experiments](docs/HUMAN_EXPERIMENTS.md)

## Run

```bash
bash scripts/serve_pi05.sh
env -u PYTHONPATH ../.conda/envs/ipu/bin/python \
  scripts/run_pure_pi05_scenario_sr.py --variant capability
```
