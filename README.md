# Interactive Perception

Can a frozen vision-language-action policy change the scene when the task
cannot be solved from the current image?

## Benchmark

Six LIBERO-derived tasks test hidden targets, removable occluders, multiple
containers, absence, and severe clutter. Simulator state and segmentation are
used only for evaluation. The policy receives the stock LIBERO RGB cameras and
robot state.

## Method

The method separates four states: visible, recoverable by viewpoint change,
recoverable by manipulation, and absent. It compares the risk of acting,
acquiring information, and stopping, then sends a short instruction to a frozen
π0.5 policy and observes again.

Repeated action chunks are grouped into coarse semantic intents. A
split-conformal calibrator turns their evidence into an intent prediction set;
large sets trigger information gathering instead of treating trajectory spread
as semantic uncertainty. The router is not paper-ready until this calibrator is
fit on held-out data.

## Current result

The stock checkpoint and camera path pass their controls. On the custom T01
scene, pure π0.5 obtains:

| Instruction | Success |
|---|---:|
| Open the middle layer | 15/30 |
| Search, then place the butter | 0/5 |
| Final goal only | 0/5 |

The direct executor fails the preregistered 0.90 reliability gate: its
one-sided 95% lower bound is 0.339. Successful runs move the middle layer by
0.142–0.149 joint units; failed runs move it by at most 0.000007. Thus the
failure is not an endpoint artifact. This is a context-sensitive transfer
failure, not proof of broad overfitting.

Semantic intent coverage and physical skill reliability are separate gates. A
singleton conformal intent set is executed only when the corresponding
capability lower confidence bound also passes a preregistered requirement.

## Evidence

- [Debug and gate report](results/PURE_PI05_DEBUG.md)
- [Capability demo](results/demos/T01_multi_drawer_search_capability_seed000.mp4)
- [Final-goal demo](results/demos/T01_multi_drawer_search_implicit_seed000.mp4)
- [Pipeline](docs/PIPELINE_V04.md)
- [Remaining human experiments](docs/HUMAN_EXPERIMENTS.md)
- [Environment](env/README.md)

## Run

```bash
bash scripts/serve_pi05.sh
env -u PYTHONPATH ../.conda/envs/ipu/bin/python \
  scripts/run_pure_pi05_scenario_sr.py --variant implicit
```

G4-v1 is frozen and passes held-out validation for the LIBERO binary intent
scope (`ACT` versus `REMOVE_OCCLUDER`): coverage 20/20 at error rate 0.1, with
mean set size 1.0. It does not cover `NOT_FOUND`, `ROTATE`, or `MOVE_CLOSER`,
and it does not guarantee physical task success. The main experiment is blocked
until the missing classes are calibrated and the drawer primitive is replaced
by an executor (or retry protocol) that passes the 0.90 gate.
