# Interactive Perception

We study when a frozen vision-language-action policy should gather information
before acting.

## Method

The controller separates semantic intent uncertainty from physical executor
reliability. Repeated π0.5 action chunks produce a conformal intent set over
`ACT`, `MOVE_CLOSER`, `ROTATE`, and `REMOVE_OCCLUDER`. A primitive is executed
only when its calibrated set and its independent capability gate both pass.

Simulator state and segmentation are evaluation-only. Calibration and test
scenes are disjoint.

## Stock-aligned benchmark

The new benchmark copies the original LIBERO ketchup task. Cameras, robot
reset, controller, prompt, objects, and goal remain unchanged. Only existing
distractors are moved onto the two policy-camera rays.

| Condition | Success |
|---|---:|
| Original visible task | 5/5 |
| One blocker | 5/5 |
| Two-camera occlusion | 2/5 |

Removing both blockers restores target pixels in both cameras on all three
validation seeds.

## Gates

| Gate | Result | Decision |
|---|---:|---|
| Stock π0.5 reproduction | 100/100 | GO |
| `MOVE_CLOSER` | 30/30; 95% lower bound 0.905 | GO |
| `ROTATE` | 0/30 | NOT-GO |
| Open hidden container | 15/30; lower bound 0.339 | NOT-GO |
| G4 marginal conformal | coverage 0.825 | NOT-GO |
| G4 class-conditional audit | overall 0.90; worst class 0.80 | NOT-GO |
| G5 complete executor set | 2/4 primitives authorized | NOT-GO |

π0.5 reliably grasps, brings an object closer, and releases it. It does not
produce the requested wrist rotation. `ACT`, `MOVE_CLOSER`, and `ROTATE` also
generate overlapping initial action chunks, so trajectory statistics alone do
not identify semantic intent reliably.

## Evidence

- [G4/G5 report](results/G4_G5_STOCK_ALIGNED.md)
- [G4 independent audit](results/calibration/semantic_intent_g4_stock_aligned_mondrian_audit_v4.json)
- [G5 executor gate](results/capability/g5_executor_gate_stock_aligned_v1.json)
- [30-trial unit actions](results/capability/stock_aligned_unit_actions_30episode.json)
- [Two-camera occlusion validation](results/validation/stock_aligned_v1.json)
- [MOVE_CLOSER demo](results/demos/stock_aligned_units_30/move_closer_episode000.mp4)
- [ROTATE failure demo](results/demos/stock_aligned_units_30/rotate_episode000.mp4)

## Run

```bash
bash scripts/serve_pi05.sh
env -u PYTHONPATH ../.conda/envs/ipu/bin/python \
  scripts/run_stock_aligned_unit_actions.py \
  --episodes 0 1 2 3 4 --output /tmp/unit_actions.json
```
