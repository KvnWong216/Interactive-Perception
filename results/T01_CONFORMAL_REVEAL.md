# T01 Historical Conformal Route to Drawer Opening

The legacy result JSON uses fields named `reveal_success`, but its endpoint is
only the middle-drawer joint threshold. It does not certify that the butter
entered a stock policy camera and is not an RGB outcome-critic result.

## Frozen protocol

- Policy: `pi05_libero`
- Policy cameras: stock `agentview` and wrist camera
- Calibration: 20 examples per class, `alpha=0.1`
- Test: seeds 90--189, disjoint from calibration and audit
- Route set: `ACT`, `REMOVE_OCCLUDER`
- Executor: explicit middle-layer prompt, fixed 300-step horizon
- Controller joint reads: zero

## Results

| Measure | Result | 95% one-sided lower bound | Gate |
|---|---:|---:|---:|
| Correct singleton route | 100/100 | 0.970 | GO |
| Drawer joint opening | 97/100 | 0.924 | GO |
| Monolithic reveal | 0/5 | -- | NOT-GO |
| Full routed retrieval | 0/5 | -- | NOT-GO |

The three drawer-opening failures are seeds 110, 127, and 186. G4 selected
`REMOVE_OCCLUDER` in all three, but the executor did not move the drawer.

The result supports prompt-conditioned routing to a motor option only. It must
not be reported as target reveal, information acquisition, or end-to-end task
success.
