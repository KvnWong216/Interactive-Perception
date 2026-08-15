# T01 Conformal Reveal

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
| Physical target reveal | 97/100 | 0.924 | GO |
| Monolithic reveal | 0/5 | -- | NOT-GO |
| Full routed retrieval | 0/5 | -- | NOT-GO |

The three reveal failures are seeds 110, 127, and 186. G4 selected
`REMOVE_OCCLUDER` in all three, but the executor did not move the drawer.

The result supports the information-acquisition stage only. It must not be
reported as end-to-end task success.
