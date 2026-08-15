# Interactive Perception

We study when a frozen VLA should interact with an occluder before acting.

## Method

Given a final-goal prompt, π0.5 samples eight action chunks. A Mondrian
conformal calibrator returns a set over `ACT` and `REMOVE_OCCLUDER`. The router
executes a primitive only when the set is a singleton and the corresponding
physical executor passes an independent reliability gate.

Policy inputs are the stock LIBERO `agentview`, wrist RGB, robot state, and
prompt. Simulator joints and the BEV camera are evaluation-only.

## T01

T01 copies the stock LIBERO middle-drawer layout. Butter is hidden in the
closed middle layer and a basket is added to the table.

| Test | Result |
|---|---:|
| Stock drawer control | 30/30 |
| Hidden butter added | 30/30 |
| Hidden butter + basket | 30/30 |
| G4 audit, `ACT` vs `REMOVE_OCCLUDER` | 20/20 |
| Monolithic final prompt: drawer revealed | 0/5 |
| Conformal router: correct route | 100/100 |
| Conformal router: physical reveal | 97/100 |

The reveal rate has a one-sided 95% lower bound of 0.924 and passes the frozen
0.90 requirement. Failures are seeds 110, 127, and 186; all are executor
failures, not routing errors.

This result supports reliable action selection and target reveal. It does not
support full retrieval: `OPEN_CONTAINER → ACT` completes the final butter task
in 0/5 trials. `ROTATE` is also blocked at 0/30.

## Evidence

- [100-trial T01 result](results/capability/t01_conformal_reveal_100seed_v1.json)
- [G4 artifact](results/calibration/semantic_intent_g4_t01_binary_audit_v5.json)
- [G5 executor gate](results/capability/g5_executor_gate_stock_aligned_v2.json)
- [Monolithic control](results/capability/t01_monolithic_screen_5seed.json)
- [Two-stage retrieval control](results/capability/t01_stock_chain_screen_5seed.json)
- [Bi-view demo](results/demos/t01_conformal_reveal_30seed_v1/t01_conformal_reveal_seed090.mp4)

## Run

```bash
bash scripts/serve_pi05.sh
env -u PYTHONPATH ../.conda/envs/ipu/bin/python \
  scripts/run_t01_conformal_reveal.py \
  --seeds 190 191 192 193 194 \
  --output /tmp/t01_reveal.json
```
