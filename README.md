# Prompt-Aligned Interactive Perception

We study when a frozen VLA should acquire information before acting.

## Method

The base policy is the public `pi05_libero` checkpoint. Its weights and action
decoder remain frozen.

```text
RGB + prompt
  → typed target-state belief
  → probabilistic task-progress automaton
  → conformal plausible set
  → calibrated action effects
  → expected-risk planning
  → ACT / information action / NOT_FOUND / SAFE_STOP
  → observe and update
```

Policy inputs are the stock LIBERO `agentview`, wrist RGB, robot state, and
prompt. Simulator joints, segmentation, hidden poses, task predicates, and BEV
images are evaluator-only.

We define action-resolvable uncertainty as the expected task risk removed by
an information action after its cost and measured failure rate are included:

\[
U_{\mathrm{res}}(a\mid b)=V_{\mathrm{stop}}(b)-
\left[c(a)+\mathbb{E}_{y}V(b^{a,y})\right].
\]

The planner operates on the product of world belief and task-progress belief.
The automaton contains only generic rules such as “commit only after sufficient
evidence” and “return `NOT_FOUND` only after search exhaustion”; it never says
which drawer to open. The system interacts only when its expected risk reduction
is positive. No confidence threshold decides whether to open the drawer.

## Current evidence

| Test | Result |
|---|---:|
| Stock π0.5 reproduction | 100/100 |
| T01 drawer opening (joint only) | 97/100 |
| T01 target visible after opening | 57/60 |
| T01 empty layer visually certified | 56/60 |
| Open drawer at stock observation pose (diagnostic) | 60/60 |
| Return to observation pose from perturbations (diagnostic) | 30/30 |
| Evaluator wrapper preserves policy packet | 20/20 |
| Hidden-target belief | 97/100 |
| Same RGB, visible-target prompt | 100/100 |
| Same prompt, target already visible | 99/100 |
| Target-observability risk routing | 296/300 |
| Always-ACT baseline | 200/300 |
| Drawer-state rule | 200/300 |
| Final retrieval after reveal | 0/5 |

The 97/100 result proves joint motion, not information acquisition. Under the
policy-camera endpoint, the one-sided lower bounds are 0.876 for `REVEALED` and
0.854 for `EMPTY`; both fail the frozen 0.90 requirement. The best paired-RGB
critic reaches 20/21 coverage but only 15/21 singleton-correct on a development
diagnostic. Open-drawer retrieval is separately 0/5. The strict closed loop is
therefore blocked.

The measured failure is post-open self-occlusion. `OPEN_AND_OBSERVE` now
combines frozen π0.5 opening with a proprioceptive return to the stock
observation pose. Its return controller passes a 30/30 simulator diagnostic,
but the full composite action has not yet been run and is not authorized.

| Milestone | Decision |
|---|---:|
| T01 decision prototype | GO |
| RSS method paper | NOT-GO |
| Final product | NOT-GO |

`PARTIAL` always counts as `NOT-GO`.

## Reproduce

Run the immutable paired-RGB development diagnostics. Do not run the audit
until a development critic passes:

```bash
bash scripts/run_t01_action_effect_pipeline.sh development
bash scripts/run_t01_action_effect_pipeline.sh development-v2
bash scripts/run_t01_action_effect_pipeline.sh development-v3
bash scripts/run_t01_open_and_observe_pipeline.sh smoke
```

Render the strict decisions:

```bash
../.conda/envs/ipu/bin/python scripts/summarize_rss_gate.py
../.conda/envs/ipu/bin/python scripts/summarize_final_product_gate.py
```

See [method](docs/RSS_METHOD_V1.md), [RSS gates](benchmarks/rss_v1/gates.yaml),
[final-product gates](benchmarks/final_product_v1/gates.yaml), and the
[weekly slide kit](docs/SLIDES_WEEK_2026-08-16.md).
