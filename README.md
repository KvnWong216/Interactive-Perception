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
| Hidden-target belief | 97/100 |
| Same RGB, visible-target prompt | 100/100 |
| Same prompt, target already visible | 99/100 |
| Target-observability risk routing | 296/300 |
| Temporal outcome critic v8 | NOT-GO: FAILED 7/7, REVEALED 7/9, EMPTY 4/5 |
| Patch-resolvable v9 diagnostic | 20/21; clean extension still required |
| Sealed outcome audit | NOT RUN; new seeds 900–999 |
| Final retrieval after reveal | 0/5 |

The v7 outcome head is invalidated. It checked only the final frame: debug seed
770 saw the target during opening and lost it after return. Version 3 therefore
checks all six public history points and requires same-camera-pose visibility
coverage before calling a result `EMPTY`. Its 180-example development set is
now frozen. The v8 head failed development and the sealed audit remains
untouched.

Version 3 also exposed a second evaluator problem: true middle-layer reveals
occupy at least 311 target pixels, while three supposed reveals in the empty
layer contain only 5--7 leaked pixels from the upper layer. Version 9 therefore
uses one pi0.5 visual-token footprint (256 pixels) as an architecture-derived
resolvability definition. Its old-seed diagnostic is 20/21, but those seeds
helped diagnose v8 and cannot certify v9. Untouched seeds 660--699 are reserved
for the clean development extension.

The action reliability lower bound is 0.80; the original 0.90 result is always
reported. Conformal error remains 0.10. These are evidence gates, not online
confidence triggers. Open-drawer retrieval remains 0/5, so target evidence is
not reported as final task success.

| Milestone | Decision |
|---|---:|
| T01 decision prototype | GO |
| RSS method paper | NOT-GO |
| Final product | NOT-GO |

`PARTIAL` always counts as `NOT-GO`.

## Reproduce

Run the dependency-gated experiment:

```bash
bash scripts/run_rss_experiment_ladder.sh preflight
bash scripts/run_rss_experiment_ladder.sh smoke
bash scripts/run_rss_experiment_ladder.sh development
```

The audit is intentionally explicit and can run only after development passes:

```bash
ALLOW_SEALED_AUDIT=1 bash scripts/run_rss_experiment_ladder.sh audit
```

See [method](docs/RSS_METHOD_V1.md), [RSS gates](benchmarks/rss_v1/gates.yaml),
[experiment](benchmarks/rss_v1/experiment_v1.yaml),
[threshold register](docs/THRESHOLD_REGISTER.md), and
[temporal-label failure](results/T01_TEMPORAL_LABEL_BUG.md).
