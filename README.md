# Prompt-conditioned Interaction Uncertainty (PIU)

We study which prompt-relevant information a frozen VLA is missing, where that
uncertainty is located, and which executable interaction is expected to reduce
it before the task is resumed.

## Method

The base policy is the public `pi05_libero` checkpoint. Its weights and action
decoder remain frozen.

```text
public RGB + prompt + history
  → structured task-fact belief + node uncertainty
  → typed hard-valid candidates
  → learned action-effect and future-belief distributions
  → explicit expected information/task utility
  → semantic subtask → frozen π0.5
  → six public observations → conformal outcome → belief update → replan
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

PIU V0 freezes the π0.5 PaliGemma prefix as its prompt-conditioned visual
encoder and trains small belief, effect, outcome, future-belief, and ranking
heads. Its current `target_location` domain is `visible_workspace`,
`middle_drawer`, `other_unsearched_region`, and `absent`; only the first two
have V0 initial-belief training data. Control memory is deterministic
`(searched_regions, attempt_counts, public outcomes)`. No confidence threshold
decides whether to open the drawer.

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
| v4 OPEN_AND_OBSERVE disposable smoke | 3/3: REVEALED / EMPTY / FAILED |
| PIU V0 learned-head diagnostics | belief 27/30; outcome 20/21 contaminated diagnostic |
| PIU V0 end-to-end disposable smoke | 5/5 routes and outcomes; non-paper |
| Sealed outcome audit | NOT RUN; new seeds 900–999 |
| Final retrieval after reveal | 0/5 |

The PIU smoke covers hidden butter, a same-RGB visible cream-cheese prompt,
already-visible butter, local EMPTY, and a constructed FAILED control. Hidden
butter routes to real π0.5 `OPEN_AND_OBSERVE`, obtains singleton `REVEALED`,
updates the public belief, and replans to a `DIRECT_ACT` semantic handoff.
EMPTY excludes only the middle drawer and FAILED preserves the hypothesis.
`DIRECT_ACT` is not yet physically executed, so this is an information-loop
success, not final retrieval success.

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
| PIU V0 wiring / inference smoke | GO (non-claim) |
| Clean PIU development | NOT-GO / not run |
| RSS method paper | NOT-GO |
| Final product | NOT-GO |

`PARTIAL` always counts as `NOT-GO`.

## Seed and data provenance

The single machine-readable authority is
[`benchmarks/rss_v1/seed_registry.yaml`](benchmarks/rss_v1/seed_registry.yaml).
Historical manifests remain immutable, but their old “future audit” text does
not override this registry.

| Seeds | Current role | State |
|---|---|---|
| 90–189 | legacy prompt router / joint-open test | consumed; not RGB reveal evidence |
| 220–269 | prompt-state train/calibration/development | consumed |
| 280–379 | prompt-state audit + precision extension | consumed |
| 400–459 | legacy action-effect development | consumed |
| 500–599 | obsolete v1 audit allocation | quarantined; not reused |
| 600–619 | outcome prototype train | consumed |
| 620–652 | conformal calibration | consumed |
| 653–659 | v8/v9 diagnosis | contaminated; diagnostic only |
| 660–699 | v9 clean development extension | frozen and untouched |
| 700–799 | final-frame-bug debugging | whole block quarantined |
| 800–899 | later closed-loop validation | frozen; audit-gated; non-paper |
| 900–999 | one-time sealed outcome audit | sealed |
| 1300, 1399 | wiring smoke | disposable; never claim-bearing |
| 1500–1599 | future PIU counterfactual prototype train | untouched |
| 1600–1699 | future PIU conformal calibration | untouched |
| 1700–1799 | future scene-disjoint clean development | untouched |
| 1800–1899 | future scene-disjoint sealed audit | sealed |

The three-class collection and label contract is frozen in
[`outcome_data_protocol_v1.yaml`](benchmarks/rss_v1/outcome_data_protocol_v1.yaml).

## Reproduce

Run the dependency-gated experiment:

```bash
bash scripts/run_rss_experiment_ladder.sh preflight
bash scripts/run_rss_experiment_ladder.sh smoke
bash scripts/run_rss_experiment_ladder.sh development
```

Run the local PIU V0 path:

```bash
../.conda/envs/ipu/bin/python scripts/build_piu_v0_dataset.py
../.conda/envs/ipu/bin/python scripts/train_piu_v0.py --help
../.conda/envs/ipu/bin/python scripts/run_piu_v0_smoke.py --help
```

The recorded five-case smoke itself was run with
`EXPERIMENT_GPU_INDEX=0 bash scripts/run_piu_v0_smoke.sh --output
results/smoke/piu_v0_end_to_end_v3_seed1399.json`.

This PC has one physical GPU at index 0. The generic preflight rejects unknown
compute processes and permits the current-user RustDesk process only when
explicitly enabled. `LAB_SERVER_MODE=1` still hard-requires physical GPU1 and
the lab Server User Guide.

The audit is intentionally explicit and can run only after development passes:

```bash
ALLOW_SEALED_AUDIT=1 bash scripts/run_rss_experiment_ladder.sh audit
```

See [method](docs/RSS_METHOD_V1.md), [RSS gates](benchmarks/rss_v1/gates.yaml),
[experiment](benchmarks/rss_v1/experiment_v1.yaml),
[PIU training protocol](docs/PIU_TRAINING_V0.md),
[PIU data protocol](benchmarks/piu_v0/data_protocol.yaml),
[PIU main matrix](benchmarks/piu_v0/main_experiment_matrix.yaml),
[PIU strict gates](benchmarks/piu_v0/gates.yaml),
[PIU code ownership](benchmarks/piu_v0/code_ownership.yaml),
[seed/data provenance registry](benchmarks/rss_v1/seed_registry.yaml),
[threshold register](docs/THRESHOLD_REGISTER.md), and
[temporal-label failure](results/T01_TEMPORAL_LABEL_BUG.md).
