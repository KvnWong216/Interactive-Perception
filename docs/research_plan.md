# Candidate-conditioned calibrated interaction: research plan

This document records the first-round artifacts requested before expensive
training. Literature and novelty evidence are in
[`literature_lineage.md`](literature_lineage.md) and
[`novelty_audit.md`](novelty_audit.md). The architecture decision is
[`ADR-0001`](adr/0001_candidate_conditioned_calibrated_interaction.md).

## 1. Current repository audit

The repository at `e7db12b` is a useful engineering prototype but not the
requested method:

- `vlm_reasoner.py` exposes separate identity, resolution, occlusion, and state
  uncertainty and combines them with fixed `0.60/0.20/0.10/0.10` weights;
- it computes additional expected utility/cost/risk and applies a deterministic
  selector after action effects;
- `NEXT_BEST_VIEW` appears in contracts, prompts, registry, and routing although
  camera motion is excluded;
- Grounding DINO, SAM, DINOv2, and SigLIP are default main-path dependencies;
- the original result table reports final butter retrieval at `0/5` and labels
  the full method NOT-GO.

The old code is therefore pinned under `baselines/heuristic_v0/` at Git tag
`baseline/heuristic-v0`. Existing uncommitted frontend/reproducibility fixes are
preserved, but no new heuristic rules are added.

## 2. Literature lineage

The audit covers capability-grounded planning, VLA interfaces, uncertainty and
calibration, and physical information acquisition. The main result is that the
components are individually established: capability registries (SayCan/CaP),
feedback loops (Inner Monologue/ReAct), candidate-conditioned futures (VLMPC,
CNABU), physical semantic search (SMS, ZS-IP, PROBE), VLA execution (π0.5), and
conformal routing (KnowNo). The paper must contribute and falsify the specific
combination, not rename those components.

## 3. Novelty audit

The gate is GO only for the narrow conjunction documented in
[`novelty_audit.md`](novelty_audit.md). PROBE is the largest late-breaking
threat because it already uses manipulation to answer prompt-relevant hidden
questions. The distinction is continuous task completion plus calibrated
candidate-effect routing, not manipulation-grounded perception itself.

## 4. Final research question

Given a user prompt, recent first-person RGB, public action-observation history,
and the robot's current executable manipulation primitives, can a lightweight
candidate-conditioned decoder predict each action's task-relevant observable
effect and make a calibrated high-level choice that improves full closed-loop
task success when executed by frozen π0.5?

## 5. Falsifiable hypotheses

- H1: effect supervision improves macro route F1 and counterfactual ranking on
  unseen primitive-target compositions over route-only training.
- H2: held-out temperature/conformal calibration reduces false-direct and
  invalid-information-action rates at matched empirical coverage versus top-1.
- H3: proposed routing exceeds a prompted Qwen-VL router with the same images,
  history, candidates, and frozen executor.
- H4: the complete observe-interact-reobserve-direct loop improves final task
  success over direct π0.5, not merely drawer-opening success.
- H5: benefits persist across an articulated container and at least two of
  `REMOVE_OCCLUDER`, `MOVE_CLOSER`, and `ROTATE`; an `OPEN`-only gain falsifies
  the general method claim.

## 6. Chosen minimal architecture

```text
q + I[t-m:t] + h[t] -> frozen shared VLM -> H[t]
                                     candidate a[j] --+
                                                      v
                             one cross-attention interaction decoder
                                      -> c[t,j]
                                      -> factorized effect head
                                      -> route head
                                      -> temperature + conformal sets
                                      -> typed candidate / abstain
                                      -> deterministic text -> frozen pi0.5
                                      -> public RGB/history -> repeat
```

The new code is `src/calibrated_interaction/`. It contains one decoder, two
heads, one calibration subsystem, and one executor adapter. The same original
drawer scenario remains the only Stage 0 replay; no new experiment scenario is
introduced in this revision.

## 7. Rejected alternatives and reasons

| alternative | decision |
|---|---|
| patch the heuristic V0 utility | rejected: cannot learn or calibrate the route and hides category shortcuts |
| binary “information sufficient” then a selector | rejected: splits one decision into redundant classifiers/rules |
| single six-way effect softmax | rejected: empty-region, rejection, and ambiguity reduction can co-occur |
| EDL by default | rejected: fixed mutually exclusive ontology assumption is unmet; no coverage guarantee |
| raw token/action entropy as uncertainty | diagnostic baseline only; not the finite candidate decision |
| semantic entropy as main method | baseline only; costly sampling and language invariance do not ground action effect |
| projected π0.5 token bridge | later ablation only after text exposes a measured bottleneck |
| SAM/Grounding DINO/DINOv2/SigLIP/depth | heuristic baseline or isolated ablation, never default main method |
| NBV/camera motion | rejected by scope and already crowded by SaPaVe/ActiveVLA |

## 8. Mathematical specification

Let `q` be the prompt, `I[t-m:t]` the temporal first-person RGB, `h[t]` public
action-observation history, and `a[j]` candidate `j`. Define

```text
x[t] = {q, I[t-m:t], h[t]}
H[t] = F_VLM(x[t])
c[t,j] = D_int(Embed(a[j]), H[t])
p(y | x[t], a[j]) = product over six calibrated Bernoulli effect factors
p(a[j] | x[t]) = softmax_j(Head_route(c[t,j], p(y | x[t], a[j])))
Gamma_epsilon(x[t]) = {a[j] : 1 - p(a[j] | x[t]) <= calibrated quantile}
```

`c[t,j]` is a hidden representation, not uncertainty. `p(y|...)` is the
candidate effect distribution; `p(a[j]|...)` is decision probability;
`Gamma` is the held-out conformal route set. Conformal coverage is marginal
under exchangeability. Entropy is reported only as a diagnostic.

Stage 1 minimizes binary cross-entropy over observed effect factors. Stage 2
adds route cross-entropy. The effect-loss coefficient has no default: choose it
once by matching development-set gradient norms, freeze it before calibration,
and report a zero-effect-supervision ablation.

## 9. Data schema

```json
{
  "episode_id": "scene42-fork-open",
  "initial_state_id": "scene42-state0",
  "prompt": "Place the requested package in the basket",
  "observation_frames": ["public/pre_agentview.png", "public/pre_wrist.png"],
  "history": [],
  "candidate_actions": [
    {"candidate_id": "open-1", "primitive": "OPEN", "target": "the middle drawer", "reference": null, "reference_region_xyxy": [0.2, 0.3, 0.7, 0.8], "purpose": "inspect inside"}
  ],
  "executed_candidate": "open-1",
  "post_action_frames": ["public/post_agentview.png", "public/post_wrist.png"],
  "effect_labels": {
    "execution_succeeded": true,
    "task_relevant_change": true,
    "ambiguity_reduced": true,
    "target_confirmed": false,
    "candidate_rejected": false,
    "region_confirmed_empty": false
  },
  "route_label": "open-1",
  "task_success": false,
  "privileged_metadata_for_evaluation_only": {"sim_state_digest": "..."}
}
```

`CounterfactualSample.policy_input()` excludes the executed action, post-action
labels, task success, and all privileged metadata. A recursive firewall rejects
semantic IDs, segmentation, simulator state, hidden pose, joint truth, and task
predicates before tokenization.

## 10. Benchmark split design

Counterfactual branches share `initial_state_id` and must remain in one split.
Use group splits, not random episodes:

- train: seen objects/layouts/compositions;
- development: model and loss-weight selection only;
- calibration: temperature and conformal quantiles only;
- test: unseen objects, prompt paraphrases, layouts, occluders, container
  appearances, and primitive-target compositions;
- sealed audit: one-time confirmation after all choices freeze.

The full BenchV0 later needs clear/direct, drawer, fridge/articulated container,
remove occluder, partial clutter, move closer, rotate, and empty/not-found. This
revision deliberately retains only `original_drawer` for wiring, so it cannot
pass the method-paper go gate.

## 11. Baseline matrix

| ID | method | learned route | effects | calibration | executor |
|---|---|---:|---:|---:|---|
| B0 | π0.5 Direct | N | N | N | frozen π0.5 |
| B1 | Prompted VLM Router | N | prompted | N | frozen π0.5 |
| B2 | Heuristic V0 | manual | manual/scored | partial legacy | frozen π0.5 |
| B3 | Route-only decoder | Y | N | N | frozen π0.5 |
| B4 | Route + executed-effect decoder | Y | Y | N | frozen π0.5 |
| B5 | Route + effect + calibration, no spatial bridge | Y | Y | route/belief/effects | frozen π0.5 |
| B6 | Oracle executed effects | evaluator | evaluator | N/A | same executor |
| B7 | Oracle target binding | evaluator mask rendered only after visibility | N | N/A | same executor |
| B8 | Calibrated action-causal binding (ours) | Y | Y | route/belief/effects | frozen π0.5 |

## 12. Main experiment matrix

| experiment | primary question | metrics |
|---|---|---|
| routing | DIRECT vs information action vs STOP | macro F1, per-action P/R, false-direct, unnecessary exploration, invalid action |
| effects | predicts what each fork actually reveals | factor macro F1, NLL, Brier, ECE, OOD, counterfactual ranking |
| calibration | calibrated finite decision sets | coverage, set size, singleton precision, abstention, risk-coverage, NLL/Brier/ECE |
| efficiency | evidence per physical interaction | acquisition rate, effective/ineffective actions, steps, time, success/interaction |
| closed loop | completes final task after information | primitive success, acquisition, recognition, reroute, final task success separately |
| generalization | transfers beyond prompt/scene shortcuts | grouped unseen object/prompt/container/layout/occluder/composition results |

## 13. Ablation matrix

| ablation | necessity tested |
|---|---|
| no prompt | prompt defines relevant uncertainty |
| last frame only | temporal/history evidence |
| no effect head / no effect supervision | candidate effect representation and auxiliary signal |
| no calibration / verbal confidence | statistical calibration versus self-report |
| text versus projected token | π0.5 semantic interface bottleneck |
| fixed versus VLM-grounded candidates | open-vocabulary candidate construction |
| RGB versus RGB-D | whether depth is truly required |
| final pre/post only versus process frames | temporal effect labeling |
| softmax+calibration versus EDL | whether evidential complexity earns its cost |
| heuristic V0 versus learned interaction | removal of category/weight shortcuts |

## 14. Compute estimate

Stage 0 replay and learned-head training are CPU-only after feature extraction.
This workstation has a hard 1500 MiB GPU-use cap and therefore never loads
π0.5 or a replacement Qwen stack locally. Feature collection calls an
identified external `pi05_libero` service and retains the frozen PaliGemma
multimodal-prefix patch tokens used by the downstream policy. Binder and effect
heads train from those cached tensors with `CUDA_VISIBLE_DEVICES=''`; every
external extraction report records checkpoint identity, protocol version, and
public-input hashes. Measure and retain server-side peak allocation separately.

Full π0.5 fine-tuning, multi-node training, video generation, and eight-H100
ActiveVLA-scale experiments are out of scope.

## 15. Implementation plan and current state

| stage | artifact | state |
|---|---|---|
| audit | literature, novelty, ADR | implemented |
| baseline freeze | commit/tag lock and baseline README | implemented; tag created at publish |
| contracts | candidates, capabilities, factorized effects | implemented |
| learned head | one cross-attention decoder + two heads | implemented; shape/gradient test passed |
| calibration | temperature, route LAC, Bonferroni effect LAC | implemented |
| data safety | purpose-specific prospective splits, immutable role assemblers, policy projection, split/leakage checks | implemented |
| executor loop | deterministic text + closed-loop controller | implemented |
| Stage 0 trace | original drawer fixture replay | passed: OPEN -> reobserve -> DIRECT; wiring evidence only |
| VLM feature extraction | external frozen PaliGemma full-prefix protocol | implemented and contract-tested; real cache pending endpoint |
| counterfactual collection | calibrated public execution plan, exact same-state candidate-fork collector, causal matrix exporter, and exact role-matrix assembler | implemented and contract-tested; ineligible candidates stay in the route matrix with masked effects; real branches pending certificates/endpoint |
| training | binder, isolated binder calibration, effect/route ablations, isolated effect calibration | blocked by data, intentionally not run; binder/effect calibration groups are disjoint |
| live full loop | B3/B4/B5/B8 hash-chained runner | implemented and dry-run verified; physical run blocked by checkpoint/certificates/endpoint |
| oracle columns | B6 executed-effect trace and same-source B7 dynamic target-marker full loop | implemented and contract-tested; real tree/style selection/endpoint pending |
| formal oracle causal gate | pilot-sized disjoint state freeze, qualified attempted OPEN, paired same-state oracle/raw execution, single-use receipts, and exact analysis | implemented and contract-tested; real pilot plan/certificate/cohort/endpoint pending |
| empirical orchestration | schema/hash/split-validating stage DAG with terminal negative outcomes | implemented; current first unblocked stage is the external authority/endpoint contract |
| frozen B2 adapter | exact tag inference attestation and one-decision episode projection | implemented and contract-tested; external legacy-model inference pending |

One-command Stage 0 validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra learned --extra vlm --extra dev pytest -q
uv run python scripts/pipeline/run_calibrated_replay.py \
  --output runs/original_drawer_calibrated_replay.json
```

Generated outputs are immutable. On failure, choose a new run path; do not
overwrite evidence. Calibration artifacts record split ID, sample count,
alpha, ontology, and the non-guarantee of task success.

## 16. Risk register

| risk | signal | mitigation |
|---|---|---|
| novelty collision with PROBE/ZS-IP | reviewers reduce claim to “VLM explores” | lead with calibrated candidate effects + continuous task completion; cite directly |
| shortcut labels | prompted router matches learned head | group splits, prompt swaps, counterfactual same-state forks |
| visually inaccessible supervision | good train, random OOD | audit each label from public pre/post RGB; privileged labels evaluator-only |
| effect factors do not help route | route-only B3 equals route+effect B4 | remove effect head; benchmark/calibrated baseline paper |
| conformal sets too large | high coverage, unusable autonomy | report set size/risk-coverage; improve data/model, never hide ambiguity |
| π0.5 cannot execute primitive | low primitive success | capability-specific validation; report separately, do not blame router |
| one drawer overfit | all gain from OPEN | no method claim until two enrichment primitives + articulated container pass |
| VLM memory/latency | OOM or slow loop | precompute frozen tokens, gradient checkpoint only if measured necessary |
| legacy/user changes overwritten | dirty worktree collision | new package only; preserve and test existing diffs |

## 17. Go / no-go criteria

Position as a method paper only if effect supervision improves unseen-composition
routing; calibration reduces false-direct/invalid interactions; proposed beats
prompted VLM; the full loop beats direct π0.5; gains use no extra privileged
input or stronger backbone; more than `OPEN` contributes; an articulated
container and two enrichment primitives pass; and the novelty comparison to
PROBE, LIBERO-Occ, CoMe, ZS-IP, and SaPaVe remains accurate.

At this revision the status is **GO for implementation and data collection,
NO-GO for a method-performance claim**. The original drawer replay verifies
software structure only.
