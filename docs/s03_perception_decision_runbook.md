# S03 public perception-decision prospective runbook

## Freeze status and scope

This document freezes the version-1 **design** of the public-input
perception-decision qualification. It does not authorize execution. At this
freeze point:

- `design_status: FROZEN_BEFORE_S03_OUTCOMES`;
- `rollout_executed: false`;
- `outcomes_loaded: false`;
- `paper_claim_ready: false`;
- the complete finite input cohort is the 124 entries of the immutable S02
  schedule, in its existing execution order;
- no S02 file may be edited, regenerated, or replaced.

The scientific scope is development mechanism evidence in the unchanged
original-drawer scenario. A positive S03 result would show that a frozen
public-input perception/router stack reacts coherently to evidence already
produced by the qualified OPEN primitive. It would not establish task-success
improvement, deployment reliability, calibration coverage, or a paper-ready
main result.

## Repository audit and canonical-DAG mismatch

The current canonical DAG names
`S03_oracle_development_gate`, but its expected artifacts are the privileged
Oracle visual-prompt screen and confirmation pilot:

- `results/method/original_drawer_oracle_prompt_screen_v2.json`;
- `results/method/original_drawer_oracle_prompt_pilot_v2.json`.

The corresponding runner passes an evaluator-only target marker rendered from
simulator instance segmentation to pi0.5. That is a valid privileged upper-bound
diagnostic under its original claim scope, but it is not the public
perception-decision S03 defined here. It violates this runbook's controller
firewall and must not be run as a substitute for subtests A/B/C.

The existing pre-outcome Oracle screen schedule is retained unchanged at
`results/method/original_drawer_oracle_prompt_screen_schedule_v1.json`
(SHA-256
`6c9bb68a1b1ef0af3e2682d9bfc1c72ed5bca7c34de767658e940e8c587d8ffa`).
It binds an older offline repro lock and does not authorize the new S03. This
task deliberately does not amend the canonical DAG. A later, separately
reviewed DAG amendment must either assign the public qualification a distinct
stage ID or replace the legacy S03 contract before any A/B/C outcome is
generated.

The frozen upstream references are:

| Object | SHA-256 |
| --- | --- |
| S02 schedule | `45d41b75fd982b977a0c9c5f82f63c2f33b686fd1512308cacf3cc690847e762` |
| S02 certificate | `5291b6450230b5c8d36a28a805676cf61f655583431f2e9c53b0d548872cbd75` |
| original-drawer scenario | `fa2d550a2fa9a17128dac9bae93bc06f9601fb2f6a5d16b8e09f890c329bbc67` |
| public candidate set | `8580c622ac25fcddde9707b4fec2d82b0e4652f5af53b608084fe3bd716a2ab4` |
| canonical DAG at audit | `4f390bb0b9f0ca1216544f0da05f114f0b39029b286f74e10190f3283b3deda0` |

## Research questions

S03 asks three prospectively fixed questions:

1. Does the public observation after OPEN reduce prompt-conditioned location
   uncertainty and make task-relevant evidence more sufficient than the paired
   pre-OPEN observation?
2. Does one frozen perception/router identity distinguish the three required
   routes: visible target to ACT, hidden target to OPEN, and publicly exhausted
   absent-target search to STOP?
3. Can a hash-linked replay perform `observe -> select OPEN -> bind the
   recorded OPEN trace -> re-observe -> update belief -> select the next
   primitive` without any controller-visible simulator semantics?

## Frozen finite input split

No S03 group is selected using S02 success, OPEN joint displacement, target
visibility, or any later perception result. The input cohort is a census of all
124 S02 schedule entries, including execution indices 101 and 104. There is no
replacement, exclusion, or retry rule.

S03 v1 performs no fitting, threshold selection, prompt selection, or model
selection. Therefore the same finite cohort is intentionally reused by all
three mechanistic subtests instead of being divided into train/calibration/test
roles. If any model, prompt, route rule, metric, or threshold is changed after
an S03 result is inspected, all 124 groups become development-only and the
changed method requires a new version and new outcome-unseen groups.

For every S02 execution index `i`, the future input manifest must bind these
public assets directly by path and SHA-256:

- pre observation: `00_before_agentview.png` and `00_before_wrist.png`;
- post observation at the comparable public home pose:
  `05_returned_home_agentview.png` and `05_returned_home_wrist.png`;
- public low-level action history: `public_action_history.json`;
- the prompt, candidate registry, frozen model identities, and any public
  proprioception explicitly exported from the controller keyframe.

The manifest must not expose the enclosing semantic report to the controller,
because that report also contains an evaluator-private subtree. It must not
reference the S02 outcome JSONL, S02 evaluator fields, S02 success/failure, or
the opaque initial/final NPZ states as perception inputs.

The fixed prompt strata are:

| Stratum | Observation/history | Prompt | Expected route (evaluator label only) |
| --- | --- | --- | --- |
| visible | pre-OPEN public RGB, empty history | `Place the cream cheese in the basket.` | ACT |
| hidden | pre-OPEN public RGB, empty history | `Place the butter in the basket.` | OPEN |
| exhausted/absent | post-OPEN public RGB plus the public OPEN/evidence history | `Place the milk in the basket.` | STOP, only with a public coverage certificate |

`milk` is fixed before S03 inference and is absent from the frozen BDDL object
declaration. That fact is an evaluator label, not controller input. The
controller receives only the ordinary user prompt, public images, and public
history. If the public RGB critic cannot produce a singleton empty/coverage
certificate, STOP is not authorized; ABSTAIN is safe but counts as a routing
miss in the fixed STOP stratum.

The resulting finite evaluation sizes are 124 paired A records, 372 B records
(124 in each prompt stratum), and 124 C traces. No new pi0.5 action call is part
of S03 v1. The physical OPEN segment in C is the already frozen S02 trace,
bound by index and hashes. A new physical rollout would require a new design
version and explicit authorization.

## Public-input firewall

The following inputs are allowed online to the perception/router:

- complete task prompt;
- agent-view and wrist-view RGB;
- public robot proprioception represented in the frozen observation schema;
- RGB-derived Grounding DINO proposals, SAM masks, DINO associations, and VLM
  features, provided their model identities are hash-bound;
- candidate IDs and public semantic action descriptions;
- previous public belief, selected primitive, serialized subtask, RGB critic
  outcome set, coverage certificate, and public action history;
- deterministic hashes, indices, and schema metadata that carry no evaluator
  value.

The following are forbidden controller or policy inputs:

- simulator semantic or instance IDs;
- simulator segmentation, target masks, marker pixels, oracle boxes/points, or
  an oracle target location;
- object poses, drawer joint values, contacts, task predicates, reward, success,
  or BDDL object membership;
- S02 qualification outcomes, failure indices, evaluator reports, or
  certificate result fields;
- decoded opaque NPZ simulator state;
- a route label, sufficiency label, revealed label, or expected next primitive.

RGB-derived SAM masks are allowed; simulator-rendered instance masks are not.
Every public report must contain `online_oracle_inputs: []`. A nonempty value,
an unknown input field, or a hash reference into an evaluator-private artifact
invalidates the entire S03 result rather than merely one group.

Evaluator-private labels may be joined only after all public inference reports
are immutable. They may include BDDL target presence, simulator instance-mask
visibility, drawer joint state, contacts, and task predicates. The join must be
performed by execution index and hashes and must never write those labels back
into a controller input or public-history artifact.

## Frozen statistical interpretation

S03 is a one-shot development mechanism test, not a deployment reliability
qualification. The S02 threshold of 122/124 is not reused: no task-owner
perception-error budget currently justifies a 0.95 router requirement.

There are five primary directional tests: A, three B strata, and C. Familywise
alpha is 0.05 with a Bonferroni allocation of 0.01 to each primary test. The
thresholds are exact binomial boundaries, not tuned confidence cutoffs:

- for A and C, the null is that no more than half of the complete finite cohort
  has the prospectively defined positive transition. With `n=124`, support
  requires at least 76 positives; the exact one-sided tail at 76 is
  `0.007492309431666172`, while the tail at 75 is
  `0.012185085061199809`;
- for each B stratum, the null correct-route probability is uniform three-way
  chance, `1/3`. With `n=124`, support requires at least 55 correct routes; the
  exact one-sided tail at 55 is `0.0068749601466991115`, while the tail at 54 is
  `0.01129276877169164`.

No minimum entropy magnitude, probability confidence, pixel count, or learned
score threshold is introduced. Continuous deltas, confusion matrices, exact
confidence intervals, abstentions, and every failure category must be reported.
Crossing these boundaries permits only the label
`DEVELOPMENT_MECHANISM_SUPPORTED`.

## Subtest A: Information Effect

### Inputs and outputs

For each index, run the same frozen perception identity on the paired pre and
post public observations with the butter task prompt. The post computation may
consume only the public OPEN history and public RGB-critic outcome. It outputs
two belief reports, their hashes, normalized target-location entropies, and the
public task-sufficiency representation if the frozen backend supplies one.

After public reports are sealed, an evaluator-only record adds pre/post
task-information sufficiency and target-revealed labels. If sufficiency lacks a
defined frozen schema, it is recorded as `UNAVAILABLE`; it must not be replaced
by a hand threshold over an uncalibrated score.

### Metrics and success definition

Report for every group:

- `delta_location_entropy = H_pre - H_post`;
- pre/post target-location distributions and top hypotheses;
- pre/post public sufficiency representation;
- evaluator-private pre/post sufficiency and target-revealed transitions;
- OPEN execution and public-critic failure categories, without excluding them.

The fixed A primary indicator is true only when all three statements hold:

1. `H_post < H_pre`;
2. evaluator-private task information changes from insufficient to sufficient;
3. evaluator-private target evidence changes from not revealed to revealed.

Missing/nonfinite values, ties, failed OPEN, and unavailable sufficiency count
as nonpositive in the complete denominator. A is supported at 76/124 or more.
The three component rates and continuous entropy change remain separately
visible so a binary result cannot hide whether failure came from execution,
perception, or sufficiency.

### Failure categories and validator

Failure categories are `INPUT_HASH_MISMATCH`, `MODEL_IDENTITY_MISMATCH`,
`POLICY_FIREWALL_VIOLATION`, `OPEN_EXECUTION_FAILURE`,
`PUBLIC_OUTCOME_AMBIGUOUS`, `BELIEF_NONFINITE`, `NO_ENTROPY_REDUCTION`,
`SUFFICIENCY_NOT_IMPROVED`, and `TARGET_NOT_REVEALED`.

The A validator must recompute entropy from the stored distributions, verify
the pair hashes and complete 124-index denominator, prove that private labels
were joined only after public report hashes were frozen, and recompute the
exact binomial tail.

## Subtest B: Decision Routing

### Inputs and outputs

For every index, evaluate the three fixed prompt strata above using one frozen
perception/router identity and one candidate order. Output the full route
distribution or prediction set, calibrated/set-valued evidence if present,
selected decision kind, candidate ID, primitive, reason, and public coverage
certificate. ACT denotes the public task-execution candidate (`DIRECT` in the
current candidate registry); it does not execute that candidate in S03.

### Metrics and success definition

Report the 3x3 ACT/OPEN/STOP confusion matrix, ABSTAIN separately, per-stratum
accuracy, macro accuracy, STOP-without-coverage count, and all route/prediction
sets. A stratum is supported at 55/124 or more correct singleton routes. All
three strata must meet their threshold, and `STOP_WITHOUT_PUBLIC_COVERAGE` must
equal zero, for B to be supported.

ABSTAIN never becomes a hidden success. It is reported as safe nonexecution and
counts as incorrect for this routing mechanism test. An unsafe STOP is both a
routing miss and a structural contract failure.

### Failure categories and validator

Failure categories are `VISIBLE_NOT_ACT`, `HIDDEN_NOT_OPEN`,
`ABSENT_NOT_STOP`, `ABSTAINED`, `MULTI_ROUTE_SET`,
`STOP_WITHOUT_PUBLIC_COVERAGE`, `UNKNOWN_CANDIDATE`,
`MODEL_IDENTITY_MISMATCH`, and `POLICY_FIREWALL_VIOLATION`.

The B validator must verify exact prompts and frame/history hashes, the fixed
candidate order, one report per index and stratum, no physical dispatch, no
unknown route, no private input reference, and exact recomputation of all three
binomial tests.

## Subtest C: Closed-loop Transition

### Inputs and outputs

For every index, construct a replay trace with these immutable links:

1. pre public observation and butter prompt;
2. pre belief/report hash and selected OPEN decision;
3. S02 execution-index link to the recorded public OPEN action history;
4. post public observation and RGB-critic outcome;
5. post belief/report hash with the previous belief, executed OPEN, and public
   outcome explicitly recorded;
6. next selected primitive and reason.

The output must include observation, belief, decision, history, and model hash
links at both time steps. It must identify the S02 physical failure separately
from a stale update or wrong route. It must not execute the next ACT/OPEN/STOP.

### Metrics and success definition

`transition_chain_integrity` requires 124/124 valid hash chains. Any broken
chain invalidates C rather than reducing its scientific success count.

The fixed C primary indicator is true only when the pre route is OPEN, the
public outcome is singleton evidence acquired, the post belief hash and public
observation hash differ from pre, and the next singleton route is ACT. Failed
OPEN, ambiguous public outcome, stale belief, ABSTAIN, repeated OPEN, or STOP
count as nonpositive. C is supported at 76/124 or more.

The result must additionally report the full next-route distribution. Repeated
OPEN after a physical OPEN failure may be a sensible safe behavior, but it is
not relabeled as success for this fixed evidence-to-ACT transition question.

### Failure categories and validator

Failure categories are `BROKEN_HASH_CHAIN`, `PRE_ROUTE_NOT_OPEN`,
`OPEN_EXECUTION_FAILURE`, `PUBLIC_OUTCOME_AMBIGUOUS`,
`POST_OBSERVATION_STALE`, `BELIEF_UPDATE_STALE`, `POST_ROUTE_NOT_ACT`,
`ABSTAINED`, `UNSAFE_STOP`, `MODEL_IDENTITY_MISMATCH`, and
`POLICY_FIREWALL_VIOLATION`.

The C validator must re-hash every link, enforce complete execution-index order,
verify that the update consumes only the previous public report and public
outcome, reject physical dispatch in the post-decision step, and recompute the
fixed indicator and exact tail.

## Required future artifacts and SHA coverage

No artifact in this section is created by the design-freeze task. Before S03
outcomes, a later input-freeze task must create, in this order:

1. `results/method/piu_s03_perception_decision_input_manifest_v1.json` with
   schema `piu.s03-perception-decision-input-manifest.v1`, status
   `FROZEN_BEFORE_S03_OUTCOMES`, all 124 execution indices, direct public-asset
   hashes, model identities, prompts, and explicit forbidden-field audit;
2. `results/method/piu_s03_perception_decision_schedule_v1.json` with schema
   `piu.s03-perception-decision-schedule.v1`, the 620 hash-keyed offline
   evaluation records, `rollout_executed=false`, and `outcomes_loaded=false`;
3. immutable public reports below
   `runs/piu_s03_perception_decision_v1/{information_effect,decision_routing,closed_loop_transition}/`;
4. evaluator-only labels at
   `results/evaluation/piu_s03_private_labels_v1.jsonl`, which must not be
   referenced by any public report;
5. results
   `results/method/piu_s03_information_effect_v1.json`,
   `results/method/piu_s03_decision_routing_v1.json`, and
   `results/method/piu_s03_closed_loop_transition_v1.json`;
6. `results/method/piu_s03_perception_decision_certificate_v1.json` with status
   `DEVELOPMENT_MECHANISM_SUPPORTED`,
   `DEVELOPMENT_MECHANISM_NOT_SUPPORTED`, or `INVALID_INPUT_FIREWALL`.

The input manifest must cover this runbook, S02 schedule and certificate, every
public input byte, candidate registry, public observation schema, all frozen
model/checkpoint identities, and inference/controller code identities. The
schedule must cover the manifest and exact record order. Each raw report must
cover its schedule row and previous-report hash where applicable. Results must
cover every raw report, private-label manifest, failure record, and exact-test
calculation. The certificate must cover the three result hashes and complete
artifact-tree hash.

Every result and certificate must set:

- `formal_method_claim: false`;
- `paper_claim_ready: false`;
- `s02_artifacts_modified: false`;
- `new_physical_rollout_executed: false`.

## Stop conditions and next authorized step

Do not execute S03 while any of these conditions holds:

- the canonical DAG still maps S03 exclusively to the privileged Oracle gate;
- the public perception/router model identity and output schema are not frozen;
- the 124-entry sanitized input manifest or 620-record schedule is absent;
- a schedule builder reads S02 outcome/evaluator fields while constructing
  public inputs;
- the runner can call pi0.5, `env.step`, or execute the post-decision primitive;
- a validator cannot prove controller/evaluator artifact separation.

The next authorized task is therefore **input-manifest and offline-schedule
scaffolding only**, together with a reviewed DAG amendment. It is not S03
inference, Oracle execution, physical rollout, model tuning, or result
generation.
