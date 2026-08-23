# PIU action-effect and calibrated-control pipeline

Status: software-ready on CPU; real executed counterfactual data and physical
evidence pending.

The downstream model consumes the frozen binder `target_token` and the full
frozen prefix for each public candidate prompt. It predicts one route score and
eight non-exclusive effect factors per candidate. It does not receive a hidden
correct action, mask, contact, joint, object pose, or task predicate online.
The route head explicitly consumes the learned interaction token concatenated
with predicted factor probabilities. There is no manual factor-to-utility
formula; the three declared variants control whether effect-loss gradients may
alter the shared representation.

The temporal index is causal. One feature row represents public history ending
at decision state `s_t`; its `post_interaction` image is the current image, not
the outcome of the candidate being scored. Every candidate label comes from a
separate next-action rollout whose pre-observation digest must exactly equal
the feature row's `decision_observation_sha256`. The candidate rollout's post
image is provenance/label evidence and is never part of the score input. The
join rejects a different starting state, preventing retrospective effect
recognition from being reported as action-effect prediction.

## Label and candidate contracts

Every non-null factor label must come from an actually executed action fork and
the separate replay evaluator. STOP and REPORT_NOT_FOUND alone may use exact
null transitions: no action is executed and the pre/post public-observation
hashes must match.
Capability declarations cannot stand in for outcomes. If identity resolution,
rejection, empty-region coverage, or task sufficiency is not independently
annotated, its target is null and its loss/calibration are unsupported.

The candidate generator always emits the public task-action superset (PICK and
PLACE), every information primitive attached to every public affordance,
REPORT_NOT_FOUND, and STOP. It receives neither inferred holding/task progress
nor the true target location. The calibrated binder and controller gate task
actions after candidate generation, avoiding a circular or caller-injectable
candidate set. A finite public trace certifies search exhaustion only after
every registered information source has a singleton post-observation
empty-region verifier; predicted action effects never become observed facts.

Before same-state action forks are collected, an immutable public execution
plan applies those same binder-only primitive prerequisites. Information forks
require singleton insufficient evidence; DIRECT/PICK require singleton
sufficiency, non-holding, target presence, and a current-frame spatial set;
PLACE requires singleton sufficiency and holding. These are typed execution
preconditions, not confidence cutoffs, and no effect prediction or evaluator
outcome may enter the plan. An ineligible candidate remains in the candidate
and route-label matrix, but it is not executed and every unobserved effect
factor is null/masked. Thus binder ambiguity produces safe abstention at
execution time without silently changing the evaluator's route target or
inventing a candidate outcome.

## Training and ablations

CPU training retains three variants:

- `route_only`: route loss only, with the factor channel fixed to exact zeros so
  neither random nor trained factor-head values can affect routing;
- `stop_gradient_effect`: effect supervision cannot update the shared route
  representation;
- `joint_effect`: route and supported effects update the shared representation.

Task scales are learned on training groups. The development split selects from
the declared width/dropout/learning-rate/seed grid using proper scores and
retains every trial. The binding trainer also runs full, no-prompt,
prompt-swap, last-frame, global-mean, no-history, single-camera, shuffled-space,
and shuffled-time variants with the selected hyperparameters. These transforms
are label-blind.

## Isolated calibration and controller

One calibration role fits temperatures. A group-disjoint role fits finite
multiclass route and class-conditional binary LAC sets. Primary alpha is 0.10;
0.05 and 0.20 are reports, not a result-selected risk level. Factors lacking
both calibration classes remain unsupported and can never authorize an action.

The controller contains no top-1 fallback or scalar utility. It acts only when
the route set and every logically required belief/effect set are singleton.
Information actions require certified insufficiency, execution, and
task-relevant change. PICK/PLACE require certified sufficiency, execution, and
task progress. PICK requires a singleton calibrated not-holding state, singleton
presence, and a nonempty calibrated current-frame spatial patch set; PLACE
requires a singleton calibrated holding state. This holding set is a calibrated
post-observation binder output from RGB, public gripper state, and action
history—not the previous action's predicted success and not simulator grasp
truth. An ambiguous set cannot authorize either next task primitive. STOP
requires the current binder's singleton task-complete set;
REPORT_NOT_FOUND requires target
absence plus exhaustive search coverage; all other cases ABSTAIN. The reported
Fréchet joint lower bound is diagnostic only and never selects a candidate.
The executable controller entry point accepts frozen features, label-free
binder predictions, separate binder/effect calibration artifacts, and the
hash-chained public search-memory set. It has no
`--labels` argument; training/evaluator targets cannot enter this online join.
It serializes the selected primitive/referent and exact normalized enclosure of
all calibrated current-frame patches into a deterministic pi0.5 text subtask.
Confidence values, chain of thought, and evaluator fields are excluded.

## Commands and evidence boundary

After real group-disjoint artifacts exist, run:

```bash
CUDA_VISIBLE_DEVICES='' python \
  scripts/data/build_piu_counterfactual_execution_plan.py \
  --public-transition PATH/decision.jsonl --sample-id SAMPLE \
  --binding-predictions PATH/binder.npz --binding-report PATH/binder.json \
  --binder-calibration PATH/binder_calibration.json \
  --feature-report PATH/features.json --output PATH/execution_plan.json

python scripts/data/collect_piu_counterfactual_branches.py \
  --decision-transition PATH/decision.jsonl --decision-sample-id SAMPLE \
  --execution-plan PATH/execution_plan.json \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --qualification-map PATH/qualified_executor_map.json \
  --source-state PATH/source_state.npz --seed SEED \
  --host PI05_HOST --port 8002 --output-dir PATH/branches

CUDA_VISIBLE_DEVICES='' python scripts/training/train_piu_action_effect.py \
  --config configs/experiments/piu_action_effect_v1.yaml \
  --features PATH/features.npz --feature-report PATH/features.json \
  --binding-predictions PATH/binder.npz \
  --binding-report PATH/binder.json \
  --labels PATH/effects.jsonl --output-dir PATH/train

CUDA_VISIBLE_DEVICES='' python scripts/calibration/calibrate_piu_action_effect.py \
  --config configs/experiments/piu_action_effect_calibration_v1.yaml \
  --temperature-predictions PATH/temperature.npz \
  --temperature-report PATH/temperature.json \
  --conformal-predictions PATH/conformal.npz \
  --conformal-report PATH/conformal.json \
  --output PATH/effect_calibration.json

CUDA_VISIBLE_DEVICES='' python scripts/pipeline/run_piu_calibrated_controller.py \
  --checkpoint PATH/joint_effect.pt --training-report PATH/training_report.json \
  --features PATH/features.npz --feature-report PATH/features.json \
  --binding-predictions PATH/binder.npz --binding-report PATH/binder.json \
  --binder-calibration PATH/binder_calibration.json \
  --effect-calibration PATH/effect_calibration.json \
  --public-state-sets PATH/public_state_sets.jsonl \
  --expected-split development --output PATH/controller.json

python scripts/pipeline/execute_piu_controller_decision.py \
  --controller-report PATH/controller.json --sample-id SAMPLE \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --primitive-qualification PATH/formal_primitive_certificate.json \
  --seed SEED --host PI05_HOST --port 8002 --run-dir PATH/immutable_run
```

The second command refuses live execution unless the certificate is
`FORMALLY_QUALIFIED` for the exact candidate payload, primitive, and serializer
mode. Qualification first derives a rate contract from an externally declared
episode risk budget and power alternative. The same external task-owner
contract must declare a positive-integer maximum qualification-group count per
primitive; there is no CLI default. The planner either freezes the first exact
binomial design within that collection-resource cap or retains a blocked plan
without changing the test. It then freezes unique new state/controller groups
with `build_piu_primitive_qualification_schedule.py` and executes them in order
with `run_piu_primitive_qualification.py`.
Before a binder exists, each OPEN group may obtain its decision artifact from
`build_piu_primitive_qualification_probe.py`. The probe reads the candidate ID
from the frozen plan and the full candidate payload from a hash-bound public
candidate set; it loads no model, calibration, outcome, or oracle field. This
breaks only the OPEN-before-data dependency and is not a controller result.
PICK/DIRECT probes are rejected because those primitives must retain calibrated
current-frame boxes from the learned path.
`evaluate_piu_primitive_qualification.py` generates its own outcome JSONL from
those single-use receipts and registered simulator/task predicates; it does not
accept user-written success flags. Missing or failed certificates permit only
`--dry-run`, never a paper-method policy call.

For the preregistered ablations, B3/B4 use
`run_piu_uncalibrated_ablation_controller.py`: B3 requires the `route_only`
checkpoint and an exact-zero factor channel; B4 requires `joint_effect` and
uses predicted factors in the learned route head. Both choose a unique route
argmax and abstain on exact ties. B5 uses the calibrated controller with
`--method-id B5`, which retains the same calibrated route/belief/effect sets but
serializes no patch geometry. B8 uses `--method-id B8` and enables the spatial
bridge. No variant is selected from sealed results.

B6 is never mixed with these public methods. A one-step oracle decision uses
`run_piu_oracle_effect_controller.py`; a complete precollected counterfactual
tree is replayed with `replay_piu_oracle_effect_trace.py`. The replay follows
only the evaluator-correct executed branch, verifies the opaque simulator-state
and public-observation chain at every node, and emits an
`oracle_upper_bound` episode with
`online_oracle_inputs=[executed_candidate_effect_labels]`. It cannot be used as
the proposed method or as public online performance.

B7 likewise stays separate. `run_piu_oracle_binding_full_loop.py` begins at the
same paired hidden-target simulator state as B0 and uses the same DIRECT budget.
The evaluator mask is audited on every policy call: it makes no RGB change
while the target is hidden and applies only the uniquely development-selected
visual marker once the target is visible. This produces a complete B7 oracle
episode without relabeling the existing conditional post-OPEN pilot as a
same-source result.

Synthetic end-to-end tests establish serialization, split isolation, numerical
execution, and sealed immutability only. They do not establish calibrated risk,
better routing, target binding, target contact, or full-loop task success.
