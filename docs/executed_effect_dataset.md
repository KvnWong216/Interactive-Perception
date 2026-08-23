# Executed counterfactual effect labels

**Frozen on:** 2026-08-22

The `original_drawer_executed_v2` development set replaces the earlier
seed-1399 capability proxy with candidate outcomes measured on ten fresh
same-state seeds. It is intentionally a development artifact: these seeds have
already been inspected and must not be reused as an untouched test set.

## Fork structure

Each seed contributes one closed initial simulator state, two task prompts with
identical public RGB, and three candidate branches:

| prompt condition | DIRECT | OPEN | STOP |
|---|---|---|---|
| hidden butter | frozen pi0.5 rollout | frozen pi0.5 rollout | exact null option |
| visible cream cheese | frozen pi0.5 rollout | same real OPEN rollout, task-specific offline relabel | exact null option |

`DIRECT` and `OPEN` therefore have measured action trajectories and public
pre/post frames. `STOP` is not sampled from pi0.5: its contract is a
deterministic zero-action transition, so the post frame equals the pre frame.
The cream-cheese OPEN labels replay the retained action history in a separate
evaluator with cream cheese declared as the target. The policy trajectory is
not rerun, its input is not changed, and privileged evaluator outputs never
enter the policy index.

All six branches from one simulator seed share an `initial_state_id` and the
same split. This prevents both action-fork and prompt-pair leakage.

## Label policy

The factors describe observable post-action facts or changes; they are not an
arbitrary scalar utility.

| factor | operational definition in this fixed scenario |
|---|---|
| `execution_succeeded` | intended physical result holds: target remains in the destination at terminal replay for DIRECT, drawer threshold crossed for OPEN, or STOP completed |
| `task_relevant_change` | target was picked/placed, or OPEN produced new target evidence for the hidden-target prompt |
| `ambiguity_reduced` | historical v2-dataset label: target evidence crossed from below 256 instance pixels before the action to at least 256 after it |
| `target_confirmed` | target is publicly observable after the action or was physically picked |
| `candidate_rejected` | a target/location hypothesis was disproved |
| `region_confirmed_empty` | an inspected region was shown empty |

The last two factors have no positive examples in T01D because the butter is
always inside the drawer. They are marked unsupported rather than assigned
synthetic positives. A learned six-factor generalization claim therefore
remains blocked until a preregistered empty/rejected fork exists.

The 256-pixel crossing is retained only to reproduce this frozen development
dataset. It was not calibrated as a recognition threshold and is prohibited
from successor-method claims. The threshold audit shows that its 8/10 physical
conclusion is unchanged for every integer cutoff from 1 through 447 pixels.

Instance segmentation, object contact, object lift, destination predicates,
and drawer joints are evaluator-only label sources. The online policy receives
only stock agent-view RGB, wrist RGB, public proprioception, the semantic
candidate, and public action history.

## Scope and next split

This set supports schema validation, effect-head debugging, and development
model selection. It cannot support a final calibrated risk or generalization
claim because it has only ten inspected same-scene groups and two factors lack
positive support. A final cycle must collect new group-disjoint calibration and
test seeds under the same frozen label policy; changing thresholds after those
runs begin invalidates the split.
