# ADR-0002: grounded intervention outcome planning

- Status: accepted for formal-v1 software implementation
- Date: 2026-09-07
- Scope: Stage-1 method contract over a separately hosted frozen VLA
- Supersedes: ADR-0001 as the active method decision

ADR-0001 remains immutable documentation of the previous candidate-conditioned,
factorized, calibrated-interaction design. This ADR supersedes it only for the active
research method; it does not reinterpret or invalidate historical experiments performed
under the older contracts.

## Context

The previous active design did not align the physical action being predicted with the
one ultimately executed. In particular:

- a primitive name is not a complete physical intervention;
- future-feature similarity is not task value or information value;
- a grounding map decoded after effect prediction does not guarantee that prediction
  and execution refer to the same instance;
- singleton conformal routing can abstain even when several actions are acceptable; and
- opening a drawer reliably does not show that newly revealed evidence changes the next
  task action.

The reviewed priority is therefore not another uncertainty head. It is an auditable
chain from grounded candidate, through policy-relative outcome prediction and physical
execution, to actual post-action reobservation and contingent next action.

Superseded implementations and artifacts remain recoverable from Git snapshot
`8c4be631`; formal v1 has no runtime dependency on them.

## Decision

Implement the active formal method in a new, independently versioned package. Do not
extend `latent_interaction` into the formal method.

Formal v1 uses exactly the following central path:

1. Accept the complete user prompt, public RGB/history, public action history, and
   optional public proprioception.
2. Consume frozen multimodal token fields with camera, time, patch, and coordinate
   metadata.
3. Accept a finite set of complete grounded interventions
   `(primitive, referent, parameters, grounding)` constructed before scoring.
4. Predict one bounded task-success logit per candidate under a frozen serializer,
   executor, continuation policy, and horizon.
5. Filter invalid, unsafe, unsupported, or unqualified candidates.
6. Select the maximum predicted value from the finite feasible set, with a frozen
   deterministic tie rule.
7. Serialize that same selected candidate into an instance-distinguishing text subtask.
8. Execute it through a separately hosted frozen VLA.
9. Encode the actual post-action public observation, append public history, and decide
   again.

The current package does not implement the upstream adapter that proposes candidates,
discovers referents, or binds them to image regions. It consumes already-bound
interventions.

`DIRECT` and `STOP` are ordinary candidates in the comparison. `STOP` never invokes the
executor. Active viewpoint change is excluded; formal-v1 information actions are
interactive manipulation and information enrichment.

## Input and output decision

### Stage-1 input

```text
prompt
public RGB history
public action/observation history
optional public proprioception
frozen token metadata: provider ID, camera, time, patch support, coordinates
```

Simulator segmentation, semantic/instance ID, target truth, hidden poses, joint truth,
task predicates, and evaluator masks are forbidden online inputs.

### Candidate input to the scorer

```text
candidate_id
primitive
referent text
primitive parameters
current public-frame grounding: camera, frame, image digest, box, point
```

Serializer and executor identity, continuation policy, horizon, outcome name, and
failure handling belong to `OutcomeContract` and the execution chain. They are not
fields of `GroundedIntervention`.

### Stage-1 output

```text
bounded task-success logit for every valid candidate
selected candidate_id or fail-closed status
```

`GroundedCandidateEncoder` consumes pre-bound patch support and only fuses the candidate
with those patches. It does not discover a referent or predict grounding.
`OutcomePrediction` has no branch logits or `grounding_logits`.

### Stage-2 output

The frozen VLA returns a continuous action chunk. Formal v1 does not alter the executor's
weights or inject unregistered soft tokens/K/V states.

## Candidate identity decision

The same immutable `candidate_id` must be consumed by proposal, scoring, selection,
serialization, execution receipt, post-observation record, and evaluator sidecar. A
primitive-only join is invalid. A correct primitive executed on the wrong referent is a
binding failure and must remain visible in evaluation.

This chain is part of the method contract because the learned value is intervention
specific:

\[
V_q^{\pi_C}(x,u)=
\mathbb E[G_q\mid x,\operatorname{do}(u;\sigma,\pi_E,v_E),\pi_C,H].
\]

Changing the serializer, executor, decoding contract, horizon, or continuation policy
changes the intervention distribution and requires a new version.

## Supervision decision

Train the first version from reset-controlled, actually executed branches. One record
contains the common public pre-action context, one complete candidate, the actual public
post-action observation, a privileged evaluator sidecar, decomposed execution
diagnostics, and the bounded outcome under the frozen continuation contract.

Do not:

- fabricate outcomes for unexecuted candidates;
- infer the value of OPEN from the fact that the task is not complete immediately after
  OPEN;
- use expert route labels as if they were measured counterfactual outcomes; or
- allow prompt/action branches of one physical initial state to cross dataset splits.

The sole formal-v1 objective is receipt-backed, executed-candidate Bernoulli NLL for the
bounded task outcome. The core method has no branch, grounding, or ranking auxiliary and
no hand-composed uncertainty scalar.

## Selection decision

The first planner is a finite feasible-set argmax over predicted policy-relative task
value. Hard feasibility, executor support, qualification, and safety constraints are
applied before ranking. Exact ties use a frozen deterministic rule.

This is intentionally a one-step/receding decision interface, not a claim that the
repository solves a full continuous POMDP. Information gain may be reported later as a
diagnostic, but it does not replace the task-outcome objective.

## Deferred decisions

The following are deferred until result-only prediction, binding, and hidden-result
branching work:

- future-latent/JEPA supervision;
- EMA target encoders;
- stochastic latent world models;
- multi-step search;
- conformal action sets;
- evidential/Dirichlet output heads;
- typed point/region injection into the frozen executor;
- soft-token or K/V adapters;
- branch, grounding, or ranking auxiliary losses;
- Stage-2 fine-tuning; and
- reinforcement learning.

Future-latent prediction may return only as a controlled auxiliary with equal-capacity
`no path`, `path without future-target loss`, and `path with future-target loss`
conditions. Calibration may return only as a secondary held-out risk layer; it must not
define the central behavior or force abstention merely because multiple good candidates
exist.

## Consequences

### Positive

- Prediction, supervision, selection, and execution refer to the same physical
  intervention.
- The learned target is tied to a preregistered task outcome rather than a renamed latent
  distance.
- The method can compare `DIRECT`, `STOP`, and several same-primitive/different-referent
  interactions in one finite set.
- Real post-action observations, rather than imagined states, close the loop.
- Failures can be decomposed into proposal, grounding, value ranking, serialization,
  binding, motor execution, post-observation interpretation, and continuation failures.
- The frozen VLA keeps Stage-1 research separable from low-level policy training.

### Costs and limitations

- Reset-controlled candidate execution is more expensive than route-label supervision.
- Values are executor-, serializer-, continuation-, and horizon-relative.
- A text-only frozen-VLA interface may fail to preserve spatial identity; this must be
  measured before large outcome collection.
- The first version is one-step and does not provide a global POMDP optimality guarantee.
- Frozen VLM tensors and synthetic tests do not demonstrate real VLM integration or
  open-world generalization.
- Calibration and OOD behavior remain unresolved until independent data exist.

## Alternatives considered

Extending the previous future-feature or factorized controllers was rejected because it
would not repair candidate identity or add task-outcome supervision. Unconstrained VLM
subtask generation remains a baseline. End-to-end active-VLA training is deferred until
the benchmark supports a fair comparison and the information-decision/execution failure
boundary is measurable.

## Verification and claim boundary

The initial implementation may claim only that its software contracts pass synthetic
tests. It must not claim that:

- a concrete Stage-1 VLM adapter is connected;
- an upstream candidate-proposal/grounding adapter is connected;
- MolmoAct2 is running or qualified;
- the outcome model has been trained;
- uncertainty is calibrated;
- hidden-result branching succeeds;
- a LIBERO task has improved; or
- the method has been evaluated on a real robot.

Those claims require separate, versioned evidence. The required empirical order is:

1. oracle candidate-to-executor binding ceiling;
2. learned proposal and physical binding;
3. randomized hidden-content second-action branching;
4. reset-controlled outcome learning against route/history baselines;
5. paired closed-loop task comparison; and
6. held-out scene/object/prompt tests and a second executor contract.

The companion specification is [`docs/formal_pipeline_v1.md`](../formal_pipeline_v1.md).
