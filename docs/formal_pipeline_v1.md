# Formal pipeline v1: grounded intervention outcome planning

## Status

This document is the active software and data contract for the first formal method
implementation. Its status is **software-only**: it defines typed interfaces, learning
targets, selection semantics, and the closed-loop boundary, but it does not claim that a
candidate-proposal/grounding adapter, real Stage-1 VLM, MolmoAct2 endpoint, learned
checkpoint, LIBERO evaluation, or robot experiment is available.

The method question is deliberately narrow:

> Given a user's task and public observation history, can a robot compare complete,
> spatially grounded interventions by their policy-relative task outcomes, execute one
> with a frozen VLA, and use the actual post-action observation to choose what to do
> next?

Active viewpoint change is outside the current scope. Information actions are restricted
to interactive manipulation and information enrichment, such as `OPEN`, `REMOVE`,
`ROTATE`, and `BRING_CLOSE`.

## 1. Method boundary

The formal v1 path is:

```text
public prompt + public RGB/history + optional public proprioception
                              |
                 frozen VLM tokens (protocol)
                              |                 externally supplied
                              |              pre-bound interventions
                              |                        |
                              +-----------+------------+
                                          |
             candidate-conditioned outcome/value prediction
                              |
                  finite feasible-candidate argmax
                              |
                 deterministic referential serialization
                              |
                        frozen VLA executor
                              |
                  action chunk -> actual reobservation
                              |
                     append history and repeat
```

The method does **not** compute a hand-weighted uncertainty score. It does not call a
latent token a Bayesian belief. Its statistically testable prediction is the outcome of
a complete candidate under a frozen interface, executor, continuation policy, and
horizon.

The current package starts at the two typed inputs shown above. It does not implement
the upstream component that proposes a referent and binds it to a public image box and
point.

## 2. Online inputs and forbidden inputs

At decision time $t$, the public context is

\[
x_t=(q,o_{\le t},h_{<t}),
\]

where:

- $q$ is the original user request;
- $o_{\le t}$ is the current and recent public RGB observation sequence, plus optional
  public proprioception; and
- $h_{<t}$ is the public executed-action and observation history.

The online path may consume only fields that would be available at deployment. It must
reject simulator segmentation, semantic or instance IDs, hidden object poses, drawer
joint truth, task predicates, evaluator target masks, and oracle target locations.
Concrete collectors must also audit provenance because a schema cannot detect private
state deliberately encoded into a nominally public text or tensor field.

### 2.1 Frozen multimodal token input

A token provider exposes

\[
H_t=E_{\mathrm{VLM}}(x_t),
\qquad H_t\in\mathbb R^{N\times D}.
\]

In addition to the token tensor and validity mask, its contract must retain sufficient
metadata to map a prediction back to a public observation:

- camera ID;
- temporal/frame ID;
- current-frame indicator;
- visual-patch support;
- normalized patch coordinates or patch `xyxy`; and
- `provider_id`, which must identify the checkpoint and preprocessing version.

The v1 core accepts frozen token tensors through a protocol. A concrete Qwen, Molmo, or
other VLM adapter is a separate integration artifact and is not implied by the tensor
tests. Candidate proposal and grounding are a second, separately missing upstream
adapter; the token-provider protocol does not implement them.

## 3. Complete grounded interventions

The unit of decision is not a bare action verb. Candidate $u_j$ is

\[
u_j=(m_j,\rho_j,\eta_j),
\]

where:

- $m_j$ is a registered primitive;
- $\rho_j$ binds that primitive to a referent in a specific public frame and camera;
  and
- $\eta_j$ records high-level parameters that distinguish its intended physical
  effect.

Examples include:

```text
(OPEN, middle drawer below the countertop, pull outward)
(OPEN, bottom drawer below the countertop, pull outward)
(REMOVE, red bottle in front of the cereal box, place left)
(ROTATE, blue carton beside the bowl, label toward camera)
(DIRECT, object requested by the prompt, execute the original task)
(STOP, no admissible continuation, do not call the executor)
```

`GroundedIntervention` carries exactly the following decision fields:

```text
candidate_id
primitive
referent
parameters
grounding = (camera, frame_id, frame_index, image_sha256, box_xyxy, point_xy)
```

Every physical candidate requires that complete current-frame grounding; `STOP` is the
only exception. Serializer/executor identity and the continuation horizon belong to the
separate `OutcomeContract` and execution chain, not to `GroundedIntervention`.

The current release validates and consumes candidates that are already proposed and
bound. The upstream proposal/grounding adapter is not implemented.

`GroundedCandidateEncoder` consumes a `CandidateTokenField` whose
`grounding_support` is already fixed. Its attention is restricted to that support; it
does not learn to find the referent, produce a box or point, or decide which patches
belong to a candidate. Multiple candidates may share the same primitive while referring
to different instances.

### 3.1 Candidate identity chain

One immutable `candidate_id` must refer to the same physical intention across the whole
pipeline:

```text
proposal
  -> model input
  -> predicted outcome/value
  -> selected candidate
  -> serialized subtask
  -> executor receipt
  -> post-action observation
  -> evaluator outcome
```

No component may reconstruct a candidate from only its primitive label. The execution
receipt must preserve the selected `candidate_id`, serializer identity, executor
identity, public pre-observation identity, and public post-observation identity. A
correct primitive applied to the wrong instance is an interface/binding failure, not a
successful decision.

## 4. Candidate-conditioned outcome model

For every complete feasible candidate, the Stage-1 model predicts one bounded
task-success logit

\[
s_\theta(H_t,u_j),
\qquad
\hat V_\theta(x_t,u_j)=\sigma\!\left(s_\theta(H_t,u_j)\right).
\]

The binary target $G$ is the single primary outcome named by `OutcomeContract` and is
measured under its frozen continuation policy and horizon. It is an evaluator target,
never an online input or a claimed latent world state.

The policy-relative value is

\[
V_q^{\pi_C}(x,u)=
\mathbb E\!\left[
G_q(\tau)\mid
x,\operatorname{do}(u;\sigma,\pi_E,v_E),\pi_C,H
\right],
\]

where:

- $\sigma$ is the frozen serializer;
- $\pi_E$ and $v_E$ identify the frozen executor and its decoding/action contract;
- $\pi_C$ is the frozen continuation/evaluation policy; and
- $H$ is the fixed evaluation budget.

An outcome collected with another serializer, executor checkpoint, action horizon, or
continuation policy belongs to a different intervention distribution and cannot be
silently reused.

### 4.1 Required model outputs

`OutcomePrediction` contains exactly one learned tensor:

```text
task_success_logits : [batch, candidates]
```

It also carries `candidate_valid_mask`, candidate IDs and fingerprints, and context
fingerprints for identity checking. Invalid candidates are masked before selection.
There are no branch logits or `grounding_logits`; support-constrained attention is an
encoder operation over pre-bound support, not a learned grounding output.

## 5. Training-data contract

Formal outcome supervision comes from reset-controlled execution branches. For one
initial-state group $g$, candidate $u_j$, and repetition $r$, record

\[
(x_g,u_j,x^+_{gjr},G_{gjr},C_{gjr}),
\]

where $C$ stores decomposed diagnostics such as primitive success, first-contact
identity, disturbance, number of steps, timeout, and infrastructure failure.

Before reading any outcome, freeze:

- candidate-generation version and maximum candidate count;
- candidate identity and grounding schema;
- serializer version;
- executor checkpoint, decoding, action horizon, and controller version;
- continuation/evaluation policy and evaluation horizon;
- primary outcome and any deterministic tie-break;
- feasibility and safety rules;
- number of repetitions and failure handling;
- execution randomization; and
- initial-state-group-disjoint train, development, calibration, and sealed-test splits.

All prompt variants and candidate branches derived from one physical initial state stay
in the same split group. Public transitions and privileged evaluator sidecars remain
separate and hash-linked.

Current historical counterfactual files may be used as loader fixtures, but their
immediate post-action `task_success` field is not automatically a valid target for
$V_q^{\pi_C}$. In particular, an `OPEN` branch that ends before a frozen continuation
policy acts cannot be labelled as having zero long-horizon task value merely because the
final task is not yet complete.

## 6. Learning objectives

The first implementation is strictly result-only.

### Outcome likelihood

\[
\mathcal L_{\mathrm{outcome}}
=-\sum_{g,j,r}
\left[
G_{gjr}\log \sigma(s_{gjr})
+(1-G_{gjr})\log(1-\sigma(s_{gjr}))
\right].
\]

This Bernoulli NLL reads only the receipt-backed candidate actually executed in each
row. Unexecuted alternatives never contribute labels or loss. Probabilistic claims must
be evaluated with proper scoring rules such as NLL or Brier score on group-disjoint
data. Formal v1 has no branch, grounding, or ranking auxiliary.

For `STOP`, “executed” means the episode is genuinely terminated and evaluated under
the same outcome contract. It receives a local termination receipt and a fresh public
observation, does not call the VLA, and is never filled in as an assumed negative.

## 7. Selection

Let $\mathcal F(x_t)$ be the finite set of candidates that are:

- structurally valid;
- grounded when the primitive requires grounding;
- supported by the frozen executor interface;
- qualified for the declared context; and
- compliant with hard safety and task constraints.

The first version chooses

\[
u_t^*=\arg\max_{u\in\mathcal F(x_t)}\hat V_\theta(x_t,u).
\]

`DIRECT` and `STOP` are members of the same candidate set. A preregistered cost or stable
candidate order resolves exact ties. The core selector contains no post-hoc weighted
uncertainty sum, confidence threshold, or singleton-conformal rule.

If the feasible set is empty, the software fails closed. It must not invent an action or
fall back to an unqualified primitive.

## 8. Frozen VLA boundary and actual reobservation

The selected candidate is serialized into an instance-distinguishing public subtask.
The serializer consumes the same complete candidate that the scorer ranked. The first
formal interface is text-based; a future point, region, soft-token, or K/V adapter would
change the executor contract and must receive its own version and evaluation.

The frozen executor consumes only its declared public observation, public
proprioception, and serialized subtask, and returns a continuous action chunk. The
software Protocol does not imply that a real MolmoAct2 server is available.

After execution, the method encodes the **actual** new public observation:

\[
H_{t+1}=E_{\mathrm{VLM}}(q,o_{\le t+1},h_{\le t}).
\]

The executed action, receipt, and observation are appended to history before the next
decision. A predicted latent state may never replace actual reobservation in the closed
loop. Policy history contains only the issued public subtask, primitive, step index,
and deployment-visible execution status. Candidate fingerprints, request/receipt
digests, and evaluator-only semantic branch labels remain in the audit trace rather
than becoming model input; the next decision must infer what changed from public frames
and public execution status.
Hidden-content tests must verify that identical pre-action inputs can lead to
different second actions after `TARGET_REVEALED`, `ABSENCE_RESOLVED`, distractor, or
failed-execution outcomes.

## 9. Deferred mechanisms

The following mechanisms are explicitly outside formal v1:

- future-latent or JEPA targets;
- EMA target encoders;
- stochastic or multi-hypothesis world models;
- conformal prediction sets;
- EDL/Dirichlet evidence;
- multi-step tree search;
- learned soft-token or K/V injection into the executor;
- branch, grounding, or ranking auxiliary losses;
- Stage-2 fine-tuning; and
- end-to-end reinforcement learning.

They may be introduced only after the result-only model and physical identity chain work.
Future prediction, if studied, requires three distinct controls: no future path; equal
future path with no future-target loss; and equal future path with future-target loss.
Calibration, if studied, is a secondary risk layer and uses independent initial-state
groups; it does not define the central action policy.

## 10. Historical boundary

Formal v1 has no runtime dependency on the superseded pipelines. Their exact source and
historical artifacts remain recoverable from Git snapshot `8c4be631`; none of those
artifacts is formal-v1 or MolmoAct2 evidence.

## 11. Minimum software verification

Before any empirical claim, unit and integration fixtures must verify:

- public-input firewall rejection;
- strict tensor ranks, masks, finite values, and metadata consistency;
- multiple candidates with the same primitive and different referents;
- grounding exists before scoring;
- changing one candidate changes only its own prediction in the synthetic isolation
  test;
- invalid candidates can never be selected;
- the selected `candidate_id` survives serialization and receipt creation;
- frozen token inputs do not receive gradients;
- executed-candidate Bernoulli NLL backpropagates into the trainable outcome model;
- deterministic tie handling;
- `STOP` never calls the executor;
- no outcome label or evaluator field enters model input; and
- reobservation, rather than a predicted future, supplies the next online context.

Random tensor tests establish software contracts only. They are not VLM integration,
grounding accuracy, candidate-value accuracy, physical binding fidelity, or task-success
evidence.

## 12. Evidence ladder

The maximum defensible claim advances only as follows:

| Evidence | Maximum interpretation |
| --- | --- |
| tensor and synthetic fixture tests | formal v1 software contracts are internally consistent |
| real frozen-VLM tokens | the concrete Stage-1 adapter is connected |
| oracle candidate-to-executor test | the declared interface has a measurable physical ceiling |
| learned proposal and binding test | Stage 1 can specify the intended physical instance |
| reset-controlled outcome data | candidate outcomes can be learned under the frozen executor contract |
| randomized hidden-result branching | the loop uses new evidence to change its second action |
| paired closed-loop comparison | the method improves the preregistered task outcome under a fixed budget |
| held-out benchmark and second executor | the claim extends beyond one scene/executor contract |

At the current software-only stage, only the first row may be claimed after the tests
pass.
