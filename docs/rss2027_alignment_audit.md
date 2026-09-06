# RSS 2027 method-alignment audit and corrected research contract

## Status and decision

This document supersedes `rss2027_method_redesign.md` as the active scientific
contract. It does not claim that the corrected method has been implemented, trained, or
validated.

The v0.1 latent router remains useful as a reproducible baseline, but its defensible
description is only:

> a high-level primitive policy with an untyped action-description-conditioned future-feature
> auxiliary path, followed by a frozen text-conditioned VLA executor.

It is not yet a value-of-information learner, a complete counterfactual action model,
or a token-level conditioning method inside MolmoAct2.

The corrected RSS 2027 question is:

> When task-relevant state is not directly observable, can a robot learn which
> **grounded physical intervention** to perform and then use the actually revealed
> observation to change its next action?

The working paper position is:

> **Learning grounded intervention outcomes for interactive perception with a frozen
> VLA executor.**

This is a learned high-level interactive-perception policy over a frozen VLA, not a new
end-to-end VLA architecture. A future soft-token or typed-geometry adapter would be a
separate system variant and would require its own training and claim.

## 1. Why the previous theory, supervision, and interface did not align

| Intended claim | What v0.1 actually computes | Unsupported jump | Correction |
| --- | --- | --- | --- |
| compare physical interventions | the documented method uses one action-description token per registered branch; its tensor contract permits free text but does not type the referent or parameters | a phrase such as `open middle drawer` may distinguish language descriptions, but the predicted visual instance is not an input to the effect model | construct a complete grounded candidate before effect or outcome prediction |
| learn task-relevant information value | route classification plus deterministic post-feature cosine similarity | future predictability is neither task usefulness nor uncertainty reduction | supervise candidate outcomes or task-risk reduction from actual branches |
| ground an executed action | learned grounding is decoded after future prediction | the predictor may read referent words, but it never receives its own visual referent distribution and therefore has no explicit, testable instance binding | make grounding part of the candidate representation |
| constrain frozen MolmoAct2 | send only RGB, state, and serialized text | the predicted region never enters the stock action model | use instance-distinguishing text and measure end-to-end binding fidelity |
| handle multiple acceptable actions | use a singleton-only conformal prediction set | two good actions can force an unnecessary abstention | remove conformal from the core; optionally calibrate false inclusion after learning |
| prove the future module matters | call `lambda_J=0` “no predictor” | route gradients still traverse the predictor | separate no-path, path-without-target-loss, and path-with-target-loss ablations |

Two general lessons follow.

First, a latent is not a belief distribution merely because it is called `belief`.
Second, a prediction about what may be observed after an action is not automatically a
prediction that the action is worth taking.

## 2. The complete unit of decision: a grounded intervention

The public policy context is

\[
x_t=(q,o_{\le t},h_{<t}),
\]

where `q` is the complete user request, `o` contains only public observations, and `h`
contains only public action/observation history.

A candidate is not a bare verb. It is

\[
u_j=(m_j,\rho_j,\eta_j),
\]

where:

- \(m_j\) is an interaction type such as `OPEN`, `REMOVE`, `ROTATE`, or
  `BRING_CLOSE`;
- \(\rho_j\) binds the action to a referent in the current public image, represented by
  a distribution over visual patches or a derived region; and
- \(\eta_j\) contains only the high-level parameters needed to distinguish effects,
  such as `middle drawer`, `pull outward`, `place left`, or `rotate label forward`.

`DIRECT(q)` and `STOP` are included in the same candidate set. Continuous robot
trajectories are not Stage-1 parameters; they remain the responsibility of the VLA.

Examples are:

```text
(OPEN, middle drawer below the countertop, pull outward)
(OPEN, bottom drawer below the countertop, pull outward)
(REMOVE, red bottle in front of the cereal box, place left)
(ROTATE, blue carton beside the bowl, label toward camera)
(DIRECT, target described by q, execute q)
(STOP, no resolvable grounded intervention, no execution)
```

The realized intervention distribution also depends on the frozen interface and
executor:

\[
X^+\sim P_{\mathrm{env}}\!\left(
\cdot\mid x,\operatorname{do}(u;\sigma,\pi_E,v_E)
\right),
\]

where \(\sigma\) is the frozen text serializer, \(\pi_E\) is the executor, and \(v_E\)
binds its checkpoint, decoding, action horizon, and controller version. An outcome
collected with a scripted controller belongs to a different intervention distribution
and cannot be silently treated as a MolmoAct2 outcome.

## 3. Minimal two-stage pipeline

```text
public RGB/history + prompt + public action history
                         |
                  frozen Stage-1 VLM
                         |
                multimodal token field H_t
                         |
       grounded interaction candidate proposals
        u_j = (primitive, referent, parameters)
                         |
       candidate-conditioned outcome/value scorer
          + optional future-latent auxiliary path
                         |
       select a feasible candidate or abstain
                         |
       frozen referential-text serialization
                         |
                 frozen MolmoAct2
                         |
           action chunk -> reobserve -> repeat
```

Only Stage 1 learns the active-perception decision. Stage 2 realizes its selected
subtask. The loop is not complete unless the post-action RGB enters the next Stage-1
decision and changes the next action when the hidden result changes.

### 3.1 Frozen multimodal tokens

\[
H_t=E_{\mathrm{VLM}}(x_t).
\]

`H_t` retains prompt, visual patch, temporal, and public-history information. It is not
called uncertainty. A small cross-attention module may extract task-conditioned state
tokens \(B_t\), but those tokens earn no probabilistic interpretation by themselves.

### 3.2 Grounded set proposal before prediction

Use multiple learned proposal queries per interaction type, rather than one token per
verb. For query \((k,m)\):

\[
\rho_{k,m}=\operatorname{softmax}
\left(q_{k,m}K_{\mathrm{patch}}^\top\right),
\qquad
r_{k,m}=\sum_p \rho_{k,m,p}V_{\mathrm{patch},p}.
\]

The complete candidate token is

\[
c_{k,m}=f_\theta
\left(B_t,E_m(m_k),r_{k,m},E_\eta(\eta_{k,m})\right).
\]

Outcome, route, and any optional future prediction must all condition on \(c_{k,m}\).
This order allows the model to compare three drawers or several occluders of the same
primitive type.

The proposer may use masks, instance IDs, and simulator geometry only as private
training/evaluation labels. At test time its input ends at public tokens. Learned
proposal recall and an oracle-candidate upper bound must be reported separately.

### 3.3 Candidate outcome model

The core scalar is not a hand-composed uncertainty score. The model predicts a
candidate's policy-relative task outcome:

\[
V_q^{\pi_C}(x,u)=
\mathbb E\left[
G_q(\tau)\mid x,\operatorname{do}(u;\sigma,\pi_E,v_E),\pi_C,H
\right],
\]

where \(G_q\) is a preregistered bounded task outcome, \(\pi_C\) is a frozen
continuation/evaluation policy, and \(H\) is a fixed budget. A simple first version uses
the probability of completing the requested task within the budget while satisfying
hard safety and disturbance constraints. Feasible ties are broken by a preregistered
cost or deterministic order, not by a post-hoc weighted uncertainty sum.

The selected candidate is

\[
u_t^*=\arg\max_{u\in\mathcal F(x_t)}\hat V_\theta(x_t,u),
\]

where \(\mathcal F\) contains only executor-qualified and task-contract-compliant
candidates.

A useful diagnostic is the information-intervention advantage

\[
\Delta_{\mathrm{info}}(x)
=\max_{u\in\mathcal F\cap\mathcal U_{\mathrm{info}}}\hat V_\theta(x,u)
-\hat V_\theta(x,\operatorname{DIRECT}).
\]

This is a learned, policy-relative outcome advantage. It must not be renamed mutual
information or environment uncertainty.

## 4. Where the supervision comes from

The action policy cannot be defined by undocumented labels. For each reset-identical
initial-state group \(g\), execute candidate \(u_j\) under the frozen intervention
contract for \(r=1,\ldots,M\) repetitions:

\[
(x_g,u_j,x^+_{gjr},S_{gjr},C_{gjr}).
\]

`S` is the preregistered task or post-interaction decision outcome; `C` records
execution success, disturbance, steps, and other decomposed diagnostics. Before any
outcome is observed, freeze:

- candidate-generation version and maximum candidate count;
- serializer and executor identity;
- continuation/evaluator and horizon;
- safety/feasibility constraints;
- the primary outcome and any deterministic tie-break;
- repetitions \(M\), randomization, and failure handling; and
- initial-state-group-disjoint train, development, calibration, and sealed-test splits.

The primary prediction can be trained directly on Bernoulli rollout outcomes:

\[
\mathcal L_{\mathrm{out}}
=-\sum_{g,j,r}
\left[S_{gjr}\log\sigma(v_{gj})
+(1-S_{gjr})\log(1-\sigma(v_{gj}))\right].
\]

When candidates have replicated bounded returns, add an optional within-group ranking
loss only for empirically separated pairs:

\[
\mathcal L_{\mathrm{rank}}
=\sum_{\bar G_{gj}>\bar G_{g\ell}+\epsilon}
\operatorname{softplus}\!\left(-(v_{gj}-v_{g\ell})\right).
\]

This is **supervised counterfactual outcome modeling**, not reinforcement learning: it
has no temporal-difference bootstrap, policy gradient, or on-policy policy-improvement
loop. If only expert/canonical route labels are available, the method must instead be
called a grounded high-level behavior-cloning policy.

An acceptable-action set, if needed, is derived transparently after outcome collection:

\[
Y_g^\epsilon=
\left\{u_j\in\mathcal F_g:
\bar G_g(u_j)\ge \max_{u\in\mathcal F_g}\bar G_g(u)-\epsilon
\right\}.
\]

`Y` is therefore a dataset product with an audited generation procedure, not an
unexplained source of strategic knowledge.

## 5. What future-latent prediction may and may not do

An optional candidate-conditioned target is

\[
\hat B^+_j=P_\theta(B_t,c_j),
\qquad
\mathcal L_{\mathrm{future}}
=1-\cos\!\left(
\hat B^+_{u_{\mathrm{executed}}},
\operatorname{stopgrad}(\bar B^+)
\right).
\]

Its only initial interpretation is **action-conditioned representation regularization**.
It may help encode predictable post-action structure. It does not establish that the
action reduces uncertainty, supplies useful information, or improves the task.

The opaque-box counterexample is decisive: opening a box has high information value
when its hidden content is random, yet the exact future feature is multimodal and hard
to predict; doing nothing is easy to predict but uninformative. A single cosine target
can average modes. If stochastic future structure later becomes necessary, compare a
distributional or multi-hypothesis model, but only after real-outcome supervision wins
the minimal experiment.

The mechanism ablation must distinguish:

1. **no future path:** the scorer reads \((B_t,c_j)\) and no predictor exists;
2. **future path, no future-target loss:** the predictor remains and receives outcome
   gradients, but \(\lambda_F=0\); and
3. **future path plus target supervision:** the identical predictor also receives
   \(\mathcal L_{\mathrm{future}}\).

An equal-parameter control is required. EMA and stop-gradient stabilize a target branch
but do not guarantee non-collapse.

## 6. Execution binding is an empirical contract

The stock MolmoAct2 interface accepts images, task text, and proprioceptive state, then
returns an action chunk. It exposes no documented input for the Stage-1 patch
distribution, point, region, or arbitrary belief/K/V tokens. Consequently, the first
system serializes the complete candidate into an instance-distinguishing instruction:

```text
Open the middle drawer below the countertop.
```

The region is used to construct/supervise the candidate and audit its identity; it is
not a hard constraint on stock MolmoAct2. The paper must report the binding chain:

\[
\text{candidate proposal}
\rightarrow\text{grounded referent}
\rightarrow\text{serialized expression}
\rightarrow\text{executor contact identity}.
\]

`Binding Fidelity` is the fraction of trials in which the executor first acts on the
same physical instance selected by Stage 1. A correct Stage-1 candidate followed by
wrong-instance contact is an interface failure, not a routing failure.

If text is insufficient, the next system may use a typed point/region interface or a
trained latent adapter. That changes the executor contract and must not be described as
an unchanged stock frozen-VLA interface. [Pointing-VLA](https://arxiv.org/abs/2608.23138)
is a direct architectural precedent for typed hidden-state spatial readouts; therefore
the typed interface itself is not the novelty claim.

## 7. Calibration is optional and secondary

The core method first has to learn the behavior. The v0.1 rule

```text
prediction set has exactly one element -> execute
otherwise -> abstain
```

is removed from the central contribution because multiple good information actions do
not imply ignorance.

If calibration is retained, apply it after the proposer, scorer, serializer, and
executor identities are frozen. The independent calibration unit is an initial-state
or episode group, not an action branch, prompt paraphrase, frame, or camera view.

One optional autonomous-system formulation is a calibrated inner candidate set. Let
\(Y_i\) be the acceptable candidates and let larger \(v(x,u)\) mean better. On each
calibration group define

\[
M_i=\max_{u\in\mathcal C_i\setminus Y_i}v(x_i,u),
\]

using \(-\infty\) if no unacceptable candidate exists. With an appropriate finite-sample
quantile \(\hat q\), define

\[
\mathcal S_\alpha(x)=\{u:v(x,u)>\hat q\}.
\]

Under the required exchangeability assumptions, this targets marginal false-inclusion
control rather than conditional correctness after execution. A nonempty set can use a
preregistered tie-break; an empty set abstains. Report false inclusion, nonempty rate,
selection rate, and sealed-test selective error separately.

This optional construction follows the limited-false-positive/inner-set direction of
[Fisch et al. (ICML 2022)](https://proceedings.mlr.press/v162/fisch22a.html); its exact
finite-sample theorem and tie handling must be stated for the implementation actually
used, rather than inferred from ordinary outer-set coverage.

Single-step marginal calibration is not an episode-success guarantee because each
action changes the next input distribution. The first paper should report step-level
calibration and closed-loop empirical confidence intervals rather than claim a
trajectory-level conformal theorem.

## 8. Novelty boundary after current literature

The paper cannot rely on token-level fusion, future latents, subtask text, memory, or
outcome learning alone:

- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) is a strong reminder that simple,
  carefully controlled action imitation is a necessary baseline.
- [pi0.5](https://arxiv.org/abs/2504.16054) already predicts semantic subtasks and then
  continuous actions.
- [CoMe-VLA](https://arxiv.org/abs/2602.04600) already uses temporal context and a
  learned cognitive head for active-perception subtask transitions.
- [LaWAM](https://arxiv.org/abs/2606.15768) and
  [StageWAM](https://arxiv.org/abs/2608.10780) already condition robot control on
  compact predicted future or next-stage latents.
- [ThinkAct](https://arxiv.org/abs/2507.16815) already connects reinforced high-level
  visual-plan latents to an action model.
- [RECAP / pi*0.6](https://arxiv.org/abs/2511.14759) already learns from task outcomes,
  value, and advantage-conditioned policy improvement.
- [Pointing-VLA](https://arxiv.org/abs/2608.23138) already supplies typed spatial
  interfaces between reasoning and execution.

The defensible conjunction to test is narrower:

> prompt-conditioned partial observability + complete grounded interaction candidates
> + reset-controlled candidate outcomes + closed-loop branching on newly revealed
> observations + execution by the same frozen VLA.

This is a hypothesis until it beats strong history-conditioned imitation baselines.

## 9. Three decisive experiments before full-scale development

### E1. Grounded candidate and executor binding

Place several same-type affordances in one image: multiple drawers, bottles, cartons,
or removable occluders. Construct candidate pairs that differ only in referent or
parameter. From reset-identical initial states, execute each selected candidate.

Report candidate recall, primitive accuracy, referent/region accuracy, text referent
accuracy, intended-instance first-contact rate, primitive success, and post-action
information effect. Decompose every failure at the first broken link.

**Gate:** if the text command does not preserve Stage-1 instance identity often enough,
do not run the large outcome study; test a typed interface or change the claim.

### E2. Hidden-content randomized branching

Keep prompt, pre-action RGB, robot state, and the correct first intervention identical.
Randomize only hidden content:

```text
target revealed   -> grasp/continue the user task
empty receptacle  -> search the next admissible location
distractor found  -> ignore, inspect, or continue as the prompt requires
search exhausted  -> STOP/report not found
```

The first `OPEN` may be identical; the second action must differ after reobservation.
Add post-action-frame shuffling and cross-episode-history shuffling as negative controls.

Report next-action branch accuracy, final task success, unnecessary exploration, false
not-found, target-revealed-then-continue-search, and empty-then-grasp rates.

**Gate:** a system that cannot branch on hidden-result randomization has not demonstrated
interactive perception, even if it opens the drawer reliably.

### E3. Controlled source-of-supervision study

Hold the frozen VLM, candidate proposer, candidate count, history window, executor,
trajectory groups, split, and parameter budget fixed. Compare:

1. history-conditioned route/behavior cloning only;
2. future path plus deterministic post-feature supervision;
3. real-outcome supervision with no future path; and
4. real-outcome plus future-target supervision.

Also compare a history-conditioned standard VLA or the closest compute-matched direct
action baseline trained on the same trajectories.

**Interpretation:**

- if result-only matches result-plus-future, delete future prediction;
- if the history-conditioned standard VLA matches the full system, make the benchmark,
  not the router, the main contribution;
- if the full model only beats a smaller model, the result is a capacity effect;
- only a controlled gain of future-target supervision over an equal-path/no-target-loss
  model supports the future-representation mechanism.

## 10. Benchmark requirements implied by the method

BenchV1 must contain more than task templates. Its identifying unit is a grouped set of
physical interventions from a matched initial state.

Each group must support:

- identical public pre-action input with randomized hidden content;
- multiple complete candidates, including same-primitive/different-referent cases;
- clear `DIRECT` controls so that opening everything is not rewarded;
- exhausted/not-found `STOP` cases;
- real failed-primitive cases retained rather than filtered;
- public pre/post histories separated from evaluator-only state;
- action-branch and prompt variants kept within one split group; and
- separate labels for proposal, binding, execution, information effect, branching, and
  terminal success.

Same-scene/different-prompt pairs test language conditioning. They do not by themselves
establish causal action effects. Causal comparisons require reset-controlled candidate
branches or a justified logged-policy/propensity design.

## 11. Required baselines and fairness rules

### System-level baselines

1. frozen MolmoAct2 direct execution;
2. the same executor with the same public RGB/action history appended;
3. a history-conditioned standard VLA fine-tuned or parameter-efficiently adapted on
   the same trajectories, with OpenVLA-OFT as the methodological reference;
4. prompted VLM to referring subtask to frozen MolmoAct2;
5. a CoMe-style binary subtask-transition policy;
6. a simple grounded high-level behavior-cloning policy;
7. Heuristic V0 and the v0.1 latent-router scaffold; and
8. the grounded real-outcome model.

### Mechanism-level ablations

- no history;
- no prompt or mismatched prompt;
- oracle candidate set versus learned candidate proposal;
- no future path;
- future path with no post-feature target loss;
- future path with target loss;
- route/expert labels versus actual outcome supervision;
- text binding versus any later typed interface; and
- calibration applied identically to every score-based router as a secondary layer.

Report trajectory count, unique initial-state groups, counterfactual branches, private
label types, trainable parameter count, GPU hours, and executor calls. “Same data” is
not a valid claim if one method receives additional outcome labels or branches without
that accounting.

## 12. Immediate implementation order

1. Freeze v0.1 as a baseline; do not enlarge its future predictor or run a large S03
   outcome campaign for the RSS method.
2. Replace `PrimitiveTokenBatch` in a new package/version with a typed
   `GroundedInterventionCandidate`; grounding must exist before scoring or prediction.
3. Build E1 on the smallest multi-drawer and multi-object scenes; measure binding all
   the way to the physical contact made by MolmoAct2.
4. Build E2 with randomized hidden content and post-observation branching; run the
   history-conditioned standard policy and simple grounded-policy baselines first.
5. Freeze the candidate-outcome data contract and collect only the reset branches needed
   for E3.
6. Train result-only before future-result. Add JEPA only if it wins the controlled
   ablation.
7. Add calibration only after ranking and branching work; calibrate by independent
   initial-state groups.
8. Expand primitives and OOD splits only after E1--E3 pass.

## 13. Go/no-go claim ladder

| Evidence reached | Maximum defensible claim |
| --- | --- |
| OPEN primitive qualification only | the executor can often open this drawer |
| candidate proposal and binding pass | Stage 1 can specify which physical intervention is intended |
| hidden-result branching passes | the loop uses newly acquired evidence to change behavior |
| outcome supervision beats route/future controls | candidate outcomes, not labels or capacity alone, improve selection |
| result-plus-future beats equal-path result-only | future-target supervision contributes an additional mechanism |
| strong history-VLA baseline is beaten on held-out groups | the proposed high-level decomposition is empirically necessary |
| typed/text binding and closed-loop sealed tests pass | the full frozen-executor system supports the RSS method claim |

If the method gates fail while the benchmark exposes a stable failure of strong VLAs,
the project should become an RSS benchmark paper. That is a successful scientific
outcome, not a reason to preserve an unsupported method narrative.
