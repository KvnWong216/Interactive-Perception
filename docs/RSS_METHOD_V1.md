# Prompt-Aligned Action-Resolvable Uncertainty

## Claim

A frozen VLA may possess a useful action without knowing when that action is
needed. We add a lightweight decision layer that represents *what information
the prompt is missing*, predicts which embodied action can resolve it, and
chooses that action only when its calibrated reduction in task risk exceeds
its cost.

This is not a scalar-confidence router and not a new action decoder. The
`pi05_libero` checkpoint stays frozen.

### Current minimal T01 execution contract

The first embodied closure does not use the probabilistic progress belief
described below. It deliberately restricts the world state to `OBSERVED` and
`MANIPULATION_ONLY`, and uses deterministic control memory
`m_t=(phase,searched_set,attempt_counts)`. `A_valid(m_t)` is enforced before
planning. The v5 runner records `probabilistic_temporal_progress_used: false`.
The richer product-belief formulation remains future method scope until this
minimal RGB outcome/belief-update loop passes.

## System

```text
stock RGB + prompt + public history
                  |
          frozen pi0.5 prefix
                  |
       typed target-state belief
                  +
     probabilistic progress automaton
                  |
       conformal plausible set
                  |
    calibrated action-effect registry
                  |
       finite-horizon Bayes risk
          /       |        \
        ACT   information   NOT_FOUND / SAFE_STOP
                  |
          frozen pi0.5 option
                  |
       six-point RGB/state outcome
                  |
              belief update
```

The state variable is

\[
z_t \in \{\mathrm{OBSERVED},\mathrm{VIEWPOINT\_BLOCKED},
\mathrm{MANIPULATION\_ONLY},\mathrm{ABSENT}\}.
\]

Given prompt \(x\) and public history \(h_t\), the belief head estimates

\[
b_t(z)=P(z_t=z\mid x,h_t).
\]

A Mondrian split-conformal calibrator returns a class-conditional plausible
set \(\Gamma_\alpha(x,h_t)\), with \(\alpha=0.1\) frozen before audit. The set
is a coverage object, not a claim that the discriminative head's softmax is a
calibrated probability.

For information action \(a\), the effect model has observable branches

\[
y\in\{\mathrm{FAILED},\mathrm{REVEALED},\mathrm{EMPTY}\},
\qquad P(y\mid b_t,a,h_t).
\]

The outcome is defined by what the policy could acquire, not just by the final
frame. `REVEALED` means the target became visible at any of six public history
points. `EMPTY` requires that the target was never seen, the searched region
remained open, return-to-observe completed, and an independent view-coverage
certificate says that location would have been visible. `FAILED` means the
option did not certify either result. `EMPTY` is local evidence about one
searched region, never a global `NOT_FOUND` decision. Simulator joints and
segmentation construct offline labels only; the online critic sees six
stock-camera frames, six public robot-state vectors, the prompt, and the
executed option role.

The critic is hierarchical. Its first conformal head predicts physical
`FAILED` versus `COMPLETED`; conditional on completion, its second head predicts
`REVEALED` versus `EMPTY`. This prevents motor completion and prompt-relevant
content from being collapsed into one visually confounded class. If either
head returns an ambiguous set, the controller preserves the ambiguity and
uses `SAFE_STOP`; it never selects the convenient member with a threshold.

The posterior is updated from the observed physical effect:

\[
b_{t+1}(z) \propto b_t(z)P(y_t\mid z,a_t,h_t).
\]

### Prompt-conditioned temporal progress

World belief alone does not say whether an action is legal *now*. We therefore
maintain a second distribution over a finite-trace task automaton,

\[
\mu_t(q)=P(q_t=q\mid x,h_t),
\]

with generic phases `NEEDS_EVIDENCE`, `AWAITING_EFFECT`, `READY_TO_COMMIT`,
`SEARCH_EXHAUSTED`, `COMMITTING`, and terminal or violation states. The runtime
state is the product \((b_t,\mu_t)\): \(b_t\) describes the uncertain world,
while \(\mu_t\) describes uncertain task progress.

The automaton never contains a benchmark answer such as `T01 -> OPEN_DRAWER`.
It contains only prompt-independent action roles and prompt-derived terminal
predicates:

\[
\mathbf G(\mathrm{COMMIT}\rightarrow\mathrm{EVIDENCE\_SUFFICIENT}),
\qquad
\mathbf G(\mathrm{NOT\_FOUND}\rightarrow\mathrm{SEARCH\_EXHAUSTED}).
\]

The target-state and outcome heads supply calibrated proposition evidence. An
observed `REVEALED` effect moves progress toward `READY_TO_COMMIT`; `FAILED`
returns it to `NEEDS_EVIDENCE`; `EMPTY` reaches `SEARCH_EXHAUSTED` only when all
calibrated search hypotheses have been exhausted. This prevents an open but
unobserved drawer from being treated as a completed search.

For action role \(r(a)\), the automaton supplies a probability of violating the
current finite trace,

\[
P_{\mathrm{viol}}(a\mid\mu_t)
=\sum_q \mu_t(q)\mathbf 1[\delta(q,r(a))=q_{\mathrm{fail}}].
\]

This enters the same risk unit as other task losses:

\[
\widetilde Q(b,\mu,a)=Q(b,a)+
L_{\mathrm{logic}}P_{\mathrm{viol}}(a\mid\mu).
\]

`SAFE_STOP` is an explicit deferral or assistance action with a declared task
cost. It is distinct from `NOT_FOUND`: when evidence is insufficient and no
certified information action exists, the system must not call absence merely
because all executable choices are bad.

Terminal risks are

\[
R_{\mathrm{ACT}}(b)=
P(z\neq\mathrm{OBSERVED})L_{\mathrm{commit}}+
P(z=\mathrm{OBSERVED})(1-r_{\mathrm{ACT}})L_{\mathrm{fail}},
\]

\[
R_{\mathrm{NOT\_FOUND}}(b)=
(1-P(z=\mathrm{ABSENT}))L_{\mathrm{miss}}.
\]

An information action is valued by

\[
Q(b,a)=c(a)+\sum_y P(y\mid b,a)V(b^{a,y}),
\]

and the planner recursively minimizes

\[
V(b,\mu)=\min\left\{\widetilde R_{\mathrm{ACT}},
\widetilde R_{\mathrm{NOT\_FOUND}},R_{\mathrm{SAFE\_STOP}},
\min_a \widetilde Q(b,\mu,a)\right\}.
\]

This gives a decision-grounded definition of *action-resolvable uncertainty*:

\[
U_{\mathrm{res}}(a\mid b,\mu)=V_{\mathrm{stop}}(b,\mu)-
\widetilde Q(b,\mu,a).
\]

Here \(V_{\mathrm{stop}}\) is the lowest risk among `ACT`, justified
`NOT_FOUND`, and explicitly costed `SAFE_STOP` under the current progress
belief. It is not permission to relabel a safe stop as target absence.

It is the task risk that action (a) is expected to remove after paying for
the action and accounting for its measured failures. It is prompt aligned
because (b) is conditioned on the prompt, and action specific because every
candidate has a different effect distribution. The router explores only when
\(\max_a U_{\mathrm{res}}(a\mid b,\mu)>0\); otherwise it takes the lower-risk
terminal decision. Thus uncertainty is not an extra scalar output or a tuned
trigger. It is the value of a possible embodied intervention.

When the conformal set is not a singleton, the robust rule minimizes the
worst risk across retained hypotheses:

\[
a_t^*=\arg\min_a\max_{z\in\Gamma_\alpha}Q(a\mid z).
\]

Therefore high entropy does not automatically trigger interaction. An
information action is selected only when its expected benefit, including its
measured failure probability, exceeds its explicit cost.

The present 296/300 experiment uses **target observability** as its terminal
objective: reaching `OBSERVED` ends the routing episode. It does not substitute
stock `ACT` reliability for final retrieval. For final-task risk, every
observed hypothesis needs a context-specific terminal executor rate. That
strict audit is currently NOT-GO because open-drawer butter retrieval is 0/5
and closed-scene cream-cheese execution has not been measured.

## Model boundary

The belief probe uses the 2,048-dimensional prompt-mean block from the frozen
PaliGemma multimodal prefix. It is upstream of the pi0.5 action expert. This
choice was made on grouped prototype-training folds; probability calibration,
conformal calibration, validation, and audit use disjoint seed blocks.

The outcome critic uses six prefix histories and the executed action role. Its
`FAILED`, `REVEALED`, and `EMPTY` labels may be produced from simulator state
during offline calibration. At deployment, its inputs are only the two stock
policy RGB streams, prompt, action label, public robot state, and public
history.
The immutable v3 diagnostic used a five-pixel visibility endpoint. It exposed
5--7-pixel upper-layer leakage that is simulator-visible but not resolvable by
the frozen policy. The v9 contract instead requires one pi0.5 visual-token
footprint, `(256 / 16)^2 = 256` target pixels, in one stock policy view at any
history point. A later occlusion cannot erase evidence already acquired earlier
in the option.

The first temporal implementation is an exact, inspectable runtime automaton,
not a learned logic embedding. T-LEAF-style DFA embeddings are reserved for a
separate sample-efficiency ablation after reliable physical transitions exist.
This ordering keeps an executor/viewpoint failure from being misreported as a
failure of temporal representation learning.

Environment seeds do not by themselves fix flow-matching samples. OpenPI's
frozen `Policy` initializes its PRNG with `jax.random.key(0)` and splits the key
on every inference. Each development or audit collection therefore starts a
fresh server, performs no prior inference calls, and freezes the exact regime,
seed, and replan order. The manifest records this contract and the final data
hash.

The following are forbidden online:

- drawer joints;
- instance segmentation;
- hidden object poses;
- task success predicates;
- evaluator BEV images.

Evaluator-only signals remain in result files so claims can be checked.

## Why this differs from action spread

Repeated pi0.5 action chunks did not survive prompt/state counterfactuals. On
the same closed-drawer RGB, the old action-intent classifier continued to emit
`REMOVE_OCCLUDER` after the prompt target changed to a visible object. It also
failed to switch to `ACT` when the same target was already revealed. Action
spread was reading policy behavior or scene type, not the location of
prompt-relevant uncertainty.

The current belief is computed before action decoding from `(RGB, prompt)`.
Physical action reliability is then calibrated separately. A correct belief
cannot authorize an incapable executor, and a capable executor does not prove
that the policy knows when to invoke it.

## Relation to CoMe-VLA

[Act, Sense, Act / CoMe-VLA](https://arxiv.org/abs/2602.04600) formalizes active
perception as a history-dependent loop with information-seeking actions and
decision branching. Its taxonomy directly matches the breadth we want to test:
viewpoint discovery, manipulation discovery, and information enrichment. It
learns a cognitive auxiliary head for subtask transitions and a dual-track
visual/proprioceptive memory together with action generation from large-scale
egocentric human and robot data.

CoMe-VLA uses the word uncertainty operationally, not as an estimated random
variable. Its cognitive head predicts one binary state: whether enough
task-relevant information has been obtained. Deployment switches textual
subtasks when this score exceeds 0.7 for three consecutive timesteps. The
boundaries are manually supervised, and the resulting score, threshold, and
physical actions have no reported coverage or action-effect guarantee. Its
branching behavior is learned implicitly from demonstrations.

We adopt its two most useful mechanisms: information actions must change the
next observation, and the next decision must branch on the observed result.
Our intended contribution is different:

| CoMe-VLA | This method |
|---|---|
| learned end-to-end active-perception VLA | wrapper around a frozen VLA |
| binary cognitive transition signal | typed prompt-conditioned hidden state |
| learned temporal memory | explicit public belief and searched-state update |
| learned action behavior | separately capability-gated VLA options |
| thresholded subtask switch | conformal set plus expected-risk minimization |
| implicit branches from demonstrations | explicit `FAILED/REVEALED/EMPTY` branches |
| decides whether information is sufficient | decides which action is worth its cost |

The comparison makes the novelty boundary precise. A binary “enough
information?” head alone is not our contribution. The claim is calibrated,
action-resolvable ambiguity and decision-theoretic physical information
acquisition without retraining the base VLA.

Its ablations also give us two concrete design constraints. First, observation
history and action history cannot be collapsed into a stateless image score.
Our v1 implements the smallest frozen-policy version of this idea: paired
before/after RGB features, the executed option, an explicit belief, and a
public searched-state history. Second, high-level progress recognition and
low-level motor control should not be forced into one newly trained decoder.
We therefore do not add a new VLA encoder or action decoder: the lightweight
belief/outcome heads decide, while the frozen pi0.5 action expert executes a
unit prompt. Longer multi-container experiments must ablate both visual-only
history and no-action-history variants.

There is also an embodiment boundary. CoMe-VLA predicts a controllable
head/chassis viewpoint inside its 29-D action, whereas the LIBERO pi0.5 action
space controls one arm and gripper; its external `agentview` camera cannot be
moved by the policy. Viewpoint discovery in this benchmark must therefore be a
calibrated wrist-motion option, or be evaluated with a second policy/environment
that exposes camera motion. Moving the wrist-camera extrinsics by hand is not a
valid substitute because it already caused a 5/5-to-0/5 stock capability
collapse.

## Relation to T-LEAF and belief-space task planning

[T-LEAF](https://arxiv.org/abs/2101.11981) embeds automata compiled from temporal
logic and predicted traces into a shared neural space, then uses their distance
as a differentiable training signal. It supports the use of temporal structure
when demonstrations are scarce, but it does not estimate proposition
uncertainty, predict physical information effects, or select actions by value of
information. We take the finite-trace automaton as the task skeleton. Our
calibrated belief supplies uncertain propositions, and the risk planner supplies
the uncertainty-to-action bridge. A learned T-LEAF regularizer is optional; it
is not needed to run the frozen-policy method.

[Online Replanning in Belief Space for Partially Observable Task and Motion
Problems](https://arxiv.org/abs/1911.04577) is the closest classical system
view: it explicitly cites opening drawers and moving objects to observe hidden
state, then replans in hybrid belief space. Our distinction is prompt-conditioned
open-vocabulary state, calibrated neural evidence, context-scoped VLA action
effects, and a product with an inspectable finite-trace progress state.

[HERACLEs](https://arxiv.org/abs/2309.10092) combines temporal-logic task plans,
LLM action generation, and conformal prediction. It supports conformal logic as
a formal interface, but its uncertainty concerns whether language-generated
actions accomplish declared symbolic subtasks. It does not model whether a
physical observation is sufficient or which embodied interaction reveals a
hidden proposition.

## Evidence frozen so far

| Evidence | Result | Scope |
|---|---:|---|
| Stock pi0.5 reproduction | 100/100 | checkpoint/input wiring |
| T01 drawer opening | 97/100; lower bound 0.924 | joint motion only |
| T01 target visible after opening | 57/60; lower bound 0.876 | policy-camera endpoint |
| T01 empty layer visually certified | 56/60; lower bound 0.854 | paired counterfactual proxy |
| Six-point v8 outcome head | FAILED 7/7; REVEALED 7/9; EMPTY 4/5 | development NOT-GO |
| Patch-resolvable v9 candidate | 20/21 | diagnosed seeds; not certification |
| Open layer at stock observation pose | 60/60; lower bound 0.951 | evaluator-only coverage diagnostic |
| Proprioceptive return from perturbed pose | 30/30; lower bound 0.905 | controller diagnostic, not full executor |
| Hidden butter belief | 97/100 | T01 seed-disjoint audit |
| Visible cream-cheese prompt swap | 100/100 | identical closed-scene RGB |
| Open-drawer butter state swap | 99/100 | same prompt, changed state |
| Target-observability risk route | 296/300 | offline ACT vs REMOVE |
| Fixed-rule baselines | 200/300 or 100/300 | paired T01 audit |
| Final retrieval after reveal | 0/5 | hard failure |
| ROTATE | 0/30 | disabled action |

These results establish a T01 decision prototype. They do not establish the
full four-state belief, a reliable visual outcome critic, final task
improvement, scene-disjoint generalization, or a second-policy result. In
particular, joint motion is not evidence that the robot obtained information.

Earlier paired-RGB outcome heads are NOT-GO. Global v1 obtains 60/60 coverage
but 21/60 singleton-correct; spatial v2 obtains 18/21 and 13/21; the
target-subtask v3 obtains 20/21 and 15/21. The six-point hierarchical v7 head
passed development but is invalid: it checked visibility only in the final
frame. Debug seed 770 visibly contained the target at intermediate points but
would have been labeled `EMPTY` after later self-occlusion. Seeds 700--799 are
therefore debug-only. Temporal-label v3 subsequently froze on seeds 600--659,
but v8 failed development with per-class coverage 7/7, 7/9, and 4/5. The
architecture-derived v9 relabeling reaches 20/21 on those diagnosed seeds;
untouched 660--699 must certify it before the seeds 900--999 audit can open.

## Required RSS experiment ladder

The authoritative registry is `benchmarks/rss_v1/gates.yaml`. The next gates
are deliberately ordered so that a later success cannot conceal an earlier
confound.

1. Run the clean v9 development extension on untouched seeds 660--699. Only
   after it passes may the one-time seeds 900--999 audit run. Physical action
   lower bounds must reach 0.80; conformal class coverage remains 0.90. Report
   the original 0.90 physical criterion as a stricter sensitivity check.
2. Run an oracle-free T01 closed loop on the product belief \((b_t,\mu_t)\),
   reporting logic violations and `SAFE_STOP` separately from `NOT_FOUND`.
3. Repair or replace the post-reveal `ACT` executor; target reveal alone is not
   final-task success.
4. Add `VIEWPOINT_BLOCKED` and `ABSENT`, then test physical `EMPTY` updates and
   calibrated `NOT_FOUND`.
5. Add independently capable viewpoint-discovery and information-enrichment
   actions; an action with a reliability lower bound below 0.80 stays disabled.
6. Freeze scene-, object-, clutter-, occlusion-, and container-disjoint paper
   test sets.
7. Compare monolithic VLA, fixed rule, action spread, a CoMe-style binary
   information-sufficiency head, uncalibrated typed belief, visual-only
   history, no-history, no-effect, no temporal state, exact probabilistic
   automaton, and the complete method. If sufficient trajectories exist, add a
   T-LEAF-style learned embedding as a data-efficiency ablation.
8. Reproduce the decision benefit with a second frozen policy.

`PARTIAL` always counts as `NOT-GO` for paper readiness.
The frozen comparison contract is in `benchmarks/rss_v1/ablations.yaml`.
