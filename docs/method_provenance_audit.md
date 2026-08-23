# Method and threshold provenance audit

This audit separates runnable engineering from evidence that may support a
public-input research method. The machine-readable source of truth is
`configs/experiments/method_provenance_v1.yaml`; CI rejects an oracle,
unsupported heuristic, or baseline-only rule promoted into a main-method
decision.

## Audit outcome

| item | classification | disposition |
|---|---|---|
| drawer-open boundary `-0.14 m` | simulator contract | keep as evaluator metric; LIBERO declares open range `[-0.16,-0.14]` |
| historical target visibility `256 pixels` | unsupported recognition heuristic | deprecate as a binary claim; retain historical artifact and report continuous pixels |
| historical pick lift `3 cm` | unsupported evaluator heuristic | remove from v2 reports; use LIBERO grasp contact and continuous lift |
| oracle box padding/width, point radius, spotlight strength | hand-designed oracle intervention | diagnostic-only; changed RGB pixel counts retained; forbidden as method |
| former oracle confirmation `4/5` and wrong contact `1/5` | unsupported small-sample branch rule | deprecate v1 gate; five groups are an independent pilot only |
| old executor gates `8/10`, `7/10`, `8/10` | engineering qualification | retain as history; never treat as a statistical paper claim |
| DIRECT/OPEN step counts | safety/resource budgets | report saturation and completion source; never equate budget exhaustion with success |
| pi0.5 replan every five steps | frozen-policy interface contract | keep fixed and hash policy/checkpoint identity |
| external pi0.5 identity | frozen-policy interface contract | exact checkpoint-tree match is mandatory before every prospective action; registered capabilities may be declared but unknown metadata fields fail |
| conformal prediction threshold | fitted calibration statistic | allowed only when fit on an isolated calibration split |
| conformal alpha | declared risk contract | freeze before calibration and report risk/coverage across preregistered levels |
| formal `alpha=0.05`, target power `0.80` | preregistered inferential design | conventional Type-I/II error controls; report achieved power and effect interval, never use as an online threshold |
| formal pilot report/joint-design confidence `0.95/0.95` | preregistered inferential design | exact intervals plus Bonferroni joint lower bounds replace a hand-selected effect shrinkage; a lower directional bound at chance freezes no N |
| paired-power search limit `200` | numerical resource bound | failure to find a design blocks collection; it is not evidence for or against the method |
| formal B0--B8 order | hash-keyed protocol permutation | plan/split/config/registry hashes determine order before outcomes; order is retained and cannot be optimized after results |
| formal paired source state | pre-outcome content-addressed NPZ cohort | one numeric finite opaque state per sealed group is bound to its simulator seed; it is transport-only and never a policy feature |
| formal code identity | offline release inventory SHA-256 | pilot planning and schedule freezing fail if any required mainline file differs; schedule binds the verified lock |
| effect-selector singleton rules | semantic design in rejected pilot | do not promote; successor method requires a new preregistered justification |
| CPU fusion `0.4/0.4/0.2` | manual diagnostic baseline | baseline-only |
| `src/interaction_uncertainty` weights and utility thresholds | legacy heuristics | frozen Heuristic V0 baseline only |
| decoder width/heads/optimizer/gradient bounds | development hyperparameters | rejected pilot only; successor uses a declared development search and ablation |
| four-token PaliGemma prefix summary | unsupported fixed pooling | rejected pilot only; sharing encoder weights is not sharing the action expert's full-prefix interface |
| candidate `mean(prompt-last,prompt-mean)` | unsupported fixed pooling | rejected pilot only; successor retains candidate token masks and learns readout on training groups |
| full candidate-prefix serialization | protocol identifier | retain candidate IDs, primitives, time/token masks, and all valid frozen states; no fixed numerical pooling |
| executed-action embedding | public protocol identifier | read the exact dispatched candidate from public history; evaluator action labels are join checks only |
| candidate enumeration over public affordances | protocol identifier | enumerate all registered sources; never use hidden correct-container labels |
| binding/effect multi-task scales | training-learned | homoscedastic log variances only; unsupported tasks remain masked |
| predicted-effect-to-route bridge | training-learned | route logits consume predicted factor probabilities through trained weights; no manual utility |
| route/effect temperature and conformal boundaries | isolated calibration | temperature and conformal roles are group-disjoint; no manual confidence threshold |
| singleton controller logic | preregistered protocol semantics | typed execution authorization; STOP, NOT_FOUND, and ABSTAIN remain distinct |
| current spatial patch-to-text bridge | protocol identifier | exact enclosure of every calibrated latest-frame patch; no probability cutoff or oracle geometry |
| B3/B4 unique argmax | baseline protocol | exact ties abstain; B3's factor channel is exactly zero and B4's is learned |
| B5 spatial-bridge disable | baseline protocol | same calibrated scores/sets, but no patch geometry reaches pi0.5 |
| primitive execution certificate | external risk contract plus formal exact test | complete prospective denominator required; historical count gates and budgets never authorize live execution |
| Fréchet joint lower bound | formal diagnostic | reports a dependence-robust lower bound; never selects an action |
| B0--B8 comparison registry | protocol identifier | same paired states, frozen policy, budgets, evaluator, and failure denominator; oracle is a separate column |
| exact paired primary and Holm secondary family | formal statistical procedure | frozen before sealed outcomes; no automatic method pass threshold |

## Threshold-invariance evidence

`results/diagnostics/original_drawer_threshold_audit_v1.json` re-derives the
following directly from hashed retained reports:

- post-OPEN visibility remains exactly 8/10 for every integer threshold from
  1 through 447 raw camera pixels;
- post-OPEN butter has zero LIBERO grasp contacts in 10 trials, independent of
  any lift cutoff;
- visible cream cheese has grasp contact in all 10 trials and the smallest
  per-run maximum lift is 0.150343 m, far from the old 0.03 m cutoff;
- drawer OPEN succeeds in 9/10 under LIBERO's declared `-0.14 m` boundary.

The old results therefore remain numerically intact, but the unsupported
binary definitions are not inherited by new experiments.

The v3 policy-free oracle preflight originally recorded the hash of the same
YAML payload with one extra trailing newline, while Git retained the canonical
single-newline blob. Its experiment hash and the dependent threshold-audit
source hash were repaired to the tracked bytes; no preflight row, eligibility
decision, policy call, or metric changed.

## Rules for the successor method

A number may affect a public-input online action only if it is learned on the
training split, fitted on the isolated calibration split, imposed by a
physics/frozen-policy contract, or declared as a user risk contract. Evaluator
outcomes must come from simulator/physics predicates or a preregistered formal
statistical procedure. Safety budgets may stop execution but cannot certify
success. Oracle information and manual visual markers remain in a separately
labeled diagnostic column.

Architecture and optimization constants are chosen on development groups
only. The search space, selection objective, random seeds, and all tried
configurations must be retained. No threshold may be changed after calibration
or after a sealed test is opened.

The frozen pi0.5 encoder may still be reused, but the successor representation
must preserve valid prompt/image token masks, camera spans, and spatial token
indices. Global pooling is retained as an explicit ablation, never as an
unexamined main-method default.

Pre-interaction target pixels are not localization supervision. They remain
public contextual tokens, while only current/interaction-post patches define
the spatial target and conformal coverage event. This prevents a previously
visible target from certifying that the post-action referent is localized.
