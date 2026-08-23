# PIU drawer-binding preregistration v1

Status: frozen before any v2 oracle policy rollout on 2026-08-22.

## Population and unit

The unit is an initial-state group in the unchanged T01D hidden-butter drawer
scenario. Marker styles, prompts, and action branches from the same state remain
paired and in one data split. The screen groups, independent pilot groups, and
future formal groups are disjoint.

## Intervention and comparator

The comparator is raw RGB after an actually executed OPEN. The diagnostic
intervention modifies only the policy RGB with an evaluator-generated target
marker and discloses this oracle use in every report. The policy, checkpoint,
prompt template, action budget, initial state, and evaluator are otherwise
fixed.

The learned follow-up, if activated, receives no marker or privileged field. It
uses frozen PaliGemma prefix tokens derived from public RGB and the prompt.

## Outcomes

The formal oracle primary outcome is the LIBERO target grasp-contact predicate.
The primary estimand is the paired target-contact risk difference between the
selected marker intervention and raw RGB. The test is the exact two-sided paired
binomial test at alpha 0.05. Pilot discordance rates determine the smallest
prospective group count with target power 0.80; the count is frozen before new
formal groups are collected.

Secondary outcomes are wrong-object grasp contact, continuous maximum target
lift, destination-final success, and final task success. Visibility is reported
as continuous raw-camera pixels. Mask non-emptiness is only an oracle-rendering
eligibility condition.

The learned binder additionally requires a spatial metric on held-out public-RGB
examples. No spatial result is inferred from grasp contact.

## Development procedure

Three marker styles are screened on seeds 1400, 1403, and 1406. Selection
lexicographically maximizes target contact and destination-final success,
minimizes wrong contact, then minimizes changed RGB pixels. An exact tie triggers
another development group; it is not broken by a preferred style order.

The chosen style is then run on independent pilot seeds 1401, 1402, 1404, 1405,
and 1407. There is no 4/5 or 1/5 automatic branch. The pilot plans, but does not
replace, a disjoint formal experiment.

## Branches fixed in advance

- Positive formal oracle ceiling: collect training-only target masks and train
  the public-RGB spatial binder. Select architecture on development groups and
  fit any decision threshold only on calibration groups.
- No practically meaningful oracle improvement: stop enlarging the router and
  qualify a target-conditioned PICK primitive before returning to the same
  belief/action loop.
- Learned spatial metric improves but target contact does not: classify as an
  executor-interface failure, not method success.
- Target contact improves but placement does not: retain the binding result and
  route the remaining failure to PLACE qualification.

## Deferred claims

Effect-aware routing, information-value optimization, conformal action sets,
multi-scenario generalization, and sealed full-loop success are not evaluated in
this sprint. They cannot appear as positive claims based on these data.

## Amendment 1: binder metric freeze

Frozen on 2026-08-22 before any new real binding label, full-prefix feature
cache, binder checkpoint, or calibration outcome existed. Simulator masks are
resized with nearest-neighbor semantics and converted to the exact fraction of
target pixels in every frozen image patch; no pixel-count threshold is used.

Development selects the declared lightweight architecture by spatial negative
log likelihood, then presence Brier score, then parameter count. Reported
threshold-free spatial metrics are target-distribution NLL, argmax target-patch
hit, and probability mass on all nonempty target patches. Task-sufficiency loss
and metrics are unsupported whenever evaluator annotations are null.

Calibration is split by initial-state group into a temperature role and a
conformal role. Primary miscoverage is alpha 0.10; alpha 0.05 and 0.20 are
reported without choosing among them from results. The spatial conformal event
is that the returned patch set intersects at least one true target patch in the
current/interaction-post observation, conditional on a localizable target.
Pre-interaction tokens remain public context but are never positive
localization targets. If the finite calibration set cannot
resolve an alpha, the set saturates conservatively instead of interpolating a
smaller quantile. The sealed split requires a hash-bound single-use
authorization manifest.

## Amendment 2: action-causal controller and sealed comparison freeze

Frozen on 2026-08-22 before any real action-effect cache, calibration artifact,
controller rollout, or sealed full-loop outcome existed. Every public
affordance is enumerated; the candidate generator cannot receive a hidden
correct-container annotation. Candidate-conditioned frozen prompt tokens,
the selected binder target token, and public action history feed a route head
and eight typed effect heads. Nonterminal factors require an actually executed
counterfactual branch. STOP and REPORT_NOT_FOUND are exact zero-action,
identical-observation null transitions. Unsupported factors are null/masked,
never imputed from capability or filled with an all-negative target.

Development retains `route_only`, `stop_gradient_effect`, and `joint_effect`.
Architecture and initialization are selected by route NLL, supported-factor
Brier score, then parameter count. Multi-task scales are learned on training
groups; no manual effect or information-value weight is accepted. All binding
ablations alter public tensors without inspecting a label.

Route temperature and factor temperatures use one calibration group set;
multiclass and class-conditional LAC order statistics use a second disjoint
group set. The primary risk is alpha 0.10, with 0.05 and 0.20 reported. An
unsupported factor cannot authorize execution. A physical action requires a
singleton route set, the appropriate singleton sufficiency set, and singleton
positive execution plus task-progress or task-relevant-change sets. DIRECT and
PICK additionally require singleton target presence and a nonempty calibrated
current-frame patch set. Its exact normalized enclosure, selected public
primitive, referent, and destination are serialized as deterministic text to
the frozen pi0.5 executor; neither score values nor evaluator labels are
serialized. STOP
requires singleton task completion. REPORT_NOT_FOUND requires both singleton
absence and exhaustive registered-search coverage. Neither is an alias for
ABSTAIN.

The fair comparison registry is `piu_baselines_v1.yaml`. Public B0--B5 and B8
share the initial state, frozen pi0.5 checkpoint, option budgets, evaluator, and
failure denominator. B6 uses evaluator effect labels and B7 uses evaluator
target masks; both are shown only as oracle upper-bound columns. The sealed
primary comparison is B8 versus B0 target grasp contact with a two-sided exact
paired binomial test at alpha 0.05. All declared secondary binary comparisons
form one Holm-corrected family. Continuous lift, interaction count, and executed
steps are descriptive; no empirical pass threshold is introduced.

Physical dispatch is a separate gate from calibrated selection. Every exact
candidate/primitive must carry an immutable `FORMALLY_QUALIFIED` certificate
from a prospectively frozen external minimum-reliability contract, complete new
initial-state-group denominator, and exact one-sided binomial test. Historical
counts and action budgets cannot issue that certificate. Without it, the
dispatcher may emit a dry-run plan but must not contact the external executor.
Every external execution additionally validates the exact content-addressed
`pi05_libero` checkpoint tree before its first action. Unverified metadata
recorded only after a rollout cannot establish a fair frozen-policy comparison.

## Amendment 3: prospective counterfactual execution eligibility

Frozen on 2026-08-23 before any successor counterfactual branch, action-effect
label, or real binder checkpoint existed. The registered candidate superset is
never reduced by an evaluator label or predicted effect. Before a physical
training-data fork, an immutable execution plan applies only calibrated public
binder sets: information primitives require singleton insufficient evidence;
DIRECT/PICK require singleton sufficient evidence, singleton non-holding,
singleton target presence, and a nonempty current-frame spatial set; PLACE
requires singleton sufficient evidence and singleton holding. STOP and
REPORT_NOT_FOUND remain eligible exact null branches.

These are the controller's typed primitive prerequisites, not fitted numeric
cutoffs. The plan is hash-bound to the public transition, binder prediction,
calibration artifact, and prefix layout. It cannot read an effect prediction,
candidate outcome, evaluator mask, or correct-route label. A context-ineligible
candidate remains in the exact candidate matrix and may retain the independent
evaluator's correct-route label; it is not executed, its pre/post causal digest
is the same decision-state digest, and every unobserved effect factor is null
and loss-masked. Therefore conservative binder ambiguity changes execution
coverage, not route ground truth, and cannot be converted into fabricated
negative effects.

## Amendment 4: main paired pilot, power, and execution order

Frozen on 2026-08-23 before any real B8 development episode, B8-versus-B0
sample-size plan, formal group allocation, or sealed full-loop outcome existed.
The main pilot consists only of paired public-method B8 and B0 episodes from
the development split. Every pair must share the exact opaque source-state
hash and simulator seed, and all episodes must share the registered frozen
pi0.5 policy identity. Failed, timed-out, and abstaining episodes remain binary
observations in the denominator with their recorded evaluator outcome; they are
never dropped. Oracle B6/B7 episodes cannot enter this pilot.

The pilot reports the B8-minus-B0 target-contact risk difference, the four
paired cells, exact paired-binomial p-value, sample variance of the paired
difference, marginal Wilson intervals, and a conservative 95% risk-difference
interval formed from Bonferroni simultaneous Clopper-Pearson intervals for the
two discordant cell probabilities. Pilot significance never opens the sealed
test and does not select an endpoint.

Prospective power uses the primary two-sided exact paired-binomial test at
alpha 0.05 and target power 0.80. Its operating point uses a joint 95% lower
confidence construction: exact one-sided Clopper-Pearson lower bounds are
formed for the discordant-pair probability and for the probability that B8
wins conditional on discordance, with Bonferroni allocation. If the observed
paired effect is nonpositive, the directional lower bound is at or below 0.5,
or no design reaches power within the declared 200-group numerical search
bound, no formal group count is frozen. This retained blocked result cannot be
replaced with a round-number or post hoc sample size.

Pilot groups are excluded from every later split. After a valid count exists,
the sealed split must contain exactly that many new initial-state groups. A
pre-outcome state manifest binds every group and simulator seed to one unique,
finite numeric opaque NPZ state and its exact state key/hash. The simulator may
load this transport state, but no policy method may read the privileged vector.
The SHA-256-keyed, outcome-independent permutation freezes group order and the
within-group B0--B8 order from the hashes of the plan, split, exact state
manifest, scenario config, analysis config, baseline registry, and offline
release lock. The schedule retains every simulator seed and the frozen policy
identity. The shared limit of eight controller decisions is a pre-outcome
resource cap, not an action threshold or success rule. Formal matrix assembly
requires the schedule hash and rejects missing, extra, differently seeded, or
policy-drifted rows.

Every sealed cell is launched through a single-use ordered ticket. The ticket
binds its schedule index, method, source-state hash, seed, immutable output
directory, and expected episode path. The next ticket cannot be issued until
the preceding episode has an immutable close receipt, and every runtime B0--B8
entry point rejects sealed execution without its exact ticket. Episodes and
formal rows retain this chain, preventing post-outcome reruns or favorable
selection among attempts.

The sealed interpretation is not reduced to `p < 0.05`. The Holm family also
contains B8 full-task and wrong-contact comparisons against B1 and B3, plus the
B4-versus-B3 effect ablation. Interaction counts and executed steps are paired
descriptively against B0, B1, and B3. B8's fraction of the B0-to-B7
target-contact gap is descriptive because B7 is privileged. A claim that
unnecessary interaction did not increase is blocked until an external task-cost
contract declares a noninferiority margin; no margin is inferred from results.
Calibration efficiency requires separate sealed label-sidecar evaluation. Any
"nonsaturated" effect subgroup must be declared on development data before the
sealed split and cannot be selected after outcomes.
