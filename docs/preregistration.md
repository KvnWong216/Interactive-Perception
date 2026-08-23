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

## Amendment 5: oracle-gate execution order and runtime closure

Frozen on 2026-08-23 before any external v2 oracle screen or confirmation
outcome existed. The original v2 experiment file and its retained policy-free
preflight remain byte-identical. A separate execution-order protocol now
requires a SHA-256-keyed permutation of the nine style-by-group screen cells;
after a unique screen result exists, it separately freezes the five confirmation
groups. Each schedule binds the source-state hashes, expected immutable report
paths, exact policy identity, and the complete offline release lock. The runner
requires both this schedule and a matching external endpoint-check artifact.
That endpoint artifact must include one finite action sample from the retained
public policy frame; metadata-only reachability is insufficient.
Resume may skip only a pre-existing report that passes the full oracle report
validator; partial or invalid output cannot be overwritten. Newly started
identified servers expose a random session ID. The endpoint check and every v2
report in one phase must retain that same ID, so a server restart cannot reset
policy sampling state unnoticed during a style comparison.

The offline release audit now checks local Python imports and literal runtime
file dependencies transitively. The unified executor, external policy client,
oracle rendering transform, scenario YAML/BDDL, checkpoint identity, and GPU
safety/serving scripts are inside the content inventory. The release lock
itself is the unavoidable self-hash exception; B2's current-tree legacy
`infer.py` is excluded because the adapter executes only the separately
attested detached tag worktree.

## Amendment 6: robustness and automatic paper-table freeze

Frozen on 2026-08-23 before any real successor training report, calibration
artifact, or sealed outcome existed. The executable binder input ablations are
reported only from development groups and retain the full, no-prompt,
prompt-swap, last-frame, global-mean, no-action-history, per-camera,
spatial-shuffle, and temporal-shuffle variants in their declared order. The
effect comparison retains route-only, stop-gradient-effect, and joint-effect.
These rows may diagnose component necessity but cannot be relabeled as sealed
method evidence.

Prompt robustness remains inside the unchanged hidden-butter drawer task. Four
semantics-preserving prompt strings are fixed in the reporting protocol and
crossed within each newly collected initial-state group, so the state is the
cluster and prompt wording cannot select the model. This is a paired lexical
stress test, not unseen-object, unseen-layout, unseen-scene, or broad OOD
generalization.

Paper tables are generated only from schema-checked, content-hashed artifacts.
A missing artifact or metric is rendered as `PENDING`, never imputed as zero.
The retained negative fixed-scenario matrix remains visibly separate from
development ablations and the one-shot sealed B0--B8 table; B6/B7 remain oracle
upper bounds. The formal report now emits descriptive rates and Wilson
intervals for every prespecified binary outcome and every method so no table
cell needs manual transcription. Even a complete table sets automatic method
success to null: the preregistered multi-metric claim audit, rather than an
invented score cutoff, determines permissible wording.

## Amendment 7: externally allocated executor risk and single-use qualification

Frozen on 2026-08-23 before any external executor-risk budget, successor
primitive qualification state, controller report, execution receipt, or
outcome existed. The old 8/10 and 7/10 engineering gates are historical only.
Binder conformal miscoverage is not executor failure risk, and the retained ten
rollouts cannot set either a primitive null or the power-design alternative.

A task owner must copy the null-valued external budget template to an immutable
artifact and declare (i) the maximum acceptable episode probability `delta` of
at least one dispatched primitive failure and (ii) a per-dispatch design
alternative strictly above the derived null. With at most `M=8` physical
dispatches, the registered dependence-free equal allocation sets the maximum
per-dispatch failure probability to `delta/M` and the minimum reliable rate to
`1-delta/M`. The alternative is used for prospective power only; it is not an
estimated reliability or a task-success claim. Both values, authority, and
rationale must be frozen before qualification outcomes. No repository default
numeric budget exists.

The retrospective v2 registry is diagnostic and excludes its seeds from the
new cohort. It correctly records that the cream-cheese and post-OPEN reports
executed compound DIRECT instructions: their contact and destination endpoints
are not relabeled as separately evaluated PICK or PLACE primitives and do not
set the formal effect size. Each formal plan is sized by the external contract
with an exact one-sided binomial rejection region at alpha 0.05 and target
power 0.80.

Every qualification group has a unique finite opaque NPZ state, unique seed,
public-only controller report, and exact serializer output. A hash permutation
freezes execution order and one receipt path per group while binding the split,
scenario, policy identity, and complete offline release. The selected candidate
payload and whether the deterministic bridge uses calibrated current-frame
boxes must be identical across the cohort; boxes and exact subtasks remain
state-specific and hash-bound.

The endpoint identity is checked before a single-use `STARTED` ticket exists.
After that ticket is written, a runtime or interrupted attempt is counted as a
failure and cannot be rerun. The certificate evaluator accepts no hand-written
Boolean outcomes: it loads every scheduled receipt and re-derives success from
the registered LIBERO contact predicate, task-declared terminal `in` relation,
terminal task goal, or the versioned `WoodenCabinet.is_open` range. A complete
certificate binds this evidence chain and authorizes only the exact candidate
payload and serializer mode it tested. Missing external artifacts remain
`PENDING` and authorize no physical paper-method action.

## Amendment 8: public claim semantics and endpoint identity

Frozen on 2026-08-23 before any successor empirical artifact existed. Public
prose must distinguish an executed compound DIRECT instruction from separately
qualified PICK or PLACE primitives. Retrospective grasp-contact and destination
endpoints may diagnose the compound trajectory, but cannot populate a
prospective primitive registry or authorize dispatch. Likewise, the retained
nonempty evaluator mask is evidence that complete occlusion ended; neither it
nor the historical 256-pixel marker is a calibrated human/policy recognition
threshold.

The paper, README, status, and results surfaces are content-hashed and checked
against required evidence-boundary statements plus an explicit list of retired
interpretations. Missing successor evidence remains `PENDING`, never zero. A
passing prose audit establishes only semantic consistency of the repository; it
is not physical, learned-model, calibration, or task-success evidence.

The same audit requires field-level classification of every numeric value in
the eleven active training, calibration, baseline, formal-analysis,
qualification, and sprint protocols. Values are read from the tracked configs
into the report rather than manually transcribed. Training-development search,
isolated-calibration risk, simulator/policy contracts, external risk,
statistical design, and safety/resource budgets remain distinct; a new
unclassified numeric path blocks the release.

## Amendment 9: external qualification resources and planner single source

Frozen on 2026-08-23 after the CLI-default audit and before any external risk
budget, primitive qualification outcome, or oracle confirmation outcome
existed. Amendment 7 specified executor risk and a power alternative but left
the primitive planner's numerical search/collection bound at an unexplained
CLI default of 1000 groups. That default is removed.

The external task-owner contract now additionally requires a positive-integer
maximum qualification-group count per primitive. It is a collection-resource
cap, not a null, alternative, action threshold, empirical result, or success
criterion. If the first exact one-sided binomial design attaining frozen power
does not fit, the planner retains
`NO_PLAN_WITHIN_EXTERNAL_COLLECTION_RESOURCE_CAP`; it may not change alpha,
power, risk, the alternative, or the resource value after outcomes. The budget,
derived risk contract, plan, qualification schedule, and certificate loaders
bind the same value. The repository template contains null, not a suggested
number.

The oracle paired-power planner's duplicated CLI default of 200 is also
removed. That planner now reads and hashes the sole numerical resource bound in
`piu_formal_analysis_v1.yaml`, already classified as a resource-only control.
This amendment changes no retained outcome or empirical estimate.

## Amendment 10: model-free OPEN qualification stimulus

Frozen on 2026-08-23 before any external risk budget, new qualification state,
qualification outcome, or successor training data existed. The previous
schedule contract required a controller-shaped decision report for every
primitive. For OPEN this creates a dependency cycle: new interaction data needs
a qualified OPEN executor, while a learned controller needs those new data.

OPEN executor qualification may therefore use a preregistered stimulus report
generated from two hash-bound public artifacts: the exact candidate ID and
primitive in the frozen qualification plan, and the complete candidate payload
in that group's public candidate set. The serializer remains unchanged. The
report loads no trained model, calibration artifact, evaluator field, oracle
input, or qualification outcome and declares that candidate choice is
outcome-independent. It estimates executor reliability conditional on the
predeclared OPEN stimulus only; it cannot enter controller-selection metrics or
establish the PIU method claim.

This exception is OPEN-only. A model-free probe for PICK or DIRECT is rejected,
because those actions require the learned, calibrated current-frame spatial
reference whose contribution the paper tests. Later PICK qualification must use
a real public-input controller report after binder training. Thus the cycle is
removed without inserting a hand box, oracle target, or synthetic success.

## Amendment 11: prospective formal oracle execution chain

Frozen on 2026-08-23 before an external endpoint, independent oracle pilot,
formal oracle plan, external executor-risk declaration, qualification outcome,
or successor data existed. The v2 protocol specified a disjoint formal
target-prompt test but previously implemented only its pilot-based sample-size
planner. The missing runtime is now fixed prospectively.

If and only if the planner returns `PROSPECTIVE_GROUP_COUNT_FROZEN`, the split
must contain exactly that many new `oracle_formal` groups, disjoint from every
oracle preflight/screen/confirmation seed and every OPEN-qualification
group/seed. Exact finite pre-OPEN NPZ states, code/config
release lock, pi0.5 identity, selected development style, and a
`FORMALLY_QUALIFIED` certificate for candidate `open_middle_drawer` are bound
before formal outcomes. A blocked power plan cannot be replaced by a chosen
round number.

For each group the executor attempts the certified OPEN stimulus once. The
estimand is intention-to-treat and never conditions on OPEN success. When a
post-OPEN state exists, oracle-target-prompt and raw post-OPEN DIRECT start from
that identical state; their order is a SHA-256-keyed permutation. The source
and both arms within a group use one endpoint session; an identity-matched
restart is allowed only between groups. A start ticket is irreversible: source/arm runtime
failure is false, and a process interruption is closed conservatively with both
arms false rather than rerun. Complete receipts retain report, image, action,
state, certificate, schedule, and session hashes. The analyzer recomputes
grasp-contact outcomes from the registered v2 evaluator reports and applies the
already frozen exact two-sided paired test. A positive result is privileged
causal-mechanism evidence only, never evidence for the public-input method.

## Amendment 12: purpose-specific cohorts and empirical execution DAG

Frozen on 2026-08-23 before any external risk contract, successor split,
formal oracle outcome, successor training datum, calibration prediction, or
sealed result existed. A single appendable split file is prohibited. OPEN,
PICK, and PLACE qualification schedules each bind their own prospective
qualification cohort; the oracle mechanism test binds a separate
`oracle_formal` cohort after its pilot-sized count is frozen; learned-model
roles bind a separate training/development/calibration cohort only after the
causal gate; and the main B8-vs-B0 test binds a new `sealed_test` cohort only
after its independent development pilot fixes the count. A purpose-specific
manifest declares the roles it must completely allocate. Earlier manifests
are immutable and cannot be extended to satisfy a later sample-size result.

Binder calibration roles and action-effect calibration roles are disjoint.
The latter are named `effect_calibration_temperature` and
`effect_calibration_conformal`. Counterfactual execution rejects the binder
calibration groups, because reusing them would let binder-fit labels influence
which action-effect outcomes are physically observed. Every learned-data role
is assembled by an immutable exact join: public and binding sample IDs must
match; counterfactual labels must equal the full public candidate matrix; and
each decision must have exactly one evaluator-selected route. Manual JSONL
concatenation is not a mainline artifact.

The frozen empirical DAG is the sole readiness authority. A file is complete
only after schema, recursive content references, required flags, and complete
prospective role membership validate. Missing, malformed, partially present,
and predecessor-blocked states remain distinct. A valid negative oracle result
or `NOT_QUALIFIED` primitive certificate is terminal scientific evidence: it
blocks successors without being relabeled as a software error or ignored.

## Amendment 13: binder-grounded task-primitive qualification stimulus

Frozen on 2026-08-23 before successor binder data, a trained binder, binder
calibration, PICK/PLACE qualification inputs or outcomes, counterfactual effect
data, or an effect model existed. Task-primitive qualification must occur after
binder calibration but before effect-data collection. Requiring the complete
effect-aware controller at that point would create a cycle: effect data require
qualified physical candidates, while their certificates would require the
effect model trained from those data.

For PICK, PLACE, or DIRECT only, the qualification planner may therefore bind a
candidate fixed by the prospective qualification plan to a public,
binder-calibrated eligibility artifact from a reserved
`primitive_qualification` group. This is not a model-free exception. The frozen
binder and its isolated calibration must produce the typed sufficiency,
presence, holding, and current-frame spatial sets. PICK/DIRECT require a
nonempty calibrated current-frame box; PLACE requires the calibrated holding
condition and carries no box. The effect model, route score, evaluator labels,
oracle inputs, and qualification outcomes are forbidden.

At schedule load, all prediction, feature, calibration, public-transition,
split, and plan hashes are verified. The calibration sets, candidate
eligibility, spatial enclosure, and deterministic pi0.5 subtask are recomputed;
a hand-edited box or eligibility Boolean is rejected. The artifact estimates
executor reliability conditional on this exact preregistered binder-grounded
stimulus only. It cannot be reported as route selection, learned effect-model
performance, or end-to-end task success, and it cannot be collected into the
action-effect dataset.

## Amendment 14: semantic external and sealed artifact validation

Frozen on 2026-08-23 before any external task-owner budget, identified pi0.5
endpoint, prompted-VLM identity, B2 development arm, or sealed outcome existed.
The task-owner risk declaration and the identified pi0.5 endpoint are two
independent root gates. Neither may conceal the absence of the other. The risk
artifact is re-derived against the frozen episode dispatch cap and must contain
a nonempty external authority/rationale, a feasible per-dispatch design
alternative, and a positive collection cap. The endpoint artifact must bind the
exact checkpoint-identity file, exact server metadata, concrete host/port, a
hash-bound retained source report, and a finite typed action sample.

B1 identity requires an explicit model ID, immutable revision, and declared
public routing capability. Its probe retains the exact bounded response and
hash-bound public transition/identity; unknown candidates remain abstentions.
B2 development evidence is assembled only from the complete prospectively
assigned development arm, with one unique state and registered policy identity
per group. Manual selection or JSONL concatenation is not admissible.

For sealed evidence, schema-valid rows alone are insufficient. The DAG reloads
the frozen formal schedule and requires exact coverage of every scheduled
group-method pair, seed, source-state hash, and policy identity. The B6 and B7
oracle files must separately cover their complete scheduled columns. Thus an
incomplete but well-formed result file cannot unlock paper generation. Both
release SVGs must additionally embed the exact regenerated evidence-table hash;
an old figure cannot accompany a new result table.

## Amendment 15: predecessor results are recomputed, not trusted

Frozen on 2026-08-23 before any new qualification, oracle, learned-data, or
sealed artifact existed. Recursive file hashes establish provenance but do not
establish that a summary was calculated correctly. Before advancing S02--S05,
the DAG now invokes the registered primitive and oracle loaders. Risk contracts
are re-derived; powered plans are recalculated; schedules and every certificate
receipt are replay-validated; oracle screen and confirmation summaries are
recomputed from raw reports; and the formal oracle result is recomputed from
the complete intention-to-treat schedule.

A powered primitive design that cannot fit the externally declared collection
cap is retained as the legitimate terminal state
`NO_PLAN_WITHIN_EXTERNAL_COLLECTION_RESOURCE_CAP`. It is neither silently
replaced by a smaller test nor mislabeled as file corruption.

Before advancing S06 or S10, every role dataset is also recomputed from its
immutable source union. Public and binding rows must have exactly the same
sample IDs and group/split assignment. Action-effect rows must equal the full
public candidate matrix with exactly one selected route per sample. Output row
order, counts, references, and claim firewalls must equal the assembler's exact
result. This is integrity/protocol evidence only and creates no learned or
physical performance claim.

## Amendment 16: learned artifacts require checkpoint replay

Frozen on 2026-08-23 before any real S07--S12 learned artifact existed. A
training or prediction JSON file and a matching SHA-256 do not establish that
the scores were produced by the declared model. The DAG therefore reloads each
small binder/effect checkpoint with CPU-only placement, validates finite tensor
state and the predeclared parameter cap, reconstructs the exact hash-bound
feature/label join, replays inference, and compares every stored score and
target array. The complete hyperparameter grid, epoch history, lexicographic
development selection, group firewall, and selected proper scores are
recomputed. Calibration artifacts are refitted from the separate temperature
and conformal roles through the same shared implementation used by the writer.

Every binder development ablation now retains its own checkpoint and raw
development predictions. An ablation metric without those references is not
admissible. These checks reject a co-edited prediction file and report hash as
well as a hand-edited fitted temperature. They are reproducibility and leakage
evidence only: replaying synthetic or development artifacts cannot create a
sealed-test or physical-performance claim.
