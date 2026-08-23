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
is that the returned patch set intersects at least one true target patch,
conditional on a localizable target. If the finite calibration set cannot
resolve an alpha, the set saturates conservatively instead of interpolating a
smaller quantile. The sealed split requires a hash-bound single-use
authorization manifest.
