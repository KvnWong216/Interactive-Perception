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
| conformal prediction threshold | fitted calibration statistic | allowed only when fit on an isolated calibration split |
| conformal alpha | declared risk contract | freeze before calibration and report risk/coverage across preregistered levels |
| formal `alpha=0.05`, target power `0.80` | preregistered inferential design | conventional Type-I/II error controls; report achieved power and effect interval, never use as an online threshold |
| paired-power search limit `200` | numerical resource bound | failure to find a design blocks collection; it is not evidence for or against the method |
| effect-selector singleton rules | semantic design in rejected pilot | do not promote; successor method requires a new preregistered justification |
| CPU fusion `0.4/0.4/0.2` | manual diagnostic baseline | baseline-only |
| `src/interaction_uncertainty` weights and utility thresholds | legacy heuristics | frozen Heuristic V0 baseline only |
| decoder width/heads/optimizer/gradient bounds | development hyperparameters | rejected pilot only; successor uses a declared development search and ablation |
| four-token PaliGemma prefix summary | unsupported fixed pooling | rejected pilot only; sharing encoder weights is not sharing the action expert's full-prefix interface |
| candidate `mean(prompt-last,prompt-mean)` | unsupported fixed pooling | rejected pilot only; successor retains candidate token masks and learns readout on training groups |

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
