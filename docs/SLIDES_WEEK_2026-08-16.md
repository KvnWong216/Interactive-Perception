# Weekly slide kit — 2026-08-16

## One-sentence story

The frozen VLA can perform an information action, but it does not know when the
action is worth taking or whether the action actually resolved the
prompt-relevant ambiguity; our wrapper makes those two decisions explicit and
calibrated.

## Slide 1 — Problem

**Title:** The skill exists; the decision is missing

- π0.5 can open the stock middle drawer.
- A monolithic final prompt does not reliably invoke or finish the hidden-object task.
- Action-sample spread does not identify *why* information is missing.
- Question: which information action reduces task risk for this prompt?

Suggested visual: `method_pipeline.svg`, cropped to the observation and action
ends.

## Slide 2 — Core method

**Title:** Turn uncertainty into an action value

Use the full `method_pipeline.svg`.

Speaker line:

> We do not add another VLA encoder or decoder. A typed belief and calibrated
> action-effect model sit around frozen π0.5, and a finite-horizon risk planner
> evaluates whether an information action is worth its cost.

Key equation:

\[
U_{\mathrm{res}}(a\mid b)=V_{\mathrm{stop}}(b)-Q(b,a).
\]

Positive means the action is expected to remove more task risk than it creates.

## Slide 3 — Latest breakthrough

**Title:** Opening the drawer is not the endpoint

Use `evidence_snapshot.svg`.

- Joint opening: 97/100, lower bound 0.924 — motor capability passes.
- Target visible: 57/60, lower bound 0.876 — information effect fails.
- Empty layer certified: 56/60, lower bound 0.854 — information effect fails.
- Best paired-RGB head: 20/21 coverage, 15/21 singleton-correct — closed-loop
  outcome recognition fails.

This is the cleanest new scientific finding.

Follow-up diagnosis:

- all three strict reveal failures fully opened the layer but ended with both
  policy cameras at zero target pixels;
- the unchanged stock observation pose sees the open-layer target in 60/60;
- the proprioceptive return controller recovers that pose and visibility in
  30/30 perturbation diagnostics, lower bound 0.905;
- the continuous pi0.5 `OPEN_AND_OBSERVE` experiment is still unrun.

## Slide 4 — Visual evidence

Use `t01_revealed_vs_empty_seed458.png`.

The wrist view clearly separates a revealed butter package from an empty
drawer, while the evaluator-only global view mainly shows arm pose. This
motivates prompt-attended temporal wrist features rather than global feature
averaging.

Then show `demos/t01_return_to_observe_diagnostic_seed600.mp4`. It places the
wrist view on the left and evaluator-only global view on the right. Label it
clearly as a controller diagnostic: its drawer and start pose are evaluator-
constructed, so it demonstrates the observation-recovery mechanism rather
than full executor reliability.

## Slide 5 — Relation to CoMe-VLA

**Title:** Same act–sense–branch loop, different claim

| CoMe-VLA | Our method |
|---|---|
| end-to-end active-perception VLA | wrapper around frozen VLA |
| binary “information sufficient” head | typed target-state belief |
| fixed 0.7 / three-frame switch | class-conditional conformal set |
| implicit branches from demonstrations | explicit `FAILED/REVEALED/EMPTY` branches |
| learned action behavior | separately calibrated option reliability and cost |
| no explicit uncertainty distribution | explicit action-resolvable risk reduction |

Borrow next: current + five historical frames, visual/proprioceptive memory,
and prompt-to-token attention. Do not borrow: manual binary completion labels as
the uncertainty definition.

## Slide 6 — Honest status and next gate

| Milestone | Status |
|---|---:|
| T01 prompt/state decision prototype | GO |
| Physical information-effect reliability | NOT-GO |
| Oracle-free outcome closed loop | NOT-GO |
| Final retrieval | NOT-GO, 0/5 |
| RSS method paper | NOT-GO |

Next experiment:

1. Run the implemented `OPEN_AND_OBSERVE` smoke, then collect its immutable
   600--659 development split and require `REVEALED` / `EMPTY` lower bounds of
   at least 0.90.
2. Collect a calibration-only temporal transition set with six policy-camera
   frames and robot proprioception across the information action.
3. Fit a prompt-attended temporal outcome head; freeze it.
4. Confirm on new post-freeze development seeds.
5. Only if it passes, open its 700--799 audit once. Keep legacy 500--599 sealed.

## Demo index

- `demos/routed_reveal_success_seed091.mp4`: calibrated route and drawer reveal.
- `demos/monolithic_failure_seed001.mp4`: final prompt does not solve the hidden task.
- `demos/two_stage_retrieval_failure_seed001.mp4`: reveal does not imply final retrieval.
- `demos/t01_return_to_observe_diagnostic_seed600.mp4`: retreat restores the
  stock wrist/global observation, diagnostic-only.

All paths above are relative to `assets/slides/`.

## Claims to avoid

- Do not call 97/100 “target reveal”; it is joint-based drawer opening.
- Do not claim the paired-RGB critic passed.
- Do not claim final-task improvement.
- Do not describe action-resolvability as a third stochastic uncertainty source;
  it is a decision property of a belief and an action model.
- Do not use BEV or segmentation as controller input.
