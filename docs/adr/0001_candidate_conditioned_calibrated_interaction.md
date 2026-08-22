# ADR-0001: candidate-conditioned calibrated interaction

- Status: accepted for Stage 0/1 reference implementation
- Date: 2026-08-22
- Scope: one-GPU method design and no-training original-drawer wiring trace

## Context

The previous pipeline combines Grounding DINO, SAM, DINOv2, SigLIP, multiple
VLM-reported uncertainties, hand-set linear weights, a deterministic expected
utility selector, and π0.5. It also exposes `NEXT_BEST_VIEW`, which violates the
research boundary. Existing evidence reports final butter retrieval at 0/5, so
component success cannot justify continuing to patch that architecture.

The legacy implementation is frozen at `baseline/heuristic-v0`. The new method
must fit on one 48 GB GPU, use public temporal RGB only by default, preserve a
frozen VLM and π0.5 in Stage 1, and make calibrated decisions over actual robot
capabilities.

## Decision

Use exactly:

1. one frozen shared VLM encoder for `(prompt, temporal RGB, public history)`
   and candidate text/region embeddings;
2. one lightweight cross-attention interaction decoder in which each candidate
   is a query over shared VLM tokens;
3. two related heads: factorized action effects and route logits;
4. temperature scaling plus split-conformal prediction sets;
5. one deterministic text adapter into frozen `pi05_libero`.

`DIRECT`, `OPEN`, `REMOVE_OCCLUDER`, `MOVE_CLOSER`, `ROTATE`, and `STOP` are
capabilities, not object-category rules. VLM-proposed targets remain open-vocabulary
referring expressions or normalized regions. The registry rejects unknown
primitives before model selection or execution.

## Why effects are factorized

The proposed categorical outcomes are not mutually exclusive. An interaction
may both reduce ambiguity and certify a region empty; a rejected location can
also be task-relevant progress. The effect head therefore predicts six
Bernoulli facts: execution success, task-relevant change, ambiguity reduction,
target confirmation, candidate rejection, and empty-region confirmation. The
route head consumes the same interaction representation and predicted effect
logits. No scalar information utility is constructed.

## Calibration and action rule

Fit temperature on an independent calibration split, then LAC conformal sets on
route probabilities. Calibrate effect factors separately with Bonferroni-adjusted
binary LAC sets. Execute a singleton route only when its required effect set is
also singleton. For a multi-action route set, execute only if exactly one
in-set physical information action has singleton execution-success and
ambiguity-reduction effects; otherwise abstain. This is a logical set rule, not
a weighted confidence formula. Coverage is reported as marginal coverage under
exchangeability, never as per-trial success probability.

## π0.5 interface

Stage 1 uses only a short deterministic instruction such as “Open the middle
drawer below the countertop.” The adapter never includes route probabilities,
effect values, free-form reasoning, or simulator state. Projected soft tokens,
layer-wise cross-attention, or action-expert LoRA remain later ablations after
the text loop is successful.

## Consequences

- The proposed package is independent of Grounding DINO, SAM, SigLIP, DINOv2,
  depth, EDL, and camera motion.
- Counterfactual execution, not human “uncertainty” labels, supplies effects and
  route supervision.
- The original drawer replay validates schemas and control wiring only. It is
  explicitly not trained-model evidence or final-task evidence.
- A method-paper claim is allowed only after the go/no-go gates in
  [`research_plan.md`](../research_plan.md) pass on group-disjoint scenes.
