# Spatial-prefix successor contract

## Status and hypothesis boundary

This is a design contract, not an implemented or successful method claim. It
may be activated only if the evaluator-only oracle experiment establishes on a
prospectively sized formal set that spatial target binding can causally improve
the frozen executor. Until then, the retained four-token Stage 1 pipeline is a
rejected development baseline.

## What Stage 1 actually tested

Stage 1 runs the frozen `pi05_libero` PaliGemma prefix and then keeps four
2048-wide vectors: the last valid prompt state, masked prompt mean, global
agent-view mean, and global wrist-view mean. A candidate is compressed again
to the fixed mean of its last-prompt and prompt-mean vectors.

These tensors are genuine frozen-VLM hidden states, but their pooling is
hand-designed. The pi0.5 action expert does not consume these four summaries:
it queries the complete image/language prefix KV cache. Consequently, Stage 1
shares encoder weights and feature width with the executor but not its actual
representation interface. It also uses one observation pair and an initially
empty history, so it does not test temporal evidence binding.

## Required successor representation

A successor must retain, for every public pre/post interaction observation:

- every valid post-PaliGemma image and prompt token, with the exact padding
  mask;
- explicit camera span and token order, plus the 2-D patch coordinate implied
  by the frozen image encoder;
- the public action/option identity and pre/post temporal boundary;
- candidate prompt token sequences and masks, without a fixed two-summary
  average;
- the frozen checkpoint tree identity and the OpenPI source revision defining
  token order.

The learned component may project the 2048-wide frozen tokens to a smaller
width and use candidate-to-context cross-attention. Pooling/readout weights
must be learned on training groups. Any action or abstention boundary must be
fit on an isolated calibration split. Simulator masks, instance IDs, object
poses, contacts, and oracle marker pixels remain forbidden inputs.

## Mandatory causal ablations

Development must compare the spatial successor against:

- the frozen four-token global-mean baseline;
- no action-history and no pre-action-frame variants;
- agent-view-only and wrist-only variants;
- shuffled spatial positions and shuffled pre/post order;
- route-only versus executed-effect supervision;
- uncalibrated versus isolated-calibration decision sets;
- the separately labeled oracle ceiling.

The main estimands are target grasp contact, wrong-object grasp contact,
continuous target lift, destination entry, terminal placement, abstention, and
final task success. Visibility stays continuous. Architecture is selected on
development groups only; calibration and sealed-test groups cannot influence
pooling, widths, stopping, or thresholds.

## Compute contract

Full-prefix extraction remains an external-GPU preprocessing job because the
local GPU budget is 1500 MiB. Cached frozen tokens may be stored in float16;
the small learned projection/decoder must fit the local budget. The exact token
shape is discovered and retained from the checkpoint output rather than
hard-coded as a paper assumption.
