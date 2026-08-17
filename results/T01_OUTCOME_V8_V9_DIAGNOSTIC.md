# T01 temporal outcome status

## Frozen v3 development

- Dataset: 180 rows, seeds 600--659.
- SHA-256: `0af9eec229a74d17a37deab6df1cf6b2e27104633cf23cc2238ced9008c57c3c`.
- REVEALED physical branch: 60/60; one-sided 95% lower bound 0.951.
- Intended EMPTY branch under the v3 five-pixel label: 56/60; lower bound
  0.854. It passes 0.80 but not the original 0.90 requirement.
- Hierarchical v8 development: FAILED 7/7, REVEALED 7/9, EMPTY 4/5.
- Decision: NOT-GO. Seeds 900--999 remain sealed.

## Root cause and v9 candidate

True middle-layer reveals contain at least 311 target pixels. The three
EMPTY-scene rows labeled REVEALED contain only 5, 5, and 7 pixels from the
upper layer. They are simulator-visible but not a usable prompt-level
observation.

The v9 candidate requires one pi0.5 visual-token footprint:
`(256 / 16)^2 = 256` target pixels. This value comes from the frozen image and
token grid, not a classifier-score sweep. On the already diagnosed seeds it
reaches 20/21: FAILED 7/7, REVEALED 7/7, EMPTY 6/7. That is diagnostic only.
Untouched seeds 660--699 must independently pass before the sealed audit can
open.

Final retrieval after reveal remains 0/5. No paper-level success is claimed.
