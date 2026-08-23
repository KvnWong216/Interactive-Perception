# Original-drawer method cycle report

> **Superseded physical qualification:** the later ten-seed executed matrix,
> terminal-state relabels, and actual-effect development CV are reported in
> [`paper/main.md`](../paper/main.md) and
> `results/method/original_drawer_paper_cycle_v2.json`. This document remains
> the immutable record of the earlier proxy-effect pilot.

**Frozen on:** 2026-08-22

**Scenario:** unchanged `T01D_hidden_butter_retrieval.bddl`

**Decision:** `PILOT_ONLY_NOT_METHOD_EVIDENCE`

This cycle validates the minimal learned pipeline and identifies its current
blocking component. It does **not** establish the paper-level method claim:
route labels and shared-VLM features are real and seed-disjoint, while effect
labels are repeated seed-1399 capability proxies rather than executed
counterfactual outcomes for every seed.

## What was run

The fixed scene contains butter in the closed middle drawer and visible cream
cheese on the work surface. Each simulator seed produces two samples with
identical RGB and different prompts:

- `Place the butter in the basket` -> `OPEN_TO_INSPECT`;
- `Place the cream cheese in the basket` -> `DIRECT_ACT`.

Seeds are grouped and disjoint: train 2000-2015 (32 samples), development
2016-2019 (8), calibration 2020-2027 (16), and held-out test 2028-2035 (16).
The policy index contains public RGB, public robot state, and prompt only;
segmentation, semantic instance IDs, visibility, and location labels are kept
in a physically separate evaluator index.

The frozen `pi05_libero` PaliGemma prefix is reused as the only multimodal
encoder. It emits four 2048-wide context summaries and one 2048-wide token for
each of three public candidates. The learned component is one 128-wide
cross-attention decoder with route and factorized-effect heads. Input
LayerNorm is non-affine. Route-only B6 and effect+route B7 each use three
training seeds; the effect weight is fixed from initial shared-trunk gradient
norm matching. Model selection uses development only, and temperature/LAC
calibration uses the isolated calibration split.

## Held-out routing and ablations

| variant | test accuracy, mean +/- std | macro F1, mean +/- std | false-direct |
|---|---:|---:|---:|
| B6 route-only | 95.83% +/- 2.95% | 95.82% +/- 2.96% | 0.00% |
| B7 effect+route | 93.75% +/- 0.00% | 93.73% +/- 0.00% | 0.00% |

For the development-selected seed, both variants score 93.75% single-sample
accuracy and 87.5% paired-seed accuracy. The controlled feature diagnostics
show that this performance depends on the task condition:

| held-out feature condition | B6 accuracy / paired | B7 accuracy / paired |
|---|---:|---:|
| unaltered | 93.75% / 87.5% | 93.75% / 87.5% |
| average the two same-RGB prompt features | 50.00% / 0.0% | 50.00% / 0.0% |
| swap the two same-RGB prompt features | 6.25% / 0.0% | 6.25% / 0.0% |

This is evidence that the router uses prompt-conditioned features instead of
solving the pair from RGB alone. It is a cached-feature causal diagnostic, not
a substitute for a fresh no-prompt encoding. B7 does not improve routing over
B6, so the effect head currently fails its keep criterion.

## Calibration iteration

The first calibration applied Bonferroni correction across all six effect
factors. With only 24 observed candidate rows this forced every binary set to
`{0,1}`, producing 100% abstention. The second iteration applies simultaneous
protection only to `execution_succeeded` and `ambiguity_reduced`, the two
factors capable of authorizing an action. Other effect factors receive
individual marginal calibration and are diagnostic only.

On 16 held-out samples, the revised B7 controller has 93.75% route coverage,
mean set size 1.0, 93.75% singleton precision, 68.75% abstention, 25.00%
correct execution, and 6.25% wrong execution. These are descriptive pilot
results under exchangeability; conformal coverage is not a per-example success
probability. The residual 1/16 wrong execution prevents a safety claim.

## Frozen pi0.5 executor qualification

All runs use seed 1399, the same BDDL, public observations only in the
controller, and privileged state only in the separate replay evaluator.

| condition and candidate | key result | final task |
|---|---|---:|
| closed drawer, DIRECT butter | butter distance 0.2697 m; 0 butter contacts; 59 cream-cheese contacts | fail |
| closed drawer, OPEN | drawer minimum joint -0.1612; open succeeds | fail, as expected for information-only action |
| actual OPEN state, DIRECT butter | butter distance 0.1414 m; 0 butter and 5 cream-cheese contacts | fail |
| visible cream cheese, DIRECT | pick succeeds; lift 0.2958 m; destination not reached | fail |
| ideal open, original prompt | butter distance 0.1476 m; no contact | fail |
| ideal open, exact task prompt | butter distance 0.1550 m; no contact | fail |
| ideal open, source-aware prompt | butter distance improves to 0.0778 m; no contact | fail |

Aggregate: OPEN succeeds 1/1, target pick succeeds 1/7, the object-specific
destination predicate is 0/1 on the newly instrumented run, complete task
success is 0/7, and online oracle input count is zero. Prompt wording changes
distance but does not cross the grasp/task gate, so further prompt patching is
stopped.

## Go/no-go and next controlled iteration

- **Keep:** shared frozen prefix, compact candidate decoder, grouped
  same-state prompt counterfactual, isolated calibration, and explicit
  abstention.
- **Do not claim:** learned effects, full closed-loop improvement, placement
  success, multi-primitive generalization, or paper-level novelty evidence.
- **Do not scale yet:** B7 is not better than B6 and current effect labels lack
  per-seed counterfactual signal.
- **Immediate causal gate:** on the unchanged post-OPEN states, render an
  evaluator-only target box/point/spotlight into the two policy RGB streams.
  Screen three styles on seeds 1400/1403/1406, then require at least 4/5 target
  picks and at most 1/5 wrong-object contacts on the disjoint confirmation
  seeds. This is an oracle upper bound, not public-input method evidence.
- **If the oracle gate passes:** learn a public-RGB target-binding adapter and
  evaluate predicted prompts on new initial-state groups and scenes.
- **If the oracle gate fails:** stop visual-binding work and add a registered
  target-conditioned grasp-and-place primitive before collecting router data.
- **Deferred data gate:** only after an executor repair passes, execute
  `DIRECT`, `OPEN`, and `STOP` forks from at least 20 fresh calibration seeds
  and 20 untouched test seeds with evaluator-only effect labels.

## Reproduction

The retained source artifacts are:

- protocol: `benchmarks/calibrated_interaction/original_drawer_v1.yaml`;
- public data and separate labels: `data/calibrated_interaction/original_drawer_v1/`;
- frozen features: `outputs/calibrated_interaction/original_drawer_v1/`;
- final pilot: `results/method/original_drawer_pilot_v4_reproducible.json`;
- executor summary: `results/method/original_drawer_executor_qualification_v1.json`.
- oracle gate protocol:
  `configs/experiments/original_drawer_oracle_target_prompt_gate_v1.yaml`;
- policy-free oracle packet preflight:
  `results/diagnostics/original_drawer_oracle_prompt_preflight_v1.json`.

Feature extraction uses the sibling openpi environment and a bounded JAX
allocator:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
  /home/icon/InteractivePerception_yg/openpi/.venv/bin/python \
  scripts/data/extract_candidate_interaction_features.py \
  --dataset data/calibrated_interaction/original_drawer_v1/test.jsonl \
  --candidates configs/experiments/original_drawer_candidate_set.yaml \
  --checkpoint /home/icon/InteractivePerception_yg/checkpoints/checkpoints/pi05_libero \
  --output outputs/calibrated_interaction/original_drawer_reproduction/test/shared_vlm_features.npz \
  --batch-size 2
```

Train a new immutable result from all four retained feature splits:

```bash
.venv/bin/python scripts/training/train_calibrated_interaction.py \
  --data-root data/calibrated_interaction/original_drawer_v1 \
  --feature-root outputs/calibrated_interaction/original_drawer_v1 \
  --candidates configs/experiments/original_drawer_candidate_set.yaml \
  --effect-proxy configs/experiments/original_drawer_effect_proxy_v1.yaml \
  --output results/method/original_drawer_reproduction.json \
  --model checkpoints/calibrated_interaction/original_drawer_reproduction.pt \
  --epochs 600 --alpha 0.1
```

Validation command and result:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q
# 54 passed
```

The repository-wide Ruff baseline still contains pre-existing legacy findings;
all files added or modified in this method cycle pass targeted Ruff checks.
