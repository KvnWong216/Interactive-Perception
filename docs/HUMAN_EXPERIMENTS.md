# Experiments that still require a human decision or new data

Everything below is blocked by scientific input, not unfinished plumbing.

## 1. Freeze the paper configuration

Record immutable identifiers for:

- π0.5 checkpoint and openpi commit;
- second VLA checkpoint (OpenVLA or Octo);
- public RGB perception model, if retained;
- benchmark commit and calibration/test split.

Do not choose these after viewing test performance.

## 2. Collect the G4 calibration split

Use scenes and task instances disjoint from T01–T06. Collect at least 30
independent episodes; 100+ is preferable for a 90% coverage target because the
finite-sample resolution is `1/(n+1)`. For each episode:

1. freeze one RGB observation and task prompt;
2. draw the preregistered number of independent π0.5 action chunks;
3. decode only against public anchors—never `task_target`;
4. label the correct coarse intent from the task construction;
5. save `primitive_evidence`, `true_intent`, task, seed, checkpoint, and split.

Fit with:

```bash
env -u PYTHONPATH ../.conda/envs/ipu/bin/python \
  scripts/calibrate_semantic_intents.py calibration.jsonl \
  --output artifacts/semantic_conformal.json \
  --alpha 0.1 \
  --policy-id <checkpoint-and-commit> \
  --split-id <frozen-split-name>
```

Before the main experiment, verify empirical coverage and mean prediction-set
size on a validation split that is separate from both calibration and test.

## 3. Choose physical losses

Measure action time, collision/contact failures, false `NOT_FOUND` cost, and
failed manipulation cost. Convert them to a common unit before using Bayes
risk. Do not tune loss ratios on T01–T06 success rate.

## 4. Run the preregistered paper experiment

Required arms:

- monolithic final-goal π0.5;
- fixed interaction rule;
- semantic uncertainty without conformal calibration;
- conformal semantic router;
- oracle information action as an upper bound.

Required controls:

- stock LIBERO reproduction;
- T04 visible target;
- capability prompt for every physical information skill;
- policy-visible RGB audit proving no segmentation or hidden pose is consumed.

Report task success, information-endpoint rate, conformal coverage, set size,
interactions, time, and failure taxonomy with confidence intervals. Use enough
seeds to distinguish the expected effect; five seeds remain a debug gate.

## 5. Generality and robustness

Repeat the frozen protocol with a second VLA. Sweep clutter and occlusion
severity without moving thresholds. Re-run scene validation and NBV
certification after every geometry change.

## Claims not yet permitted

- π0.5 is broadly overfit;
- the router improves task success;
- conformal calibration guarantees robot success;
- the method generalizes across VLAs;
- segmentation wrapper caused the failures.
