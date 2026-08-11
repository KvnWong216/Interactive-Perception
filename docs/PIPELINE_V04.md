# Pipeline v0.4

## What changed

The rollout now preserves stock `agentview` pixels for pi0.5 inference. The
horizontal pose is applied only while a demo frame is rendered, then the camera
model is restored before policy inference or physics stepping. This removes the
camera-distribution confound measured by the T04 control without giving up the
more legible presentation view.

The runner also has an uncertainty-blind comparison arm:

```bash
env -u PYTHONPATH ../.conda/envs/ipu/bin/python scripts/run_challenge_rollout.py \
  --arm fixed-rule --information-skill-steps 120
```

For tasks declaring an information interaction, this arm runs the task's
`capability` prompt for a bounded prefix and then restores the unchanged final
goal prompt. It is intentionally benchmark-aware and must be reported as an
ablation, not as the uncertainty method. `--arm monolithic` retains the
original one-prompt baseline.

## New challenge conditions

- `T01_multi_drawer_search` replaces the original T01 and adds three closed candidate drawers with a prior
  ordering that points first to the wrong drawer. The target is in the middle
  drawer. Its 360-pose certificate is `NBV_INSUFFICIENT` (pre 0 px, post 1235
  px on seed 0).
- `T06_severe_clutter_occlusion` replaces the original T06 and expands clutter from six to nine objects and
  requires clearing multiple named occluders. Across validation seeds it has
  7--8 visible distractors and at least a 1.8x target-visibility gain after the
  oracle clear. Its 360-pose certificate is `NBV_SUFFICIENT`, as expected for
  clutter (pre 2060 px, post 2064 px on seed 0).

Both scenes pass scene validation on seeds 0, 1, and 2. GPU policy performance
has not yet been measured. The reproduction gate and T04 control remain hard
prerequisites for interpreting a new rollout.

## Still missing

The fixed-rule arm is the required arm-C comparator, not the closed-loop
router. The closed-loop path is selected with `--arm uncertainty-router` and
requires an RGB-only evidence endpoint:

```bash
env -u PYTHONPATH ../.conda/envs/ipu/bin/python scripts/run_challenge_rollout.py \
  --arm uncertainty-router \
  --perception-endpoint http://127.0.0.1:8101/v1/public-evidence
```

The request contains only `task_id`, the final prompt, and a base64 PNG. The
response is:

```json
{
  "target_visible": false,
  "target_sufficient": false,
  "locations": {"top_drawer": "searched_empty", "middle_drawer": "closed"},
  "occluders": {"soup_occluder": "blocking"},
  "confidence": 0.91
}
```

Allowed location states are `closed`, `open_unsearched`, and `searched_empty`;
allowed occluder states are `blocking` and `cleared`. Low-confidence evidence
causes a safe stop. When every declared location is `searched_empty`, the
runner terminates with `NOT_FOUND`. Every nonterminal decision becomes a short
pi0.5 prompt, after which the runner obtains a new RGB observation and routes
again.

The evidence server itself must be frozen and versioned before the experiment.
Evaluator segmentation and `task_target` anchors must never be used to
implement it.

Run all three paper arms with:

```bash
PERCEPTION_ENDPOINT=http://127.0.0.1:8101/v1/public-evidence \
  bash scripts/run_method_comparison.sh
```
