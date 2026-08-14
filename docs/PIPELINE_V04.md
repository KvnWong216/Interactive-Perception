# Pipeline v0.4

## G4 revision: semantic conformal intent sets

The raw VLM confidence cutoff is deprecated as a paper method. Repeated π0.5
action chunks are decoded into the fixed primitive space and normalized only
after semantically equivalent chunks have been grouped. A split-conformal
calibrator then returns a set such as `{ACT}`, `{REMOVE_OCCLUDER}`, or
`{ACT, REMOVE_OCCLUDER}`.

The intended control rule is:

1. singleton `ACT`: execute the final task;
2. singleton information primitive: execute it, observe again, and replan;
3. ambiguous set containing an information primitive: acquire the cheapest
   information that can separate the candidates;
4. calibrated `NOT_FOUND` only after public search state is exhaustive;
5. otherwise safe-stop or ask for help.

`scripts/calibrate_semantic_intents.py` writes a versioned artifact with alpha,
policy ID, split ID, sample count, and finite-sample resolution. It refuses
fewer than 30 held-out samples by default. G4-v1 is now GO for the frozen
binary LIBERO scope: 40 prototype-training, 40 conformal-calibration, and 20
held-out validation observations yield 1.0 coverage and mean set size 1.0 at
alpha 0.1. It remains NOT-GO for `NOT_FOUND`, `ROTATE`, and `MOVE_CLOSER`.

Online decoding must exclude `role: task_target` before anchor resolution.
Those anchors reveal hidden simulator state and are permitted only in the
evaluator. Conformalizing an oracle-derived score does not remove the leak.

### Two uncertainty axes, not one score

The literature and the T01 control require two independent checks:

- **intent ambiguity:** do repeated chunks imply one semantic primitive or
  several? This is the conformal semantic set above;
- **executor competence:** can the frozen policy reliably realize the selected
  primitive in this visual context? This is measured by capability episodes and
  an exact binomial lower confidence bound.

A singleton `{REMOVE_OCCLUDER}` does not authorize execution when the
capability reliability gate fails. T01 demonstrates the distinction directly:
the requested intent is unambiguous, yet drawer opening reaches its endpoint in
only 3/5 trials. Flow/diffusion training loss would be a useful third OOD signal
(as in Diff-DAgger), but openpi's current websocket response does not expose it.
Adding it requires a separately versioned policy-server protocol and a fresh
reproduction gate.

SOMA-style spatial memory belongs only to the `VIEWPOINT_BLOCKED` branch. If
the NBV certificate says a target cannot be revealed by any tested view, camera
scanning cannot replace `REMOVE_OCCLUDER`.

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
