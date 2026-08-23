# PIU target-binding pipeline

Status: software-ready, real causal gate and data collection pending.

This pipeline retains every valid frozen PaliGemma image/prompt token for the
pre/post observations and every public candidate prompt at both times, learns
prompt-conditioned spatial binding on CPU, and
calibrates the frozen score function without exposing evaluator masks to the
policy. A passing synthetic regression proves only that the interfaces execute.

## Split firewall

1. `train`: adapter parameters and learned multi-task scales only.
2. `development`: architecture, learning rate, seed, and epoch checkpoint.
3. `calibration/temperature`: temperature fitting only.
4. `calibration/conformal`: finite-sample order statistics only.
5. `sealed_test`: one hash-authorized prediction and evaluation.

All branches from an initial state stay in one group. The trainer rejects every
split except train/development and records the exact group lists. Prediction
rejects overlap with model-selection groups; calibration rejects overlap between
its two roles; sealed evaluation rejects overlap with calibration groups.

## External-only preprocessing

The 1500 MiB local GPU cap prohibits local checkpoint extraction and simulator
mask rendering. On allocated external machines, first export evaluator labels:

```bash
python scripts/data/export_piu_binding_labels.py \
  --public PATH/public_transitions.jsonl \
  --evaluator PATH/evaluator_sidecars.jsonl \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --execution-location external_simulator \
  --output PATH/binding_labels.jsonl
```

Then extract the complete frozen prefix:

```bash
python scripts/data/extract_piu_spatial_prefix_features.py \
  --public PATH/public_transitions.jsonl \
  --checkpoint PATH/pi05_libero \
  --external-gpu \
  --output PATH/spatial_prefix_features.npz
```

Both commands are immutable and hash their source artifacts. The mask exporter
replays retained simulator states and verifies the raw per-camera pixel counts
against evaluator sidecars. Masks are labels only.
The executed primitive embedded by the binder is copied from
`public_action_history.last_executed_candidate`; extraction rejects a missing or
non-candidate action instead of substituting the evaluator label. The
interaction-pre image is context only. Spatial supervision is nonzero only on
the current/interaction-post patches, preventing a pre-action sighting from
satisfying the post-action localization objective.

Per-observation outputs are assembled by prospective role before extraction;
concatenating JSONL files manually is not an admissible dataset operation:

First copy the null-only template
`configs/templates/piu_learning_collection_budget_template.yaml` to
`results/method/piu_learning_collection_budget_v1.yaml` and have the external
resource owner fill/freeze each role count, seed namespace, authority, and
rationale before successor collection. The repository provides no suggested
counts. Then build the immutable learning split while excluding the already
used qualification and formal-oracle manifests:

```bash
python scripts/data/build_piu_learning_split_manifest.py \
  --budget results/method/piu_learning_collection_budget_v1.yaml \
  --exclude-split PATH/open_qualification_split.json \
  --exclude-split data/piu/mainline_v1/oracle_formal_split_manifest.json \
  --output data/piu/mainline_v1/learning_split_manifest.json
```

The counts are collection-resource allocations, not performance gates. A class
without calibration support remains `UNSUPPORTED`; data collection may not be
extended after inspecting its outcomes. Assemble each role next:

```bash
python scripts/data/assemble_piu_public_binding_role.py \
  --split-manifest data/piu/mainline_v1/learning_split_manifest.json \
  --split-role train \
  --public PATH/transition_1.jsonl --public PATH/transition_2.jsonl \
  --binding-label PATH/label_1.jsonl --binding-label PATH/label_2.jsonl \
  --output-public data/piu/mainline_v1/train/public.jsonl \
  --output-labels data/piu/mainline_v1/train/binding_labels.jsonl \
  --output-manifest data/piu/mainline_v1/train/public_binding_manifest.json
```

Repeat with distinct development, binder-temperature, binder-conformal,
effect-temperature, and effect-conformal roles. The assembler requires an
exact sample join and the complete set of groups assigned to that role.

Prospective transitions are created only after a qualified dispatcher writes an
immutable receipt. `scripts/data/export_piu_public_transition.py` verifies the
controller, qualification certificate, execution report, low-level history,
and keyframe hashes, then projects only public observations, the full public
candidate set, and `last_executed_candidate`. Although the source execution
report contains an evaluator section, no evaluator field is copied into the
public JSONL.

## CPU train and calibration

Training uses `configs/experiments/piu_binding_adapter_v1.yaml` and
`scripts/training/train_piu_target_binding.py`. The script clears CUDA
visibility before importing the learned stack, searches only the declared grid,
and writes the checkpoint, all development trials, and raw development scores.

Run `scripts/evaluation/predict_piu_target_binding.py` separately on the two
calibration roles. Fit the artifact with:

```bash
python scripts/calibration/calibrate_piu_target_binding.py \
  --config configs/experiments/piu_binding_calibration_v1.yaml \
  --temperature-predictions PATH/temperature.npz \
  --temperature-report PATH/temperature.json \
  --conformal-predictions PATH/conformal.npz \
  --conformal-report PATH/conformal.json \
  --output PATH/binder_calibration.json
```

The fitted boundary is a finite-sample calibration statistic, not a manually
chosen confidence threshold. Formal evaluation uses
`scripts/evaluation/evaluate_piu_target_binding.py` after a
`piu.sealed-test-authorization.v1` manifest binds the checkpoint, feature cache,
labels, and single output path.

Offline prediction/evaluation joins evaluator labels only after public inputs
are built. Online inference uses
`scripts/pipeline/predict_piu_target_binding_online.py`, accepts no label path,
and writes the learned `target_token`, logits, masks, patch coordinates, camera
IDs, and temporal IDs needed by the controller. The frozen binder prediction
artifact also exposes its learned `target_token`.
That token is the only binding interface consumed by the downstream
candidate-conditioned action-effect model. Evaluator masks, target patch
targets, and binder labels are not joined into the policy artifact.

## Current blockers and claim boundary

No real binder may be trained or promoted until the prospective oracle
target-binding mechanism test is positive and new group-disjoint labels are
collected. The retained old 256-pixel visibility labels and retrospective groups
are not training data. Even a positive sealed spatial metric would not alone
prove improved frozen-VLA contact; causal rollout evidence is still required.
