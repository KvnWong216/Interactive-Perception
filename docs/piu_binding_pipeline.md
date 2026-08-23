# PIU target-binding pipeline

Status: software-ready, real causal gate and data collection pending.

This pipeline retains every valid frozen PaliGemma image/prompt token for the
pre/post observations, learns prompt-conditioned spatial binding on CPU, and
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

## Current blockers and claim boundary

No real binder may be trained or promoted until the prospective oracle
target-binding mechanism test is positive and new group-disjoint labels are
collected. The retained old 256-pixel visibility labels and retrospective groups
are not training data. Even a positive sealed spatial metric would not alone
prove improved frozen-VLA contact; causal rollout evidence is still required.
