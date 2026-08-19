# PIU V0 learning protocol

PIU V0 is a frozen-encoder, multi-head supervised prototype. It is designed to
make the entire inference and training path executable before adding a larger
object-level visual front end. It is not evidence for cross-scene
generalization.

## Runtime factorization

The frozen `pi05_libero` PaliGemma prefix consumes stock agentview RGB, wrist
RGB, the user prompt, and public robot state. A deterministic projection feeds
five small heads:

1. initial `target_location` belief;
2. semantic subtask rank;
3. pre-action outcome distribution and task progress;
4. six-frame public-RGB v11 `FAILED / REVEALED / EMPTY` critic;
5. action/outcome-conditioned future location belief.

The planner computes normalized entropy from the structured fact distribution
and compares hard-valid candidates with explicit expected information utility.
It does not route on a confidence threshold.

## Losses and supervision

The current training script optimizes cross-entropy for belief, rank, effect,
outcome, and future belief, plus binary cross-entropy for task progress. All
model inputs are frozen public VLM features. Drawer joints, segmentation, and
scenario truth occur only in offline label construction after controller
termination.

V0 intentionally omits map/mask loss because no public object-level front end
is installed yet. V1 adds Grounding DINO/SAM/DINOv2 or an equivalent open
backend, then trains node relevance and uncertainty-map heads with BCE + Dice.
It must be an explicit model arm, not silently mixed into V0.

## Freeze order

1. Freeze data index and source hashes.
2. Train on prototype blocks only.
3. Freeze weights.
4. Fit class-conditional conformal thresholds on calibration blocks.
5. Run diagnostic/development once without tuning on its errors.
6. Only after a scene-disjoint clean development GO may a sealed audit run.

The current v0.3 sidecar uses old development blocks 220–259 and 600–652.
Seeds 653–699 and 1400–1439 are contaminated diagnostics/model-selection data.
The v11 outcome cascade consumes only the six stock agentview/wrist RGB pairs.
Seeds 1440–1479 remain frozen clean development and 900–999 remain sealed.

## Commands

```bash
../.conda/envs/ipu/bin/python scripts/build_piu_v0_dataset.py
../.conda/envs/ipu/bin/python scripts/train_piu_v0.py \
  --epochs 500 \
  --output results/models/piu_v0_3_sidecar.pt \
  --report results/training/piu_v0_3_training.json
EXPERIMENT_GPU_INDEX=0 bash scripts/run_piu_v0_smoke.sh \
  --model results/models/piu_v0_3_sidecar.pt \
  --output results/smoke/piu_v0_v11_full_pipeline_v1_seed1399.json \
  --asset-dir results/assets/piu_v0_v11_full_pipeline_v1_seed1399/raw
../.conda/envs/ipu/bin/python scripts/render_piu_v0_assets.py \
  --report results/smoke/piu_v0_v11_full_pipeline_v1_seed1399.json \
  --output-dir results/assets/piu_v0_v11_full_pipeline_v1_seed1399/visualizations_v1
```

These are the exact recorded arguments. Artifacts are immutable, so a rerun
must preserve the arguments while using new versioned model, report, and smoke
output paths.
