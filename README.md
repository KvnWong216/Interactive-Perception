# Interactive Perception: Geometry-aware Predictive VLA

Research implementation of a direct interactive policy built on pretrained
MolmoAct2. Geometry and real observation/action history enter one VLM; the native
per-layer K/V connection drives its continuous Action Expert. During training,
a small action-conditioned predictor aligns shared states with frozen features
of actual future observations. Deployment executes an action prefix, observes
again, and uses the native language head to continue or answer and stop.

The combined architecture and information-seeking transfer are research
hypotheses. The [unified experiment report](docs/experiment_log.md) covers
architecture and evidence, training configuration and costs, then measured
results and hypothesis tests. No IP transfer result is claimed.

## Current route

Read the [unified experiment report](docs/experiment_log.md) for the source
mechanisms, implementation choices, scientific limits, and staged training plan.
The old PSR candidate-ranking/branch-collection/calibration pipeline has been
removed; its history remains in Git. It is not a second active method.

The core implementation is repository-owned PyTorch code. Referenced papers
inform geometry binding, temporal causality, action-conditioned prediction and
receding-horizon control; their complete model stacks are not dependencies.
The pinned MolmoAct2 weights, processor, normalization and AE remain upstream.

The [native-versus-best results](docs/assets/stage1_native_best/index.html)
contain completed stage-one history tests, paired geometry results, and real
comparison videos. The [scene generalization run](docs/assets/stage1_generalization/index.html)
extends this comparison to 12 tasks and 96 scene conditions on GPU 0.
See section 2.12 of the unified report for its splits and limits.
The [expanded evaluation](docs/assets/stage1_expanded/index.html) now covers
40 candidate tasks plus cross-object position/history probes on GPUs 0/4/5/6/7.
It retains both positive and negative results and all scene exclusions; see
section 2.13 for the fixed protocol, budget and distinction between candidate
and actually eligible scenes.
The broader [evaluation protocol](experiments/evaluation_protocol.yaml) remains
**design only** for stage-two mechanisms and training controls. Its
[asset index](docs/assets/validation/index.html) retains the original plans.

| Source | Responsibility |
| --- | --- |
| `predictive_vla/geometry.py` | Native patch layout, metric backprojection, unknown-depth mask, residual injection |
| `predictive_vla/backend.py` | Temporal input assembly, block-causal attention, native per-layer KV/AE and language head |
| `predictive_vla/native.py` | Small native-API checks, switchable low-rank deltas, action postprocessing |
| `predictive_vla/model.py` | Training-only cross-attention patch predictor conditioned on real controls |
| `predictive_vla/types.py`, `data.py` | Public history, actual action alignment, trajectory loading and grouped splits |
| `predictive_vla/training.py` | Joint flow/prediction/language losses and adapter checkpoints |
| `predictive_vla/runtime.py` | Applied-prefix feedback and policy completion |

All source paths above are under `src/grounded_interaction/`.

## Local verification

```bash
uv sync --extra dev --extra learned
uv run pytest -q
uv run ruff check src tests
uv run python -m grounded_interaction.predictive_vla check
```

`check` validates configuration and reports dependency availability. It does not
load weights or establish real-model compatibility. The configuration is
[experiments/predictive_vla.yaml](experiments/predictive_vla.yaml): by default,
10-step training targets, 5-step execution prefixes and three observed frames.

## Existing real data and training

Prepare the [trajectory manifest and arrays](docs/trajectory_format.md), then:

```bash
ip-vla check-data --data /absolute/path/to/manifest.json
ip-vla train --config experiments/stage1_policy.yaml --data /absolute/path/to/manifest.json --output /absolute/path/to/new-run --device cuda
```

Commands use the in-repository loader/backend/trainer; no external factory is
required. Train and validation reset families must be disjoint. The output
contains `best.pt`, `last.pt`, configuration, optimizer/scheduler/RNG state and
per-update performance logs. Stage one uses
[stage1_policy.yaml](experiments/stage1_policy.yaml): VLM LoRA plus the full AE,
8 flow samples per real action chunk, and action supervision alone.

The checked stage-one entry point is `bash scripts/run_stage1.sh`; it requires
finalized data and current real-model checks. Read the experiment log before
using another configuration.

Stage two now starts with **frozen predictor qualification**, following the
September 23 diagnosis in [the experiment log](docs/experiment_log.md).
`train_transition_predictor.py` fits only the transport head on frozen caches;
`qualify_transition_predictor.py` tests three seeds and action-blind controls on
fresh, disjoint task families. Cached features must identify their source policy.
The existing diagnostic caches use stage-two weights and cannot qualify a head
for a stage-one warm start.

The joint entry point remains `scripts/run_stage2.sh` on GPUs **4–7**, global
batch 32, but now uses [stage2_transition.yaml](experiments/stage2_transition.yaml)
and requires `--predictor-init` and `--qualification-report` for both preflight
and the main run. It restores stage-one **best**, imports only a qualified head,
and uses new output directories. No new production training has started.
The old absolute predictor and its configuration remain for historical
checkpoint evaluation and explicit reproduction, not as the default next run.

For the pinned real model environment use `uv sync --extra dev --extra molmoact2`.
LIBERO and robosuite run in a separate `.venv-sim`. The repository-owned
`scripts/prepare_assets.py` enforces a pinned download allowlist;
`scripts/prepare_libero.py` reconstructs calibrated RGB-D from recorded states.
`scripts/preflight_stage1.py` checks the real pretrained model, gradients,
small-set fitting and checkpoint recovery. `scripts/check_native_rollout.py`
checks two ordinary LIBERO control episodes through an isolated simulator.
The full IP benchmark remains to be constructed and evaluated.

## Evidence status

- Tensor/runtime contract tests exist; see `tests/test_predictive_vla.py`.
- Real checkpoint forward/gradient/native-parity and recovery: **passed**;
  see `data/preparation/preflight.json` for this local run's evidence.
- LIBERO RGB-D reconstruction and calibration: **checked**; full control-loop
  check and data preparation status are recorded in the experiment log.
- Stage-one action training: **completed**, 2,000 updates in 12.25 hours; best
  fixed-window validation loss 0.09225 at update 1,500. See the unified report.
- Post-training paired development checks: **passed**, 4/4 ordinary rollouts
  for native, best and last; these training starts are not a benchmark estimate.
- Initial geometry diagnostic: **120 episodes completed**, no aggregate evidence
  for a geometry-specific improvement. History has a positive signal on **one
  development layout only**. Broader transfer and active-perception tests remain
  pending; see the validation asset index.

Frozen future features are supervision, not calibrated uncertainty. Neither
attention magnitude nor a lower prediction loss demonstrates information-seeking
behavior. That claim requires paired evidence-history tasks and controlled
training ablations described in the unified experiment report.

Follow upstream model, simulator and dataset licenses. This repository does not
redistribute their weights or assets.
