# Interactive Perception: PSR-VLA

Research code for **PSR-inspired predictive state conditioning in a vision-language-action model under partial observability**. This repository is the frozen software implementation of PSR-VLA V1 for the RSS 2027 project; it contains no claimed robot-performance result yet.

## Question

Given a task, current dual-camera observation, public robot state, and finite real interaction history, which open-vocabulary intention should a robot execute now to minimize final task failure?

## Frozen PSR-VLA V1

```text
H: task + current RGB + 8-D state + finite real history
                         │
              MolmoAct2 VLM + 6 soft positions
                         ▼
               predictive state tokens B
                         │
       two sampled intentions U + exact native option
                         │
         intent-only query  ×  cross-attention(B)
                         ▼
       p(C = failure | B,U) + p(E | B,U)
                         │
             argmin predicted failure probability
                         ▼
 [H,B,selected U,trigger] → native per-layer KV → Action Expert
                         │
                  real action and reobservation
```

- `H` is the public input and bounded real history.
- `B` is the contextual output at six trainable soft VLM positions; six is an engineering capacity, not six hand-authored concepts.
- `U` is an ordinary open-vocabulary short robot intention. Two independent samples share the same `[H,B]` prefix; a third option is the unchanged native MolmoAct2 route.
- `E` is a four-component diagonal Gaussian-mixture prediction of frozen, projected native visual patch features after the real 50-step intention window.
- `C` is the Bernoulli failure outcome after that window and the fixed native continuation, under the original 300-step episode budget.

The online selector uses calibrated `P(C=1 | B,U)`. `E` is a training target for the predictive representation, not a generated image and not a hand-designed uncertainty score. Only actual observations return to history.

The exact token order, tensors, losses, data firewall, stages, and scientific limits are specified in [docs/psr_v1.md](docs/psr_v1.md). Frozen defaults are in [experiments/psr_v1.yaml](experiments/psr_v1.yaml).

## Status

| Gate | Status | Meaning |
| --- | --- | --- |
| `IMPLEMENTED` | Yes | Source path, contracts, tests, CLI, runtime, collection, training primitives, and evaluation metrics exist |
| `VERIFIED_WITH_REAL_MODEL` | No | The pinned MolmoAct2 GPU canary and real LIBERO closed loop have not been run for this version |
| `TRAINED` | No | No Stage-A/S0/Stage-C/calibration artifact is claimed |
| `EVALUATED` | No | No held-out benchmark result or paper table is claimed |

These states are deliberately independent. CPU tests cannot establish that a real checkpoint loaded or that robot task success improved.

## Repository layout

```text
experiments/psr_v1.yaml             frozen PSR-VLA V1 configuration
docs/psr_v1.md                      complete method and reproduction contract
src/grounded_interaction/psr/
  config.py                          frozen configuration and architecture checks
  types.py                           H, U, observation and history contracts
  molmo_backend.py                   native embeddings, B, language U, KV and AE
  model.py                           intent-only encoder, B readout, E/C heads
  data.py                            strict real-record loading and split firewall
  training.py                        Stage A/C losses, S0, calibration, checkpoints
  collection.py                      same-reset real branches and immutable receipts
  runtime.py                         300/50/10 closed-loop execution
  libero.py                          public LIBERO RGB/state adapter
  evaluation.py                      calibration, ranking, bootstrap, execution metrics
  preflight.py                       weight-free and real-checkpoint canaries
  cli.py                             command line entry
tests/test_psr_*.py                 software-contract tests only
```

## Environment

Python 3.10+ is required.

```bash
uv sync --extra dev --extra learned
```

For the real MolmoAct2 checkpoint on Linux/NVIDIA:

```bash
uv sync --extra dev --extra molmoact2
```

The real-model environment is pinned in `pyproject.toml` to Torch 2.11.0, torchvision 0.26.0, and Transformers 4.57.6. LIBERO/robosuite remain external simulator dependencies and must use the project's existing simulator environment.

## Verify the software implementation

```bash
uv run pytest -q tests/test_psr_*.py
uv run ruff check src/grounded_interaction/psr tests/test_psr_*.py
uv run ip-psr preflight --config experiments/psr_v1.yaml --device cpu
```

CPU preflight does not download weights. On a suitable GPU host, the real interface canary is:

```bash
uv run ip-psr preflight \
  --config experiments/psr_v1.yaml --device cuda --load-model
```

The canary uses synthetic pixels/actions only to inspect interfaces and gradients. It is never admitted as training or evaluation evidence.

## Real workflow

All paths below must point to real user-provided assets or data; no command invents reset states, labels, checkpoints, or success outcomes.

```bash
ip-psr import-data --source /data/real-source --output /data/psr-warmup

ip-psr train-warmup \
  --config experiments/psr_v1.yaml --data /data/psr-warmup --output /runs/S0

ip-psr collect \
  --config experiments/psr_v1.yaml --snapshot /runs/S0 \
  --plan experiments/psr_v1/collection_plan.example.json --output /data/S0-rollouts

ip-psr train-outcomes \
  --config experiments/psr_v1.yaml --snapshot /runs/S0 \
  --data /data/S0-rollouts --output /runs/predictor

ip-psr calibrate \
  --snapshot /runs/S0 --predictor /runs/predictor \
  --data /data/calibration --output /runs/calibrated-predictor

ip-psr evaluate \
  --config experiments/psr_v1.yaml --mode psr --snapshot /runs/S0 \
  --predictor /runs/calibrated-predictor \
  --plan experiments/psr_v1/evaluation_plan.example.json --output /runs/psr-eval
```

Run `ip-psr <subcommand> --help` before execution. Commands that require a real environment/model fail closed when their concrete integration driver or assets are absent.

## Next scientific gates

1. Run native-bypass, token/KV/AE, cache-equivalence, gradient, and proposal canaries with the pinned real model.
2. Run one real LIBERO trace with at least two high-level decision boundaries.
3. Verify that the native continuation can exploit newly exposed evidence and that the sampled candidate set has same-reset oracle headroom.
4. Collect Stage-A demonstrations; freeze `S0`; collect real same-reset outcome branches; train `C/E`; fit temperature on a separate calibration split.
5. Evaluate native MolmoAct2, uniform selection, same-set rollout oracle, and the preregistered retraining ablations on untouched reset families.

## License and upstream models

This research repository does not redistribute MolmoAct2 or LIBERO weights/assets. Follow the upstream licenses and dataset terms when downloading or executing them.
