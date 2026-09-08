# Environment guide

The repository deliberately separates three dependency profiles. MolmoAct2 and
the pinned legacy LIBERO stack cannot be placed in one clean environment
without violating one side's version contract.

## 1. Core software verification

Use this profile for contracts, serializers, selectors, model/loss code, smoke
tests, and artifact-validation tests that do not load the real VLA or LIBERO.

Requirements:

- macOS or Linux;
- Python 3.10 or newer; and
- [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync --extra dev --extra integration
uv run pytest -q
uv run ip-smoke --output runs/formal_v1_smoke.json
```

The smoke command uses scripted scores and replayed outcomes. A valid smoke
artifact declares `software_verification_only=true` and
`empirical_evidence=false`.

For PyTorch forward/backward checks, add the `learned` extra on a platform that
supports the resolved PyTorch wheel:

```bash
uv sync --extra dev --extra integration --extra learned
uv run pytest -q
```

On Linux, the repository resolves PyTorch from the official CUDA 12.8 index so
the frozen MolmoAct2 environment receives the required build. The source is
platform-gated: macOS falls back to PyPI for an available native wheel. Neither
profile is a recipe for the old macOS LIBERO environment.

## 2. MolmoAct2 GPU server

### Required contract

- Linux with an NVIDIA GPU;
- Python 3.11 is recommended; the pinned upstream range is `>=3.11,<3.13`;
- PyTorch `2.11.0` and torchvision `0.26.0` from the CUDA 12.8 wheel index;
- Transformers `4.57.6`;
- `sentencepiece` installed before tokenizer initialization; and
- sufficient local/cache storage for the 21.8 GB frozen inference snapshot.

Install from the repository root:

```bash
uv sync --extra molmoact2 --extra dev
```

Keep Hugging Face and uv caches on the laboratory's persistent user storage
rather than filling a small home or system partition. For example:

```bash
export HF_HOME=/ICONAS/users/$USER/huggingface
export UV_CACHE_DIR=/ICONAS/users/$USER/uv-cache
```

First download, parse, and hash the model without allocating GPU weights:

```bash
uv run ip-serve-molmoact2 \
  --inspect-only \
  --checkpoint allenai/MolmoAct2-LIBERO \
  --revision 0d24a92bd1faf321ef497c3bbd5681af97c65aa2
```

The report must match the `executor` object in
`experiments/e1_referent_ceiling/pilot_state0_v1.json`, including the 19-file
manifest hash, config/norm hashes, dimensions, horizon, normalization mode, and
delta-control semantics.

Before loading the model, inspect the authorized physical GPU with
`nvidia-smi`. In the ICON Lab execution environment for this project, only
physical GPU 1 is authorized. Select it before process start; inside PyTorch it
then appears as `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run ip-serve-molmoact2 \
  --identity-json experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --checkpoint allenai/MolmoAct2-LIBERO \
  --revision 0d24a92bd1faf321ef497c3bbd5681af97c65aa2 \
  --host 127.0.0.1 --port 8003
```

The service is intentionally bound to localhost. Its health endpoint is:

```bash
curl -s http://127.0.0.1:8003/health
```

Do not expose the service publicly. From another machine, create an SSH tunnel:

```bash
ssh -N -L 8003:127.0.0.1:8003 USER@GPU_HOST
```

The current code uses the checkpoint's official remote-code `predict_action`
API. It does not reimplement the action decoder. The server requires exact
`10×7` finite output, BF16, `norm_tag=libero`, continuous action mode, 10 flow
steps, normalized language, and disabled depth reasoning/CUDA graph.

## 3. Pinned LIBERO simulator process

E1 was preflighted with this already-existing environment:

| Component | Version |
| --- | --- |
| Python | `3.10.20` |
| NumPy | `1.22.4` |
| PyTorch | `1.11.0` |
| robosuite | `1.4.0` |
| MuJoCo | `2.3.7` |
| LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

The E1 plan freezes the exact BDDL and pruned-init file hashes as well. Point
the simulator process at the pinned checkout and LIBERO config:

```bash
export LIBERO_REPO_ROOT=/absolute/path/to/LIBERO
export LIBERO_CONFIG_PATH=/absolute/path/to/.libero
export PYTHONPATH="$PWD/src:$LIBERO_REPO_ROOT"
```

Run the outcome-free reset preflight with the Python executable from that
legacy environment:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --validate-only
```

The preflight checks both the scored state (`0`) and the excluded canary state
(`49`), their exact RGB/state hashes, runtime identity, asset hashes, and
relative controller mode. Before the live-model canary exists, the expected
ledger status is `BLOCKED_PENDING_MODEL_CANARY`.

After the GPU server is reachable through `http://127.0.0.1:8003`, run the
single-use outcome-free canary:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --run-model-canary --allow-model-canary
```

The canary calls the real checkpoint once on excluded state 49, validates one
finite `10×7` action chunk, applies zero simulator actions, and creates no
training label. Only a validated canary unlocks formal row `0`.

## 4. Formal E1 execution boundary

Formal execution additionally requires:

- a clean Git working tree;
- runner source bytes exactly matching the frozen plan;
- the frozen model-server identity and live runtime attestation;
- a previously validated canary;
- the next unused schedule index, with no skip or rerun; and
- the explicit `--allow-execution` flag.

Example for the first row:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --execution-index 0 --allow-execution
```

The canonical result root is `runs/e1/`. If a transport, simulator, identity,
or other infrastructure exception occurs, the started row is sealed as an
infrastructure failure and v1 cannot continue. Fixes require a new prospective
execution version; never delete the failure and rerun the same row.

## 5. What has and has not run

As of 2026-09-08:

- core and integration-contract tests pass locally;
- the real pinned LIBERO reset preflight passes for states 0 and 49;
- the MolmoAct2 snapshot identity is frozen in the plan;
- the live MolmoAct2 canary has not yet completed;
- no E1a scored branch has run (`0/6`); and
- no empirical Stage-1 training or closed-loop method evaluation has run.

Do not infer robot performance from a successful package install, unit test,
HTTP test double, checkpoint inspection, or reset preflight.
