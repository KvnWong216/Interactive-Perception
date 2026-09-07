# Environment guide

The current release has one reproducible environment: the CPU/PyTorch software
environment used by the formal-v1 contracts, model, tests, and synthetic smoke
run. LIBERO and a frozen VLA service are deliberately not presented as working
dependencies until their adapters and checkpoint identities are frozen.

## Core development environment

Requirements:

- macOS or Linux;
- Python 3.10 or newer; and
- [uv](https://docs.astral.sh/uv/).

Create the locked environment from the repository root:

```bash
uv sync --no-editable --extra learned --extra dev
```

The canonical check uses a wheel-style, non-editable install so it also tests
package discovery and the command-line entry point. Re-run `uv sync
--no-editable` after changing source files.

Run the focused verification:

```bash
uv run --no-editable pytest -q
uv run --no-editable ip-smoke --output runs/formal_v1_smoke.json
```

The learned extra installs PyTorch for pre-bound candidate/context fusion and
the outcome scorer. No CUDA device is required by the synthetic smoke test.

## LIBERO integration

The formal-v1 release does not yet ship a canonical LIBERO scene, runner,
simulator lock, or rollout result.

The next environment release will freeze, together:

1. the exact LIBERO and robosuite revisions;
2. MuJoCo and renderer versions;
3. the public-observation camera contract;
4. the frozen VLA client/server protocol; and
5. the executor checkpoint and decoding identity.

Until that release, use an existing LIBERO installation only for scene
development, not for a claimed formal-v1 result.

## Frozen VLM and VLA integrations

`grounded_interaction` exposes typed boundaries for frozen token providers and
frozen executors. A Protocol or replay double is not a model integration.
Concrete VLM and MolmoAct2 installation commands will be added only after the
adapters, model revisions, input preprocessing, and checkpoint hashes have been
implemented and tested.
