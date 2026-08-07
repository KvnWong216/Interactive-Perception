# Environments

This experiment runs in **two** Python environments, and the split is not
accidental — keeping them apart is what lets the client be pinned to LIBERO's
old MuJoCo stack while the policy server runs modern JAX on the GPU.

| Environment | Manager | Holds | Used by |
|---|---|---|---|
| `ipu` | conda, at `../.conda/envs/ipu` | LIBERO / robosuite 1.4.0 / MuJoCo 2.3.7, torch, the analysis code | every script in `scripts/` |
| `openpi/.venv` | `uv`, created by openpi's own install | JAX, the flow-matching action expert, the `pi05_libero` checkpoint loader | the policy server only |

They communicate over a websocket on port 8000, so they never need to agree on
a dependency. `scripts/run_full_experiment.sh` activates `ipu` and launches the
server through `uv run` inside openpi's checkout; nothing else is required.

## Recreating `ipu`

```bash
conda env create -p "$WORKSPACE/.conda/envs/ipu" -f env/environment.yml
conda activate "$WORKSPACE/.conda/envs/ipu"

# torch is a CUDA build and does not come from PyPI's default index
pip install torch==2.13.0+cu129 --index-url https://download.pytorch.org/whl/cu129

pip install -r env/requirements-lock.txt
pip install -e .
```

`environment.yml` is exported with `--no-builds` so it resolves across driver
versions; `requirements-lock.txt` is the exact pip state of the run that
produced the results in the README, and is the file to use when a number needs
to be reproduced precisely.

## LIBERO is vendored, not installed

`import libero` fails in a bare interpreter and that is expected. LIBERO lives
in `third_party/LIBERO` and is put on `sys.path` by `scripts/_bootstrap.py`,
which every entry point imports. Pinning it as a package would make the scene
`.bddl` files and the installed Python drift apart, which is exactly the class
of bug the reproduction gate exists to catch.

## Recreating the policy server

Follow openpi's own instructions in `../openpi`; it manages `.venv` with `uv`.
The checkpoint is fetched separately:

```bash
python scripts/download_pi05_libero.py --dest "$WORKSPACE/checkpoints"
```

## Why this is one conda environment

All analysis, scene validation, rollout and figure code shares a single conda
environment on purpose, so the whole thing can be removed with one
`conda env remove -p .conda/envs/ipu`. The openpi venv is upstream's own and is
deleted with the openpi checkout.
