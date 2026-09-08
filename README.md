# Interactive Perception

Prompt-conditioned prediction and selection of grounded physical interventions,
with a separately frozen VLA as the low-level executor.

> **Status — 2026-09-08.** The formal-v1 software path and the first real
> execution-interface pilot (`E1a`) are implemented. The exact LIBERO reset,
> public RGB/state observations, frozen MolmoAct2 protocol, action application,
> evaluator trace, immutable result record, and training-target loader are now
> connected in code. Local software tests pass, and the real LIBERO reset
> preflight was reproduced locally for both frozen states.
> The released MolmoAct2 checkpoint has **not yet** completed the single-use
> model canary, so scored E1a execution remains `0/6`. There is no trained
> Stage-1 checkpoint, closed-loop method result, benchmark table, or real-robot
> result in this revision.

## Research question

A robot can possess useful manipulation primitives and still fail because the
current observation is insufficient for the user's request. The main question
is:

> Under a fixed executor and interaction budget, which grounded action should
> the robot execute now so that the requested task is most likely to succeed
> after the resulting real observation?

The intended two-stage system is:

```text
public prompt + RGB/history + public robot state
                         │
          grounded candidate interventions
                         │
                         ▼
 frozen VLM tokens → grounded outcome model → finite-set selection
                                                 │
                                      selected grounded subtask
                                                 │
                                                 ▼
                            frozen VLA → action chunk → environment
                                                 │
                                                 ▼
                                      real reobservation → repeat
```

Stage 1 predicts a preregistered bounded outcome for each complete intervention

```text
u = (primitive, referent, high-level parameters, current-frame grounding).
```

Stage 2 receives only the selected subtask, deployment RGB, and public robot
state. It generates continuous robot actions; it does not receive simulator
semantic IDs, hidden poses, predicates, rewards, or evaluator labels.

The token-level outcome model is a software implementation, not yet an
empirical result. Its internal tokens are predictive representations; they are
not called a calibrated belief or uncertainty distribution without held-out
calibration evidence.

## Why E1a comes first

Before training Stage 1, we must establish that a chosen physical referent can
actually survive the Stage-1-to-VLA interface. E1a therefore isolates one
narrow question:

> With two similar moka pots in one real LIBERO RGB observation, can three
> public referent interfaces make frozen MolmoAct2 first contact the intended
> instance?

The regions are manually annotated from public RGB. This is deliberately an
interface diagnostic—not autonomous grounding and not the final method. E1a
compares:

- coarse spatial language;
- normalized coordinates rendered in language; and
- a deterministic magenta box/cross rendered into the public agentview image.

The current preregistered pilot is only
`1 reset state × 2 referents × 3 interfaces = 6 single-use branches`. Its
primary label is whether the first contact with either candidate object is
exclusively the intended one within 300 simulator steps. Complete placement on
the stove is logged only as a diagnostic.

## Implemented real chain

```text
frozen LIBERO state
  → real agentview RGB + wrist RGB + 8-D public state
  → human public-RGB region + deterministic referent interface
  → official MolmoAct2 predict_action API over HTTP
  → exactly 10 × 7 finite continuous actions per model call
  → float32 conversion + official gripper binarization
  → relative-control LIBERO steps
  → per-step evaluator-only contact / grasp / predicate trace
  → first-contact outcome recomputed from the lowest-level trace
  → sealed ObservedBranch
  → typed training record
  → executed-candidate Bernoulli loss
```

Important guarantees:

- MolmoAct2 input is exactly two RGB images in `[agentview, wrist]` order, an
  8-D public state, text, and frozen inference settings.
- Every replan state is linked to the preceding action-chunk milestone.
- `actions_applied` stores the exact float32, gripper-binarized values passed to
  LIBERO—not an approximation of the raw model output.
- The outcome is assigned only to the candidate that was actually executed.
  Alternatives receive no fabricated counterfactual labels.
- Validation re-derives serializer output, request IDs, action prefixes,
  temporal state/frame links, first contact, diagnostics, receipt, and training
  label from the frozen plan and lowest-level traces.
- The default training loader rejects software test-double evidence.
- Learned tensor alignment uses candidate fingerprints, not semantic content in
  `candidate_id` strings.

The SHA-256 chain establishes byte integrity and internal reconstructibility.
It is not a digital signature and does not independently prove that a simulator
or HTTP server was honest. The current evaluator is structurally excluded from
policy payloads, but it still runs in the same process; this is not a
process-security boundary.

## Current evidence

| Asset | Current state | What it establishes |
| --- | --- | --- |
| Contracts, selector, serializer, runtime tests | Pass | Software invariants only |
| Token outcome forward/backward | Pass locally | Tensor/loss wiring only |
| E1 test-double full artifact → loss round trip | Pass | Result records are trainable, not robot performance |
| Real LIBERO state 0 preflight | Reproduced locally; no sealed outcome | Exact scored reset/RGB/state/controller can be reproduced |
| Real LIBERO state 49 preflight | Reproduced locally; no sealed outcome | Excluded canary reset/RGB/state can be reproduced |
| Frozen MolmoAct2 model canary | Pending | No real checkpoint action has yet been admitted |
| E1a scored branches | `0/6` executed | No empirical referent result yet |
| Stage-1 learned model / closed loop | Pending | No method claim yet |

The frozen plan is
[`experiments/e1_referent_ceiling/pilot_state0_v1.json`](experiments/e1_referent_ceiling/pilot_state0_v1.json).
Its digest binds the runner source, assets, public observations, model identity,
canary, execution order, and trial rows.

## Repository structure

```text
src/grounded_interaction/
  contracts.py          public contexts and grounded intervention contracts
  tokens.py             frozen-token and current-patch support contracts
  model.py              support-constrained candidate fusion + outcome scorer
  losses.py             executed-candidate Bernoulli objective
  selection.py          feasibility filter and finite-set argmax
  serialization.py      coarse / precise / marker referent serializers
  conditioning.py       deterministic public-RGB marker conditioning
  molmoact2.py           pinned official MolmoAct2 client/server adapter
  libero_runtime.py      exact-reset public observation and action runtime
  e1.py                  E1 preflight, canary, runner, ledger, validator
  e1_training.py         sealed live artifact → typed targets → loss
  rgb.py                 content-addressed decoded-RGB store
scripts/
  freeze_e1_plan.py      regenerate the still-unexecuted frozen E1a plan
experiments/e1_referent_ceiling/
  pilot_state0_v1.json   six-row single-use E1a pilot
tests/                   software and integration-contract verification
docs/                    formal method and E1 runbook
env/README.md            split local-LIBERO / GPU-model environments
```

The superseded pre-formal tree is recoverable from Git tag
`archive/pre-formal-v1-2026-09-07` at commit `8c4be631`.

## Software verification

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/KvnWong216/Interactive-Perception.git
cd Interactive-Perception
uv sync --extra dev --extra integration
uv run pytest -q
uv run ip-smoke --output runs/formal_v1_smoke.json
```

The smoke run uses scripted scores and replayed outcomes. It must report
`software_verification_only=true` and `empirical_evidence=false`.

On a supported PyTorch platform, the learned software checks can be installed
with `--extra learned`. The MolmoAct2 extra intentionally resolves official
CUDA 12.8 PyTorch wheels and belongs in the Linux GPU environment, not the
legacy macOS LIBERO environment.

## E1a reproduction

E1 uses two environments joined by a localhost HTTP boundary:

1. a Python 3.11 Linux/NVIDIA environment loads the frozen MolmoAct2
   checkpoint; and
2. the pinned legacy LIBERO environment runs simulation and evaluation.

First inspect the model snapshot without loading it onto GPU:

```bash
uv sync --extra molmoact2 --extra dev
uv run ip-serve-molmoact2 \
  --inspect-only \
  --checkpoint allenai/MolmoAct2-LIBERO \
  --revision 0d24a92bd1faf321ef497c3bbd5681af97c65aa2
```

The hashes and parsed normalization/action semantics must match the frozen plan.
Then, on the authorized GPU only:

```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run ip-serve-molmoact2 \
  --identity-json experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --host 127.0.0.1 --port 8003
```

With a remote GPU, tunnel its localhost port:

```bash
ssh -N -L 8003:127.0.0.1:8003 USER@GPU_HOST
```

From the LIBERO environment, perform the outcome-free preflight:

```bash
export LIBERO_REPO_ROOT=/absolute/path/to/LIBERO
export LIBERO_CONFIG_PATH=/absolute/path/to/.libero
export PYTHONPATH="$PWD/src:$LIBERO_REPO_ROOT"
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --validate-only
```

The preflight must reproduce both state 0 and excluded canary state 49 and
report `BLOCKED_PENDING_MODEL_CANARY`. That block is expected before the first
real model call.

After the repository is committed and clean, run exactly one outcome-free
canary. It calls the real checkpoint once, requires a finite `10 × 7` action
chunk, applies zero actions, and creates no training label:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --run-model-canary --allow-model-canary
```

Only a validated canary unlocks the canonical ledger. Execute one frozen row at
a time, in order, with no rerun or overwrite path:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --execution-index 0 --allow-execution
```

See [`docs/e1_referent_executor_ceiling.md`](docs/e1_referent_executor_ceiling.md)
for the exact contract, artifacts, outcome definition, and interpretation.

## Scope and limitations

In scope for the eventual method:

- interactive manipulation: `OPEN`, `REMOVE`;
- information enrichment: `ROTATE`, `BRING_CLOSE`;
- direct execution and safe `STOP`; and
- receding-horizon decisions from actual post-action observations.

Explicitly outside formal v1:

- active viewpoint change;
- Stage-1 continuous trajectory generation;
- VLA weight updates;
- a claim of solving a full POMDP; and
- hand-designed uncertainty factor sums or route labels.

E1a has additional limitations: one scored initial state, a checkpoint trained
on the LIBERO mixture that includes this task family, manual regions, a static
marker/coordinate reused after motion, and no confirmatory sample size. It can
show only whether the current executor interface is worth pursuing.

## Next scientific steps

1. Run the excluded-state real-model canary.
2. If it passes, execute the six E1a branches once in their frozen order.
3. If at least one referent interface controls intended first contact, freeze a
   new E1b design with a stock-instruction positive control, a truly ambiguous
   no-spatial control, multiple untouched reset states and seeds, balanced
   target sides, and reset-group-paired analysis.
4. Only after E1b validates the interface, integrate automatic public-RGB
   proposal/grounding and a concrete frozen-VLM token provider.
5. Collect reset-controlled outcomes and train/evaluate Stage 1.
6. Then test the full interactive-perception loop on hidden-content and
   information-enrichment scenarios.

No large outcome-model training or RSS main table should begin before Steps 1–3
establish that the frozen executor can act on the selected referent.

## Documentation

- [Formal-v1 method and data contract](docs/formal_pipeline_v1.md)
- [E1a execution-interface runbook](docs/e1_referent_executor_ceiling.md)
- [Architecture decision record](docs/adr/0002_grounded_intervention_outcome_planning.md)
- [Environment guide](env/README.md)
