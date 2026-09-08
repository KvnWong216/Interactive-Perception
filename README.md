# Interactive Perception

Prompt-conditioned prediction and selection of grounded physical interventions,
with a separately frozen VLA as the low-level executor.

> **Status — 2026-09-09.** The formal-v1 execution-interface pilot (`E1a`) and
> the core Method-V1 branch-data/training path are implemented. Method V1 now has a
> concrete frozen Qwen2.5-VL provider, strict public-RGB candidate proposal,
> processor-derived patch grounding, the trainable grounded outcome scorer,
> exact-reset paired collection, fixed continuation, MolmoAct2/LIBERO runtime,
> checkpointed training, calibration, and evaluation code. The E1a model
> canary and all Method-V1 empirical rollouts remain unexecuted: E1a is `0/6`,
> there is no trained empirical Stage-1 checkpoint, and there is no closed-loop
> method, benchmark-table, or real-robot result in this revision. The required
> semantic information-masking intervention and the separate fresh held-out
> closed-loop policy-episode runner are still experimental work, not hidden
> behind the word “implemented.”

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
                            frozen VLA → action chunks → environment
                                                 │
                                                 ▼
                               real reobservation → fixed continuation
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

## Method V1 in one paragraph

Method V1 freezes Qwen2.5-VL-3B-Instruct to propose at most three `DIRECT` and
three `OPEN` interventions and to extract task/history/image tokens. A small
two-attention `GroundedOutcomeModel` predicts final task success within a
maximum 300-step horizon under one frozen continuation policy. `DIRECT` uses
the full 300 steps. `OPEN` uses 100 steps, obtains a real public
reobservation, asks the same frozen Qwen for up to three `DIRECT` candidates,
and executes its first valid candidate for 200 steps. If Qwen returns no valid
continuation, the episode stops and is evaluated after 100 steps rather than
being padded with unregistered actions. The learned scorer is not called
again. Training uses one Boolean final-task outcome only for the candidate
actually executed; alternatives remain unknown.

The exact token construction, equations, split firewall, service boundaries,
collection commands, training protocol, and limitations are documented in
[`docs/method_v1.md`](docs/method_v1.md). Defaults and immutable model
identities are in [`experiments/method_v1.yaml`](experiments/method_v1.yaml).

Qwen has two frozen, auditable responsibilities: it proposes grounded action
candidates from public inputs, and it exports detached context/candidate
features. It does **not** choose the action using an uncalibrated verbal
confidence. Its prompt does freeze an ordinal best-to-worst candidate order so
the `frozen_vlm` baseline and post-OPEN continuation are well defined. The
trainable outcome model assigns comparable success
probabilities to the frozen candidates; the selector performs the finite-set
choice; MolmoAct2 alone generates continuous actions. A deployment decision is
admitted only through a provenance artifact that binds the exact manifest,
physical feature-cache bytes, checkpoint identity, optional calibrator, all
candidate predictions, and the recomputed selected candidate.

The audit trail preserves Qwen's raw proposal and deterministic rejections,
candidate/patch overlays, and every MolmoAct2 chunk's actual public RGB, 8-D
state, request, seed, latency, and applied actions. These artifacts make the
software path inspectable. Before a branch can enter a training dataset, the
trace validator reopens the RGB files and recomputes state hashes, chunk seeds,
fixed budgets, 7-D finite action prefixes, request/receipt identities, context
transitions, and the DIRECT or OPEN-to-DIRECT topology. This still does not
turn an unexecuted rollout into empirical evidence.

Self-contained dataset JSON is diagnostic only. Before any outcome-bearing
run, an immutable global collection plan binds the exact decision-freeze
population, split counts, candidate-by-seed schedules, source configuration,
infrastructure-failure policy, and scorer-verifier key identity. Every
single-use execution claim binds that plan. Canonical training reopens the plan
and every outcome-free decision freeze, but opens only train/validation
`attempt.json` files; a calibration or test attempt is rejected before its
contents are read. Formal evaluation later reopens the complete held-out
evidence. Within each stage's admitted split scope, an omitted or substituted
whole group is rejected just like a missing schedule row. The split-scoped
receipt-admission digest is carried into the training checkpoint and prediction
artifact.

For learned physical selection, an isolated modern-PyTorch verifier replays
the bound scorer checkpoint over the bound physical Qwen cache. The legacy
LIBERO process authenticates fresh-nonce requests and responses with the
pre-registered HMAC-SHA256 key before claiming the branch, in addition to
checking service/replay source and runtime identities. This rejects an
unkeyed endpoint that merely returns self-consistent hashes and rejects
re-hashed hand-written labels or candidate scores. The replay result also
carries the checkpoint's training-dataset, receipt-admission and training-plan
digests into the selected-action artifact. It is a shared-secret process
boundary, not hardware attestation: compromise of the host or secret, or a
maliciously fabricated checkpoint plus identity sidecar, remains outside the
guarantee. Formal offline reporting separately reopens the receipt-backed
dataset and recomputes those training identities.

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

## Implemented E1a execution chain

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
| Qwen proposal / patch-map / frozen-cache software tests | Pass locally | Public candidate and token contracts only |
| Method-V1 data / continuation / training / selection / evaluation software tests | Pass locally | Code-path invariants only |
| Method-V1 real Qwen + MolmoAct2 + LIBERO branch | Not run | No empirical Method-V1 outcome yet |
| Semantic revealed-region masking intervention | Not implemented or run | Information use has not been isolated from physical affordance benefit |
| Fresh held-out closed-loop policy episode collector | Not implemented or run | Offline paired-branch estimates are not deployment evidence |
| Registered ablation launchers | Not implemented or run | Region, history, task-text and descriptor-binding contributions are not yet isolated |

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
  proposals.py           strict public-RGB Qwen candidate schema and grounding
  qwen_provider.py       pinned Qwen proposal/token runtime and feature cache
  qwen_service.py        isolated online Qwen proposal boundary
  scorer_verification_client.py
                         stdlib-only verifier client for legacy LIBERO
  scorer_verification_service.py
                         modern-PyTorch checkpoint/cache replay boundary
  method_v1_data.py      paired branch schedules and full-task outcome dataset
  continuation.py        fixed DIRECT / OPEN-then-DIRECT execution policy
  method_v1_runtime.py   real MolmoAct2/LIBERO fixed-budget adapter
  method_v1_trace.py     semantic replay validation of public execution evidence
  collect_outcomes.py    render, freeze, single-use execute, and finalize CLI
  train_outcomes.py      executed-only Monte Carlo outcome training
  selection_provenance.py
                         checkpoint-replayed, hash-bound pre-execution choice
  evaluate_policy.py     canonical checkpoint replay and evidence-scoped metrics
scripts/
  freeze_e1_plan.py      regenerate the still-unexecuted frozen E1a plan
experiments/e1_referent_ceiling/
  pilot_state0_v1.json   six-row single-use E1a pilot
experiments/method_v1.yaml
                         frozen Method-V1 defaults and model identities
experiments/method_v1/   prompt-neutral scene asset and reset-spec template
tests/                   software and integration-contract verification
docs/method_v1.md        Method-V1 inputs, tokens, execution, data, and commands
env/README.md            split LIBERO, Qwen, scorer, and MolmoAct2 environments
```

The superseded pre-formal tree is recoverable from Git tag
`archive/pre-formal-v1-2026-09-07` at commit `8c4be631`.

## Software verification

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/KvnWong216/Interactive-Perception.git
cd Interactive-Perception
uv sync --extra dev --extra integration
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run pytest -q
uv run ip-smoke --output runs/formal_v1_smoke.json
```

The explicit `PYTHONPATH` also avoids a macOS/Python 3.13 edge case where an
editable-install `.pth` file inside a hidden virtual environment can itself be
marked hidden and ignored by Python. It does not change any model input or
experiment identity.

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
4. Use the implemented Method-V1 provider and collector to run one real,
   non-scored integration branch through Qwen, MolmoAct2, and LIBERO.
5. Freeze scene/layout/hidden-family splits, then collect the complete
   candidate-by-two-seed reset-controlled branch matrix.
6. Train three independent outcome scorers and evaluate probability quality,
   offline branch-matrix baselines, and new held-out closed-loop episodes.
7. Only after the DIRECT/OPEN method is supported, extend a separately frozen
   version to REMOVE, ROTATE, and BRING_CLOSE scenarios.

No large outcome-model training or RSS main table should begin before Steps 1–3
establish that the frozen executor can act on the selected referent.

## Documentation

- [Formal-v1 method and data contract](docs/formal_pipeline_v1.md)
- [E1a execution-interface runbook](docs/e1_referent_executor_ceiling.md)
- [Architecture decision record](docs/adr/0002_grounded_intervention_outcome_planning.md)
- [Method-V1 implementation and reproduction guide](docs/method_v1.md)
- [Environment guide](env/README.md)
