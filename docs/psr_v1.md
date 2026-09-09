# PSR-VLA V1: Predictive State Conditioning for Interactive Perception

PSR-VLA V1 is the sole research path in this repository. It uses
open-vocabulary intentions generated inside MolmoAct2 rather than a fixed
hand-enumerated action vocabulary.

This document is an implementation and reproduction contract. It does **not**
report a trained model, a calibrated probability estimate, a successful
LIBERO rollout, or an RSS result. Software tests, synthetic interface canaries,
real-checkpoint integration, training, and scientific evaluation are separate
levels of evidence.

## Research question

Under partial observability, a robot may possess the required motor skill but
lack enough task-relevant evidence to choose the right next behavior. PSR-VLA
V1 asks:

> Given only the user's task, current and finite past public observations, and
> the remaining budget, which open-vocabulary intention should the robot
> execute now to minimize failure at the end of the original task horizon?

The scope is deliberately narrower than a full POMDP solution. V1 learns a
finite predictive state from a bounded history and evaluates a finite set of
language intentions. It does not claim that six tokens are a sufficient
statistic, that the proposal set covers every useful physical action, or that
the resulting controller is globally optimal.

## One-sentence method

Insert six trainable readout positions into MolmoAct2's native multimodal
sequence, let independently generated open-vocabulary intentions query their
contextualized states to predict real post-interaction visual evidence and
terminal task failure, choose the lowest predicted failure, and condition the
unchanged Action Expert topology with the selected intention's native
per-layer VLM key/value states.

## What the symbols mean

V1 uses five method symbols; no attention statistic or verbal confidence is
renamed as uncertainty.

| Symbol | Meaning | Produced from | Used by |
| --- | --- | --- | --- |
| `H` | Public task-conditioned history | Task, current dual RGB, public 8-D robot state, at most two earlier high-level boundary observations, executed-intent records, remaining budget | Native MolmoAct2 VLM |
| `B` | Six contextualized predictive-state positions | A causal VLM forward over `H` and six trainable soft input positions | Intention-conditioned `E/C` predictor and conditioned AE prefix |
| `U` | One short, open-vocabulary robot intention and its execution route | Two independent MolmoAct2 language samples from the same `[H,B]` prefix, plus one exact native option | Predictor query and physical execution |
| `E` | Real future visual evidence target | Frozen native visual patch features from the actual dual-RGB observation after the intention window | Auxiliary probabilistic supervision |
| `C` | Terminal failure indicator | Actual result after the selected intention window and a fixed native continuation to the original budget | Bernoulli supervision and online candidate ranking |

`C=0` means task success and `C=1` means task failure. The deployed score is
`P(C=1 | B,U,route)`. `E` helps train a predictive representation; V1
does not roll predicted images forward online and does not feed predicted `E`
back into history.

## End-to-end system

```text
                          PUBLIC POLICY PATH

 task + current agent RGB + current wrist RGB + public 8-D state
       + two prior decision-boundary RGB bundles + executed U + budget
                                      |
                                      v
                native MolmoAct2 embedding / causal VLM
                    [ H | B1 ... B6 | "Next action:" ]
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
       two independent language samples       exact native task option
             U1, U2 (conditioned)                  U0 (native)
                    +-----------------+-----------------+
                                      |
                    ordinary U token IDs + route only
                                      |
                                      v
             intent-only encoder -> Q -> 2 x cross-attention(B)
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
             q_E(E | B,U,route)          q_C(C | B,U,route)
                4-component GMM              failure probability
                                                   |
                                                   v
                           stable minimum predicted failure
                                                   |
                    +------------------------------+------------------+
                    |                                                 |
                    v                                                 v
       conditioned route: [H,B,U,trigger]                 native route: original task
       -> native per-layer VLM K/V -> AE                   -> original predict_action
                    +------------------------------+------------------+
                                                   |
                                                   v
                              <=10 real 7-D control actions
                                                   |
                                                   v
                           real dual-RGB/state reobservation -> new H


                  PRIVATE COLLECTION / LABEL PATH ONLY

 same-reset simulator snapshot -----> branch restoration and audit
 real observation after U window ---> frozen target encoder ----------> E
 evaluator at original horizon -----> success/failure ----------------> C
```

The lower path never enters candidate generation, prediction, selection, or AE
conditioning. It exists only in the collector, artifact store, evaluator, and
loss computation.

## Exact public input contract

At a high-level decision boundary, `PublicHistory` contains:

```text
task: non-empty user task string
current:
  agentview_rgb: content-addressed RGB reference
  wrist_rgb:     content-addressed RGB reference
  robot_state:   exactly 8 finite values
  control_step:  non-negative global episode step
previous: at most two unique, chronological observations older than current
executed: chronological, non-overlapping records of real executed intentions
remaining_control_steps: non-negative remainder of the original 300-step budget
```

The 8-D LIBERO state is end-effector position `(3)`, axis-angle orientation
`(3)`, and gripper joint position `(2)`. Remaining budget is represented
separately; it is not smuggled into those eight values.

RGB paths and scene names are not model features. Images are resolved through
content-addressed references and checked against their dimensions and SHA-256
digests. The existing LIBERO adapter remains responsible for the repository's
single 180-degree RGB orientation conversion; PSR does not rotate again.

The fixed history rule is:

- retain only real observations from the two most recent high-level decision
  boundaries;
- do not duplicate the current observation;
- preserve native patch order for each historical camera;
- add learned camera, temporal-age, and history-marker embeddings, followed by
  a trainable projection; and
- represent executed intention, route, interval, public execution status, and
  remaining budget with a short fixed text format.

No previous `B` and no predicted future are treated as observations.

## Token and tensor path

### 1. Native multimodal prefix and `B`

The backend first calls the checkpoint's official processor, robot-statistics
normalizer, visual merge, and embedding helpers for the current agent/wrist
images, task, setup/control tokens, and discrete state. Historical RGB uses the
same frozen visual encoder. Before the native assistant boundary it inserts the
public history representation; after that boundary and before the action
trigger it inserts:

```text
[B1 B2 B3 B4 B5 B6] [Describe the next robot action ... Next action:]
```

`b_embed` has shape `[6,D]` and is a trainable input parameter, but `B` denotes
the six **output hidden states** at those positions after the causal VLM
forward:

```text
B = VLM_theta([H; b_embed; format])[:, b_positions, :]  in R^(N x 6 x D)
```

Thus a fixed soft seed does not imply a fixed state: `B` changes with task,
images, robot state, history, and budget. The implementation builds official
visual embeddings first and then calls the backbone with `inputs_embeds`; it
does not pass images and `inputs_embeds` simultaneously. Real masks, position
IDs, token types, and per-sample `b_positions` are extended together.

The trainable execution-side deltas are confined to the last eight VLM
self-attention blocks (the checkpoint's fused `att_proj` and `attn_out`
linears), the history adapter, `b_embed`, and the Action Expert's native
`context_k_proj`/`context_v_proj`. All base parameters remain frozen and every
delta can be disabled atomically.

### 2. Open-vocabulary intentions `U`

The language head independently samples up to two short conditioned intentions
from the same `[H,B,format]` prefix. Sampling is content-seeded, capped at 32
tokens and four attempts, stops on registered boundaries, excludes native
action/depth control tokens, normalizes text, and removes exact duplicates.
The second sample never reads the first sample.

One native candidate is always added. Its text is the original task and its
route is `native`. It is not the original task passed through `B`: physical
native execution disables every PSR token and adapter and calls the original
MolmoAct2 `predict_action` path.

Candidate identity is a stable hash of normalized text and route for audit and
tie-breaking. That hash is never a predictor feature. The prediction inputs are
ordinary intention token IDs and a two-valued **interface route** (`native` or
`conditioned`), not a closed action-primitive label.

### 3. Intention-only query `Q`

For `N` histories, at most `M=3` candidates, intention length `L<=32`, native
VLM width `D`, readout width `d=512`, `K=4` mixture components, `J` native
visual positions, and evidence width `P<=128`, the core tensors are:

| Tensor | Shape | Information boundary |
| --- | --- | --- |
| contextual state `B` | `[N,6,D]` | Public `H` only |
| intention IDs / mask | `[N,M,L]` | Plain `U` text only |
| route IDs / candidate mask | `[N,M]` | Native vs conditioned; padding validity |
| intention query `Q` | `[N,M,d]` | Frozen word embeddings + learned projection, CLS, position, route, one Transformer encoder |
| projected state | `[N,6,d]` | Learned projection of `B` |
| readout `R` | `[N,M,d]` | Two cross-attention/FFN blocks, each candidate querying `B` independently |
| failure logit | `[N,M]` | Bernoulli `C` parameter |
| mixture logits | `[N,M,K]` | One global future-scene component per candidate |
| evidence mean / log std | `[N,M,K,J,P]` | Future visual-feature distribution |

The frozen native word embedding table is detached before a learned
projection. A CLS position, positional embeddings, and route embedding enter a
one-layer, eight-head Transformer encoder. This preserves word order; no mean
pooling is used. Crucially, this encoder cannot accept `H`, contextual hidden
states from language generation, AE caches, or future observations.

Each `Q_i` is flattened into the batch dimension before cross-attention, so
candidate intentions never self-attend to one another. Two residual
cross-attention/LayerNorm/FFN blocks let it read the projected `B` positions:

```text
Q_i = IntentEncoder(token_ids(U_i), route_i)
R_i = CrossReadout_2(Q_i, Project(B))
```

### 4. Predictive outputs `E` and `C`

The failure head is an MLP:

```text
p_i = P(C=1 | B,U_i,route_i) = sigmoid(f_C(R_i))
```

The evidence decoder uses learned native-position queries plus camera
embeddings and attends to `[R_i; Project(B)]`. It produces a diagonal Gaussian
mixture:

```text
q_E(E | B,U_i,route_i)
  = sum_(k=1)^4 pi_(i,k) Normal(E; mu_(i,k), diag(sigma_(i,k)^2)).
```

There is one mixture-component index for the entire future patch tensor, not
an independently selected component per patch. Log standard deviations are
clamped to `[-5,2]`; likelihood is accumulated in fp32 over valid dimensions,
then normalized by their count.

The target is computed only from the actual post-window observation:

```text
E_target = Normalize_train(
             OrthogonalProject_seed17(
               FrozenNativeVisualPatches(real RGB at t + window)))
```

Projection width is `min(native_visual_dim,128)`. Projection and training-split
channel statistics are stored with the checkpoint. Patch order and camera
ownership come from native processor metadata; flattened crop tokens are not
invented into a square grid. A paired-evidence separability diagnostic checks
whether this frozen target retains a decisive visual difference before more
predictor capacity is blamed.

### 5. Selection and AE conditioning

After optional held-out temperature scaling, selection is:

```text
i* = argmin_i sigmoid(f_C(R_i) / T).
```

Ties prefer the exact native route and then stable content identity. No
information-gain bonus, action-length weight, semantic-confidence term, or
manually chosen uncertainty threshold is added.

For a conditioned candidate, the backend teacher-forces the selected ordinary
`U` tokens and native action trigger after `[H,B,format]`, obtains the
checkpoint's **per-layer** VLM key/value states using `_extract_kv_states`,
builds the native encoder mask, and calls
`generate_actions_from_inputs(..., encoder_kv_states=...,
encoder_attention_mask=...)`. Soft positions use metadata-only placeholder IDs
only after embeddings have already been built; those IDs are not re-embedded
as the content of `B`.

The conditioned sequence is therefore:

```text
[H | B | format | selected U | native action trigger]
                         -> native KV at every VLM layer -> Action Expert
```

The implementation does not duplicate final-layer `B` into every AE layer.
The Action Expert receives the actual native layer-wise context. For native
execution it instead disables all additions and invokes the original
checkpoint path. Both routes use checkpoint-discovered action horizon and
dimension, `norm_tag=libero`, official state/action normalization, continuous
flow matching, and 7-D LIBERO actions.

## Runtime semantics: one 300-step clock, 50-step intentions, 10-step chunks

V1 freezes three different time scales:

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Episode budget | 300 control steps | The single original task horizon; it is never reset after an interaction |
| Intention window | at most 50 steps | How long the chosen high-level `U` remains fixed before candidate regeneration |
| Action chunk | at most 10 steps | How many generated continuous controls are applied before real reobservation and replanning |

At each decision boundary, the policy encodes `H`, proposes candidates,
predicts `C`, and chooses one `U`. During the next `min(50,remaining)` steps,
the text and route of `U` remain fixed. After every 10 applied actions, the
robot receives real dual RGB and state, reconstructs `H/B`, and generates the
next chunk for that same intention. At the next 50-step boundary, it generates
a fresh candidate set.

Only actual applied actions and observations update the record. The runtime
contains no simulator success predicate. A missing predictor, identity
mismatch, invalid probability, empty action chunk, or incompatible checkpoint
fails explicitly; native mode may be run independently, but is not a silent
fallback whose outcome is counted as PSR.

## Same-reset result collection

Training `C` requires results generated by the controller version it is meant
to predict. Stage B therefore freezes a complete execution snapshot `S0` and
Stage C data is collected under that exact identity.

For one frozen public decision point:

1. Create the environment from a real reset reference and verify its public
   observation exactly matches the registered `PublicHistory`.
2. Capture the private simulator/RNG snapshot in the collector. It is never
   given to the policy.
3. Generate the candidate set once from public input and require exactly one
   true native option.
4. Either execute every candidate from the identical snapshot or sample one
   with a public, recorded, content-seeded uniform rule.
5. Execute the chosen intention for `min(50,r)` real steps, reobserving every
   10 steps. Persist the actual boundary observation as the `E` source.
6. Continue with unmodified native MolmoAct2 `pi_c` until the remainder of the
   original 300-step budget is exhausted.
7. Use the private evaluator only after execution to assign final success and
   hence `C`.

The candidate-specific action seed depends on episode seed, decision step,
stable candidate identity, current control step, and phase—not list position.
Receipts are create-once, fsynced, hash-chained JSONL with a final byte-level
seal. A service or model crash is recorded as infrastructure failure with
`evidence_valid=false` and `cost_valid=false`; it is not silently removed and
not relabeled as task failure.

The scientific definition of `E` is the exact intention-window boundary. An
integration that terminates before that boundary must mark `E` missing rather
than duplicate a last frame; this is a required pre-experiment check for any
environment that exposes early public termination.

## Data firewall and split rules

Three record kinds are non-interchangeable:

| Record kind | Purpose | May train |
| --- | --- | --- |
| `warmup_demonstrations` | Real action, public intention, and real future observation for representation/action warmup | Stage A only |
| `snapshot_rollouts` | Actual candidate branches under frozen `S0` plus fixed continuation | Stage C and calibration, split permitting |
| `closed_loop_evaluation` | Full online policy episodes with repeated high-level decisions | Evaluation only |

`PSRBranchRecord.model_input()` returns only public history and candidate text
token IDs/routes. `supervision()` separately returns `E` and `C` for the sole
executed candidate. Alternative candidates never inherit a counterfactual
label.

The following are private/audit-only and cannot enter forward inputs:

- reset states or simulator snapshots;
- hidden object properties, semantic/instance IDs, poses, and predicates;
- reward, success, target location, oracle choice, or evaluator output;
- future RGB/features or final outcome;
- sample path, scene ID, task-file name, split, group, candidate hash, or other
  metadata that may encode the hidden world.

Datasets reject mixed record kinds, duplicate records/branches, incompatible
`S0`/continuation/target-encoder identities, old Method-V1 protocol markers,
and any episode, decision group, or reset family crossing data splits. All
branches and time slices from a correlated reset family must remain in one of
`train`, `validation`, `calibration`, or `test`.

## Losses and four-stage lifecycle

### Stage A — predictive/action warmup

Stage A jointly trains the small predictive-state additions and their native
execution interface using real warmup demonstrations:

```text
L_A = L_native_flow + L_future_evidence_NLL + L_intent_CE.
```

- `L_native_flow` uses the pinned MolmoAct2 convention
  `x_t=(1-t)noise+t*action`, velocity target `action-noise`, native beta time
  sampling, action padding, and valid-dimension masks. It calls the Action
  Expert's differentiable velocity forward—not `predict_action`, HTTP, or a
  `no_grad` inference path—so gradients can reach `B`, upper-VLM deltas, and
  context-K/V deltas.
- `L_future_evidence_NLL` is the global-component four-Gaussian NLL over the
  actual future frozen features.
- `L_intent_CE` teacher-forces a short intention supported by the public
  decision-time history and computes CE only on its intention tokens.

The base visual encoder, base word embeddings, original VLM weights, original
Action Expert, and original context projections remain frozen. Intention
labels must record whether they were human-authored or produced by an offline
teacher; a teacher may not inspect future results or private state. Falling
back to the episode task is explicitly only an interface warmup.

### Stage B — freeze the physical execution snapshot `S0`

After warmup, freeze and fingerprint everything that can affect proposals or
physical actions:

```text
S0 = base/processor revisions + visual/history encoding + B + VLM deltas
     + AE context deltas + AE + templates + candidate decoding
     + history rule + normalization + 300/50/10 schedule.
```

The snapshot stores parameter and persistent-buffer hashes plus protocol,
configuration, split, projection, normalization, and seed identities. Once
`S0` is frozen, Stage C verifies it before and after training and requires its
parameters to be disjoint from the predictor optimizer.

### Stage C — fit actual outcomes under frozen `S0`

With candidates and labels collected by the exact frozen execution model:

```text
L_C = L_actual_future_evidence_NLL + L_actual_failure_BCE.
```

Only real executed candidates with valid labels contribute. Missing `E` and
`C` are masked independently. If a batch has neither, the optimizer step is
skipped completely so weight decay or momentum cannot update an unsupervised
model. Stage-A evidence weights may initialize Stage C, but demonstration
outcomes cannot be relabeled as `S0` rollout outcomes.

### Stage D — held-out calibration and locked evaluation

Fit one positive scalar temperature on a separate calibration split by
minimizing Bernoulli BCE. Calibration refuses single-class data rather than
inventing a temperature. Because positive temperature scaling preserves
candidate order, it may improve probability quality but cannot by itself
improve argmin choice.

Lock the calibrated predictor before test evaluation. Changing `B`, VLM/KV
deltas, proposal decoding, execution duration, normalization, or `pi_c`
creates a new execution version and requires new collection/calibration.

## Evaluation contract

Report probability, finite-set selection, and physical execution separately:

- failure BCE/NLL, Brier score, fixed-bin reliability/ECE, including label
  support;
- within-same-reset pairwise ranking accuracy, selected failure, candidate
  oracle failure, oracle gap, native selection rate, and selected-candidate
  absolute error;
- planned/completed episodes, infrastructure-failure rate, task success over
  completed and all planned episodes, applied steps, and elapsed time;
- held-out `E` NLL and paired target separability;
- proposal acceptance/duplication and candidate count; and
- paired intervals by episode/reset family, not by treating correlated chunks
  as independent samples.

Required comparisons are native MolmoAct2, uniform selection over the same
`S0` candidates, and an offline same-reset rollout oracle. Training ablations need independent
checkpoints; `cost-only` and `no-history` alter `S0` and therefore require new
rollouts. Intent-only/full-context predictors and fixed-`S0` `B` interventions
may reuse the same branch data when their information boundary permits it.

## Configuration

The frozen engineering defaults are in
[`experiments/psr_v1.yaml`](../experiments/psr_v1.yaml). The loader accepts
YAML or JSON but requires exact V1 semantics and computes a canonical
fingerprint. The important defaults are:

- MolmoAct2-LIBERO revision
  `0d24a92bd1faf321ef497c3bbd5681af97c65aa2` and upstream code revision
  `66b87e64efd99dfd103241418113955cf64dfa9c`;
- 6 state positions, 2 previous observation bundles;
- 2 sampled intentions plus 1 native candidate;
- predictor width 512, 8 heads, 1 intention encoder layer, 2 cross-readout
  blocks;
- 4 evidence mixture components, requested projection width 128, seed 17;
- 300/50/10 execution and 10 flow steps;
- upper 8 VLM attention blocks and native AE context K/V low-rank deltas,
  rank 16 and alpha 32; and
- bf16 training, microbatch 1, effective batch 16.

Native hidden width, layer/KV-head count, action horizon, and action dimension
are discovered from the real checkpoint. Preflight rejects incompatible values
instead of silently reshaping them.

## Reproduction interface

Install the repository for weight-free checks and inspect the command surface:

```bash
uv sync --extra dev
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run ip-psr --help
```

CPU tensor/loss tests additionally use `--extra learned`. The real pinned
MolmoAct2 canary uses the Linux/NVIDIA environment and
`uv sync --extra dev --extra molmoact2`; the latter includes the exact Torch,
Transformers, image, and tokenizer dependencies and must not be inferred from a
successful weight-free preflight.

The V1 command contract is:

```bash
# Weight-free: validates configuration and dependency availability only.
uv run ip-psr preflight \
  --config experiments/psr_v1.yaml --device cpu

# Real canary: loads the pinned checkpoint and exercises B, native per-layer
# KV, AE input, native bypass parity, language generation, and gradients.
uv run ip-psr preflight \
  --config experiments/psr_v1.yaml --device cuda --load-model

# These paths must contain real user-provided data/assets. The commands do not
# synthesize labels, checkpoints, reset states, or success results.
uv run ip-psr import-data \
  --source /ABSOLUTE/PATH/TO/REAL_DATA \
  --output /ABSOLUTE/PATH/TO/WARMUP_DATA

uv run ip-psr train-warmup \
  --config experiments/psr_v1.yaml \
  --data /ABSOLUTE/PATH/TO/WARMUP_DATA \
  --output /ABSOLUTE/PATH/TO/S0

uv run ip-psr collect \
  --config experiments/psr_v1.yaml \
  --snapshot /ABSOLUTE/PATH/TO/S0 \
  --plan /ABSOLUTE/PATH/TO/COLLECTION_PLAN.json \
  --output /ABSOLUTE/PATH/TO/SNAPSHOT_ROLLOUTS

uv run ip-psr train-outcomes \
  --config experiments/psr_v1.yaml \
  --snapshot /ABSOLUTE/PATH/TO/S0 \
  --data /ABSOLUTE/PATH/TO/SNAPSHOT_ROLLOUTS \
  --output /ABSOLUTE/PATH/TO/PREDICTOR

uv run ip-psr calibrate \
  --snapshot /ABSOLUTE/PATH/TO/S0 \
  --predictor /ABSOLUTE/PATH/TO/PREDICTOR \
  --data /ABSOLUTE/PATH/TO/CALIBRATION_SPLIT \
  --output /ABSOLUTE/PATH/TO/CALIBRATED_PREDICTOR

uv run ip-psr evaluate \
  --config experiments/psr_v1.yaml --mode native \
  --plan /ABSOLUTE/PATH/TO/EVAL_PLAN.json \
  --output /ABSOLUTE/PATH/TO/NATIVE_EVAL

uv run ip-psr evaluate \
  --config experiments/psr_v1.yaml --mode psr \
  --snapshot /ABSOLUTE/PATH/TO/S0 \
  --predictor /ABSOLUTE/PATH/TO/CALIBRATED_PREDICTOR \
  --plan /ABSOLUTE/PATH/TO/EVAL_PLAN.json \
  --output /ABSOLUTE/PATH/TO/PSR_EVAL
```

This is the frozen V1 CLI contract. The release integrator must keep
`psr/cli.py`, `pyproject.toml`, examples, and `--help` text synchronized with
these commands. A CPU preflight must not download model weights or set any of
the real-model/training/evaluation statuses below.

## Implementation map

| Responsibility | Source |
| --- | --- |
| Frozen config and discovered architecture checks | [`psr/config.py`](../src/grounded_interaction/psr/config.py) |
| Public history, RGB references, open intentions | [`psr/types.py`](../src/grounded_interaction/psr/types.py) |
| Native token insertion, proposal generation, layer-wise KV, AE and flow loss | [`psr/molmo_backend.py`](../src/grounded_interaction/psr/molmo_backend.py) |
| Intent-only query, `B` readout, `E/C` heads and losses | [`psr/model.py`](../src/grounded_interaction/psr/model.py) |
| Strict record schema, split/identity checks, public/label separation | [`psr/data.py`](../src/grounded_interaction/psr/data.py) |
| Stage A/C losses, `S0` freeze, calibration and checkpoint identity | [`psr/training.py`](../src/grounded_interaction/psr/training.py) |
| Same-reset real branch execution and immutable receipts | [`psr/collection.py`](../src/grounded_interaction/psr/collection.py) |
| 300/50/10 online policy loop | [`psr/runtime.py`](../src/grounded_interaction/psr/runtime.py) |
| LIBERO public observation/frame-store adapter | [`psr/libero.py`](../src/grounded_interaction/psr/libero.py) |
| Probability, ranking, bootstrap and execution metrics | [`psr/evaluation.py`](../src/grounded_interaction/psr/evaluation.py) |
| Weight-free and real-checkpoint canaries | [`psr/preflight.py`](../src/grounded_interaction/psr/preflight.py) |

## Evidence status

The four statuses are intentionally non-substitutable:

| Level | Current V1 status | What would change it |
| --- | --- | --- |
| `IMPLEMENTED` | **Implemented at the source/contract level in the PSR V1 path; release requires the freezing commit's full CPU suite and CLI checks to pass.** | Reviewed code, frozen config, executable command entry, and passing software-contract tests |
| `VERIFIED_WITH_REAL_MODEL` | **No.** | The pinned MolmoAct2 GPU canary must pass native bypass parity, real token/KV/AE shape checks, cache/replay equivalence, nonzero finite gradients to all intended additions, natural-language proposal generation, and at least one real two-boundary LIBERO trace |
| `TRAINED` | **No.** | A real Stage-A checkpoint, frozen `S0`, real `S0` branch data, Stage-C predictor, and held-out calibration artifact with identities must exist |
| `EVALUATED` | **No.** | A locked checkpoint must be run on untouched evaluation groups with the preregistered baselines, ablations, metrics, and intervals |

Local unit tests establish type, shape, loss, masking, split, lifecycle, and
timeline invariants. Test doubles and synthetic interface pixels/actions are
not evidence that MolmoAct2 loaded, that the Action Expert understood an
intention, that the model trained, or that task success improved.

No empirical success rate, learned checkpoint, calibration result, LIBERO PSR
rollout, or paper table is claimed by this revision.

## Real-model and scientific gates

Before collecting formal results, run and retain evidence for:

1. **Native parity.** With every PSR addition disabled, the native option must
   numerically match the repository's original MolmoAct2 path under identical
   pixels, task, state, seed, and solver settings.
2. **Token/KV connection.** Real weights must produce the registered `B`
   positions and all native VLM KV layers; AE masks must expose exactly the
   valid `[H,B,U,trigger]` span.
3. **Differentiability.** Real flow loss must give finite nonzero gradients to
   `b_embed`, history projection, intended upper-VLM deltas, and AE context-K/V
   deltas, and no gradients to frozen parameters.
4. **Proposal ability.** On public task inputs, MolmoAct2 must generate useful
   ordinary short intentions without a hand template library. Failures and
   duplicate coverage must be reported.
5. **Native continuation ability.** After a decisive interaction reveals the
   needed evidence, fixed `pi_c` must be able to use it; otherwise `C` may
   correctly learn to reject exploration even when the information action was
   physically useful.
6. **Candidate oracle headroom.** Same-reset real branches must show that the
   generated set sometimes contains a better executable candidate. If not,
   improve proposal/grounding before tuning the failure head.

## Known limits and risks

- **Proposal gap:** online selection is limited to the two accepted generated
  intentions and exact native option. Candidate-internal regret bounds do not
  cover useful intentions that were never proposed.
- **Language grounding:** an open phrase may be syntactically valid but still
  be ambiguous or not executable by the frozen Action Expert.
- **Target sufficiency:** the fixed random projection of native visual patches
  may discard small text, identity, or relational evidence. The separability
  diagnostic is necessary before interpreting `E` NLL.
- **Continuation bottleneck:** `C` measures `U` followed by fixed native
  continuation. If that continuation cannot exploit newly revealed evidence,
  the learned value of the interaction is necessarily low.
- **Execution-version coverage:** calibration applies only to the exact frozen
  `S0`, continuation, task family, candidate behavior, decision-time support,
  and remaining-budget range represented by real collection.
- **Finite memory:** evidence older than two high-level boundaries is absent
  unless summarized in the public executed-intent text. V1 is PSR-inspired,
  not a proof of a sufficient classical predictive state.
- **Partial observability and confounding:** supervised outcome prediction does
  not automatically identify unobserved causal factors. Same-reset branching
  and grouped splits reduce particular confounds; they do not remove all OOD
  uncertainty.
- **Discrete selection:** intention sampling and argmin are not differentiable
  through physical execution. V1 uses imitation/intent/future-evidence losses
  and supervised real outcomes, not a false claim of ordinary end-to-end
  backpropagation through the simulator.

## References that constrain the implementation

- MolmoAct2: native VLM-to-Action-Expert token/KV topology, continuous
  flow-matching controller, and checkpoint-specific normalization.
- Littman, Sutton, and Singh, *Predictive Representations of State* (NeurIPS
  2001): predictive tests as state representations; V1 does not claim the
  classical sufficiency theorem for its finite neural tokens.
- Boots, Siddiqi, and Gordon, *Closing the Learning-Planning Loop with
  Predictive State Representations* (RSS 2010): connection between learned
  predictive state and control.
- Zhang et al., *Energy-based Predictive Representations for Partially
  Observed Reinforcement Learning* (UAI 2023): future-predictive objectives
  under partial observability; it does not validate V1's chosen decoder.
- Tennenholtz, Mannor, and Shalit, *Off-Policy Evaluation in Partially
  Observable Environments* (AAAI 2020): why partial-observation behavior logs
  require care about support and hidden confounding.

The implementation specification fixes the exact upstream revisions in
`experiments/psr_v1.yaml`; paper citations motivate the design but do not
replace real integration and held-out evidence.
