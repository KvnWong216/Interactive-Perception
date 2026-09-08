# E1a: grounded referent → MolmoAct2 execution-interface pilot

## 1. What this experiment is

E1a is the first outcome-bearing experiment in the current repository. It asks
one deliberately narrow question:

> When two visually similar moka pots appear in the same public LIBERO RGB
> observation, can a manually supplied public image region be communicated to
> frozen MolmoAct2 strongly enough to control which instance the gripper first
> contacts?

This is an **executor-interface diagnostic**. It is not an evaluation of the
full interactive-perception method. In particular, E1a does not test:

- automatic proposal generation or grounding;
- a VLM token provider;
- the learned Stage-1 outcome model;
- candidate scoring or action selection;
- uncertainty calibration;
- an information-seeking closed loop; or
- generalization across scenes, reset states, objects, or tasks.

The two regions are manually annotated from the same public agent-view RGB
that a deployed vision module could observe. No simulator semantic ID is used
to construct the policy input. Simulator object names are retained only in a
private evaluator path so that the physical outcome can be scored.

An E1a branch counts as empirical only when the released
`allenai/MolmoAct2-LIBERO` checkpoint produced its action chunks, those chunks
were actually applied to the pinned LIBERO environment, and the artifact tree
passes the frozen byte and semantic validator. Unit tests, the faithful HTTP
test double, scripted scores, replay executors, and still images are software
evidence only.

## 2. Frozen design at a glance

The canonical plan is
[`experiments/e1_referent_ceiling/pilot_state0_v1.json`](../experiments/e1_referent_ceiling/pilot_state0_v1.json).
It freezes exactly:

```text
1 LIBERO reset state
  × 2 intended public regions
  × 3 referent-conditioning interfaces
  = 6 single-use execution branches
```

| Field | Frozen value |
| --- | --- |
| Plan | `e1-moka-state0-pilot-v1` |
| Scene | `KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove` |
| Scored init-state index | `0` |
| Outcome-free canary state | init-state index `49`, excluded from scoring |
| Public task | `Put the selected moka pot on the stove.` |
| Candidate primitive | `DIRECT` |
| High-level parameter | `destination = stove cook region` |
| Public candidates | left moka-pot region and center moka-pot region |
| Policy | frozen `allenai/MolmoAct2-LIBERO` |
| Episode horizon | exactly 300 applied simulator actions for every label-bearing branch |
| Replanning interval | one new model request after each 10-action chunk |
| Primary label | first candidate contact is exclusively the intended candidate |
| Full placement | diagnostic only |
| Execution order | `coarse-left`, `marker-center`, `precise-left`, `coarse-center`, `marker-left`, `precise-center` |

All six rows share the frozen model seed base `26090800`. Model-call `k` uses
`26090800 + k`. Every branch starts from the same reset state but is executed
in a fresh simulator instance and fresh model session. E1a has no confirmatory
sample size and must not be presented as a benchmark result.

## 3. Exact model and simulator contract

### 3.1 MolmoAct2 server

| Item | Frozen value |
| --- | --- |
| Checkpoint | `allenai/MolmoAct2-LIBERO` |
| Checkpoint revision | `0d24a92bd1faf321ef497c3bbd5681af97c65aa2` |
| Referenced upstream code revision | `66b87e64efd99dfd103241418113955cf64dfa9c` |
| Input images | `[agentview, wrist]`, both RGB, each `256×256` |
| Public robot state | 8-D: end-effector xyz, axis-angle orientation, two gripper positions |
| Inference API | official remote-code `predict_action` |
| Output | exactly `10×7` finite continuous actions per successful call |
| State/action normalization | `norm_tag="libero"`, `q01_q99` statistics |
| Action semantics | delta end-effector pose; gripper binarized before LIBERO step |
| Flow-matching steps | `10` |
| Language normalization | enabled |
| Depth reasoning | disabled |
| CUDA graph | disabled |
| Weight/autocast dtype | BF16 |
| Model-visible device | `cuda:0` after selecting the authorized physical GPU |
| Required runtime | PyTorch `2.11.0`, Transformers `4.57.6`, CUDA runtime `12.8` |

The plan additionally freezes hashes of `config.json`, `norm_stats.json`, and a
19-file, 21,781,401,448-byte inference snapshot. The server re-hashes and
parses these files before loading the model. Its `/health` response must attest
the same model, runtime, snapshot semantics, adapter source hash, dtype, and
device. A mismatch fails closed and produces no behavioral label.

The 16 GB RTX 4080 Super is an intended canary target, not an already-proven
memory guarantee. If the exact BF16 configuration does not fit, that is an
infrastructure finding. Do not silently quantize, resize, swap checkpoints,
change the solver, or relax identity checks under the v1 plan.

### 3.2 LIBERO process

The simulator side is pinned to:

- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- Python `3.10.20`;
- NumPy `1.22.4`;
- PyTorch `1.11.0`;
- robosuite `1.4.0`;
- MuJoCo `2.3.7`;
- the exact BDDL and pruned-init files and hashes stored in the plan;
- ten standard settling steps; and
- explicit relative/delta control.

Raw robosuite camera arrays are rotated 180 degrees according to the official
LIBERO/MolmoAct2 convention. The simulator process exposes only agent-view RGB,
wrist RGB, and the 8-D state to the model client. Object contacts, grasp state,
and predicates remain evaluator-only.

The GPU model and legacy LIBERO dependencies intentionally run in separate
Python environments and communicate through a localhost HTTP protocol. See
[`env/README.md`](../env/README.md).

## 4. Public input and evaluator-only data

### 4.1 Policy-visible data

For every inference call, the model receives only:

```text
conditioned agent-view RGB
+ current unmodified wrist RGB
+ current 8-D public robot state
+ condition-specific natural-language subtask
+ frozen inference controls
```

The initial candidate records bind a normalized public agent-view region:

| Public region | Opaque candidate ID | Box `(x0,y0,x1,y1)` | Center `(x,y)` |
| --- | --- | --- | --- |
| left | `candidate-5c89f4a2` | `(0.06, 0.43, 0.30, 0.69)` | `(0.18, 0.56)` |
| center | `candidate-9e31d6b7` | `(0.34, 0.51, 0.60, 0.78)` | `(0.47, 0.65)` |

The candidate IDs are intentionally opaque. They are audit keys, not semantic
prompts and not tensor-alignment keys. Candidate fingerprints bind primitive,
referent, parameters, source frame, box, point, and camera.

The region is fixed for the complete E1a branch. The runner does not track or
re-ground it after the objects move. That is a limitation of this interface
pilot, not a proposed final mechanism.

### 4.2 Evaluator-only data

For scoring, each public region is mapped privately to one simulator object
name, with the other moka pot treated as the distractor. The private path reads:

- gripper–candidate contacts at each applied action;
- whether the intended object is grasped; and
- whether `on(intended_object, flat_stove_1_cook_region)` is true.

Those fields never enter the HTTP request, serialized subtask, RGB marker,
public state, action history, or subsequent policy observation. Tests and the
artifact validator scan the public execution trace for evaluator object names.

This is a **structural information-flow boundary**, not an operating-system or
cryptographic isolation boundary: policy and evaluator currently execute in
the same LIBERO process. The distinction must be preserved in claims.

## 5. Three referent interfaces

The runner deterministically serializes one `DIRECT` candidate in one of three
modes.

### 5.1 `coarse_text`

The original RGB pair is unchanged. The candidate becomes a qualitative
location in text, for example:

```text
Complete this task: Put the selected moka pot on the stove.
Use moka pot, the instance near the left of the current agentview image.
Use these high-level parameters: destination: stove cook region.
```

The actual left and center regions yield different coarse strings in this
frozen pilot. Coarse language remains a low-bandwidth baseline and is not a
truly ambiguous no-location control.

### 5.2 `precise_text`

The original RGB pair is unchanged. Normalized point and box coordinates are
rendered to three decimal places in the referential phrase, including the
convention that x is measured from the left and y from the top. MolmoAct2 does
not receive a native coordinate tensor. This condition tests whether precise
public coordinates survive the text interface; it does not assume the model
was trained for this syntax.

### 5.3 `visual_marker`

The agent-view image receives a deterministic magenta rectangle and magenta
cross derived from the public box and center. The wrist image is unchanged.
The text refers to the instance enclosed by the rectangle and centered on the
cross.

The image contains **no `TARGET` word and no simulator identity**. This is a
public visual-prompt ceiling, not native spatial-token support and not learned
grounding.

## 6. Outcome-free gates

### 6.1 Reset preflight

Before any model action is admitted, `--validate-only` must reproduce:

- scored init state `0`, including reset-array, agent-view, wrist, and 8-D-state
  hashes;
- excluded canary state `49` with the same checks;
- the pinned runtime versions and BDDL/init-file hashes; and
- relative controller mode.

This loads LIBERO but calls no model, applies no scored action, and creates no
outcome.

### 6.2 Single-use live-model canary

The canary uses state `49` and the stock instruction
`Put the right moka pot on the stove.`. It performs exactly one real MolmoAct2
request. A pass requires:

- the exact frozen server identity and runtime attestation;
- exact frozen public observation hashes and state;
- the exact request hash; and
- one finite action chunk with shape `10×7`.

It applies **zero** actions, queries **zero** evaluator outcomes, and writes
only:

```text
runs/e1_canary/e1-moka-state0-pilot-v1.json
```

The report is single-use and hash-bound. A validated canary is required before
the canonical scored ledger exposes execution index `0`.

## 7. One scored branch, step by step

For the next unused frozen execution index:

1. Require a clean Git checkout and verify every outcome-generating source
   file against the plan's frozen runner-source tree.
2. Require a validated single-use canary and an unbroken in-order ledger.
3. Create `started.json`; the branch can never be overwritten or rerun under
   v1.
4. Restore the exact state-0 LIBERO reset and verify the initial public RGB and
   8-D state hashes.
5. Construct one candidate from the frozen public annotation, serialize it,
   and construct the selected referent interface.
6. Reset the model session. At model call `k`, send the current RGB pair,
   current state, fixed subtask, seed `26090800+k`, and frozen inference flags.
7. Require an exact finite `10×7` model chunk. For each action, cast all seven
   entries to float32, binarize the gripper to `-1` or `+1`, send that exact
   array to LIBERO, and record the same applied values.
8. After every applied action, append evaluator-only contact, target-grasp, and
   target-predicate values to the lowest-level trace. These values do not
   affect model input or early termination.
9. After the chunk, save both current RGB observations and the current 8-D
   state as a milestone. The next request must be linked to that milestone.
10. Repeat until 300 actions have been applied. Reaching the goal does not end
    the v1 branch early. A malformed/non-finite policy output, transport error,
    identity mismatch, simulator error, or other infrastructure exception
    produces no label and seals the ledger.
11. Recompute all diagnostics and the primary label from the per-step trace,
    create one `ObservedBranch` for the actually executed candidate, and seal
    the entire tree.

The five conceptual alternatives that were not executed receive no outcome.
There is no counterfactual label fabrication.

## 8. Primary label and diagnostics

Let `C_t` be the evaluator-only set of candidate moka pots touching the gripper
after applied simulator action `t`. Let

```text
t* = min { t : C_t is not empty }.
```

The supervised E1a outcome is:

```text
y = 1  iff  C_t* contains the intended moka pot
              and contains no distractor moka pot;
y = 0  otherwise.
```

Thus simultaneous first contact with intended and distractor is negative, as
are wrong-first contact and no candidate contact by the end of a normal
300-step horizon. Any failure that prevents completion of the frozen
intervention horizon—including malformed policy output—produces no
`ObservedBranch` and no training label.

The complete diagnostic dictionary is re-derived from the lowest-level trace:

- `first_contact_intended` — the primary outcome above;
- `wrong_first_contact` — a distractor participates in the first candidate
  contact;
- `no_candidate_contact`;
- `any_distractor_contact_within_horizon`;
- `distractor_contact_before_or_at_goal_or_horizon_end`;
- `first_contact_step`;
- `target_grasped` at least once;
- `full_physical_goal` and `goal_step`;
- `simulator_steps`, `model_calls`, and `fixed_horizon_completed`;
- `policy_output_failure`, fixed to `null` for every admitted label-bearing
  branch; and
- a fixed assertion that evaluator feedback did not change policy input.

`full_physical_goal` is not the training target. The stock BDDL goal names one
specific pot, so the evaluator instead applies the same intended-pot predicate
for either public target and logs it only as a target-balanced diagnostic.

## 9. Artifact tree and validation

A completed row lives at:

```text
runs/e1/e1-moka-state0-pilot-v1/<trial-id>/
```

Its implemented layout is:

```text
started.json
completed.json
artifact_seal.json
executor_request.json
executor_health.json
conditioned_input.json
action_chunks.json
actions_applied.json
milestones.json
public_execution_trace.json
execution_receipt.json
observed_branch.public.json
public_frames/
  sha256/<first-two-hex>/<decoded-RGB-sha256>.png
conditioned_frames/
  sha256/<first-two-hex>/<decoded-RGB-sha256>.png
private/
  evaluator_trace.json
  evaluator_sidecar.json
```

An infrastructure-failed row contains `started.json`, `failed.json`, its seal,
and any partial files created before the exception, but no completed result or
observed branch.

The static validator checks more than file existence. Given the external
frozen plan and trial ID, it:

- verifies every sealed file byte and rejects unsealed extras;
- reconstructs the candidate, serializer, execution request, session, and
  request hashes;
- resolves PNGs by decoded RGB content rather than trusting filenames;
- checks frame IDs, cameras, indices, dimensions, and hashes;
- links `state_t → request_t → returned chunk_t → exact applied actions_t
  → milestone state_(t+1)`;
- requires exactly 300 applied actions on a normal terminal branch;
- recomputes first contact and every diagnostic from `private.per_step`;
- recomputes the observed outcome and execution status;
- checks the typed `ObservedBranch`, receipt, sidecar, and all cross-file hashes;
- verifies live-backend attestation when empirical evidence is required; and
- rejects evaluator object-name leakage in the public execution trace.

The SHA chain proves byte integrity and internal reconstructibility relative
to the frozen plan. It is not a digital signature and, by itself, cannot prove
that a malicious local process did not fabricate a mutually consistent
simulator trace. A future stronger provenance layer can deterministically
replay the recorded exact actions in the pinned reset and compare per-step
contacts/predicates, or sign executor/evaluator receipts with an independently
held key.

## 10. From real result to an actual training loss

`load_e1_training_record(...)` first calls the complete artifact validator.
Its default `require_empirical=True` rejects faithful test-double runs. On a
valid live run it returns a typed `ValidatedE1TrainingRecord` containing:

```text
ObservedBranch
+ evidence class
+ plan/trial identity
+ artifact seal hash
+ outcome-contract hash
+ source directory
```

`collate_e1_outcome_targets(...)` then creates:

- a matrix of observed binary outcomes;
- a Boolean executed-candidate mask with exactly one true entry per row;
- candidate fingerprints that define tensor-column identity; and
- reset/context/contract fingerprints.

The batch checks the model output's candidate-fingerprint order before calling
the existing executed-candidate Bernoulli negative log-likelihood. Candidate
IDs are never used to align prediction columns. The software test performs a
real PyTorch forward loss and backward pass, but this repository revision does
not yet train the Stage-1 model on empirical E1 data.

## 11. Reproduction commands

### 11.1 GPU process: inspect and serve the frozen model

```bash
uv sync --extra molmoact2 --extra dev

uv run ip-serve-molmoact2 \
  --inspect-only \
  --checkpoint allenai/MolmoAct2-LIBERO \
  --revision 0d24a92bd1faf321ef497c3bbd5681af97c65aa2

export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run ip-serve-molmoact2 \
  --identity-json experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --host 127.0.0.1 --port 8003
```

With a remote server, tunnel the already-localhost-bound endpoint:

```bash
ssh -N -L 8003:127.0.0.1:8003 USER@GPU_HOST
```

### 11.2 LIBERO process: reset preflight

```bash
export LIBERO_REPO_ROOT=/absolute/path/to/LIBERO
export LIBERO_CONFIG_PATH=/absolute/path/to/.libero
export PYTHONPATH="$PWD/src:$LIBERO_REPO_ROOT"

python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --validate-only
```

Before the canary exists, the expected ledger status is
`BLOCKED_PENDING_MODEL_CANARY`.

### 11.3 Run the one outcome-free model canary

Only from a clean checkout whose source bytes match the plan:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --run-model-canary --allow-model-canary
```

To validate an existing canary without consuming another call:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --validate-canary
```

### 11.4 Execute one scored row

After the canary passes, execute only the next index in the frozen order:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --endpoint http://127.0.0.1:8003 \
  --execution-index 0 --allow-execution
```

Do not start index `1` until index `0` closes and validates. Do not rerun,
replace, skip, or overwrite an index. An infrastructure failure seals v1; any
repair requires a new, prospectively frozen execution amendment.

Validate a completed artifact tree with the external plan and explicit row:

```bash
python -m grounded_interaction.e1 \
  --plan experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --trial-id coarse-left \
  --validate-output runs/e1/e1-moka-state0-pilot-v1/coarse-left
```

## 12. Interpreting the six rows

Because E1a has one reset state and one branch per cell, it supports only
qualitative interface debugging:

- If all three modes hit the same physical pot regardless of requested side,
  the current executor interface does not control referent identity.
- If marker succeeds for both sides while text modes do not, public visual
  prompting is a viable ceiling and a spatial visual adapter is worth testing.
- If precise text succeeds for both sides, the language interface warrants a
  larger qualification study.
- If first contact follows the referent but full placement fails, referent
  transmission and manipulation reliability are separate bottlenecks.
- A single successful row is not evidence of generalization, calibration, or
  closed-loop information seeking.

The next experiment, E1b, must be frozen separately and should add a stock-task
positive control, a genuinely ambiguous no-spatial control, multiple untouched
reset states and seeds, balanced target sides, and reset-group-paired analysis.
Only after E1b establishes a usable referent channel should automatic
public-RGB proposals, a concrete frozen-VLM token provider, and learned Stage-1
outcome prediction enter outcome collection.

## 13. Primary sources

- [MolmoAct2 official repository](https://github.com/allenai/molmoact2)
- [MolmoAct2-LIBERO checkpoint and inference contract](https://huggingface.co/allenai/MolmoAct2-LIBERO)
- [MolmoAct2 paper](https://arxiv.org/abs/2605.02881)
- [LIBERO official repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
