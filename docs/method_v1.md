# Method V1: grounded outcome prediction with a fixed continuation

Method V1 is the first falsifiable implementation of the project method. It
does not ask a VLM to emit an uncalibrated scalar called “uncertainty,” and it
does not train a second action generator. It learns one quantity:

    p_theta(h, c) = P(Y = 1 | h, execute(c), then follow fixed pi_C)

Here:

- h is the policy-visible task history before the high-level choice;
- c is one complete, visibly grounded candidate intervention;
- pi_C is the frozen continuation policy;
- Y is final task success within the registered maximum 300-control-step
  horizon; and
- p_theta is a candidate-conditional outcome probability.

The robot compares these probabilities over the candidate set generated at
the current reset. A high value is useful only under the frozen execution and
continuation contract. It is not a general confidence score, a POMDP belief, or
proof that the current observation is sufficient.

## Current implementation status

The provider, candidate parser, token mapping, outcome model, fixed
continuation, reset-paired data contracts, training loop, calibration utility,
evaluation utilities, and MolmoAct2/LIBERO adapters are implemented. Software
tests use controlled tensors and test doubles only where external model or
simulator resources are unavailable.

The following have not been run in this revision:

- a full Qwen2.5-VL-3B hidden-state forward;
- a Method-V1 Qwen-to-MolmoAct2-to-LIBERO branch rollout;
- reset-group outcome collection at the configured scale;
- scorer training on empirical Method-V1 outcomes; or
- held-out closed-loop policy evaluation.

The semantic revealed-region masking intervention and a dedicated collector
for new held-out closed-loop policy episodes are also not implemented in this
revision. Existing evaluation code can verify checkpoint-derived offline
paired-branch estimates and summarize externally supplied episode records; it
does not turn those records into fresh robot executions.

The registered region/history/task/descriptor ablation launchers are likewise
not implemented or run. Their definitions below are experiment requirements,
not completed evidence.

No test-double result is reported as robot evidence.

## Model responsibilities and exact interfaces

Method V1 is not a single VLM prompted to invent an uncertainty number. Its
three model boundaries have different inputs, outputs, and trainability:

| Component | Policy-visible input | Output consumed downstream | Trained here? |
| --- | --- | --- | --- |
| Qwen proposal path | task, completed public action history, current public agentview RGB and its frame identity | a bounded set of grounded `DIRECT`/`OPEN` candidates | no |
| Qwen token path | task, labelled public RGB/history, candidate instruction text, optional public proprioception and budget | detached context tokens, candidate features, spatial support masks and public-state tensors | no |
| `GroundedOutcomeModel` | only the frozen tensors bound to one decision manifest | one final-task-success logit for every candidate | yes |
| MolmoAct2 | selected grounded subtask, live agentview/wrist RGB, live public 8-D state | one finite `10 x 7` continuous action chunk per call | no |

Qwen therefore supplies an open-vocabulary candidate vocabulary and a frozen
semantic-visual representation. It does **not** choose the Method-V1 action and
does not see outcomes. The outcome model performs candidate-conditional
prediction but does not generate robot motion. MolmoAct2 receives the selected
subtask and live observations, but it does not receive Qwen hidden states or
the evaluator label.

The exact object crossing each boundary is:

    Qwen proposal JSON
      -> GroundedIntervention[]
      -> DecisionGroupManifest + BranchScheduleEntry[]

    frozen Qwen forward
      -> CandidateTokenField
      -> immutable tensor cache + sidecar

    trained outcome-model forward
      -> ValuePrediction[]
      -> SelectionDecision
      -> provenance-bound selection artifact

    selected GroundedIntervention
      -> ExecutorRequest
      -> MolmoAct2 action chunks
      -> LIBERO transition and public reobservation
      -> one evaluator-only Boolean final-task outcome

Candidate IDs and `visible_label` strings are not embedded as separate
privileged features; referent language may still appear naturally in the
canonical instruction. Every join between these objects is checked using the
full candidate fingerprint and the frozen manifest identities.

## System boundary

The policy has two stages. Stage 1 contains frozen Qwen plus the small outcome
scorer; Stage 2 is frozen MolmoAct2. Only the outcome scorer is trainable:

    public prompt, RGB/history, public state
                    |
                    v
    frozen Qwen2.5-VL-3B
      - proposes DIRECT and OPEN candidates
      - extracts public context and candidate features
                    |
                    v
    trainable GroundedOutcomeModel
      - predicts one final-task logit per candidate
                    |
                    v
    deterministic finite-set selector
      - selects the feasible candidate with maximum probability
                    |
                    v
    frozen MolmoAct2-LIBERO
      - receives RGB, public state, and the selected grounded subtask
      - produces 10-step continuous action chunks
                    |
                    v
    real LIBERO transition and reobservation

Qwen, scorer replay, MolmoAct2, and legacy LIBERO run in dependency-compatible
processes. The scorer never sends its latent vectors into the VLA. Simulator
semantic IDs, hidden object states, predicates, rewards, and outcome labels are
excluded from all policy inputs.

## Exact policy-visible inputs

At a high-level decision boundary, h contains:

1. the original user task q;
2. the current agentview and wrist RGB images;
3. up to two earlier completed high-level boundaries, each represented by its
   public RGB and the action text/status actually issued;
4. the current public 8-D proprioceptive state, when available; and
5. remaining budget fraction, equal to 1.0 at the initial Method-V1 decision.

History is encoded as one ordered, labelled multimodal Qwen input. Images are
labelled by camera, frame identity, frame index, and whether they are current
or historical. Missing history is not filled by copying frames.

The initial proposer uses the original task and current public agentview. After
OPEN, the same frozen proposer additionally receives the completed public
action history and the real post-OPEN agentview. It never receives the hidden
contents of a container from a simulator API.

## Candidate contract

Qwen emits strict JSON with exactly these fields for each candidate:

    visible_label
    bbox
    primitive
    instruction

Only DIRECT and OPEN are enabled. There are at most six candidates in total
and at most three per primitive. The parser rejects rather than repairs:

- prose or Markdown around the JSON;
- duplicate or extra keys;
- non-finite, empty, out-of-bounds, or zero-area boxes;
- unknown or disabled primitives;
- empty instructions or reserved control-token patterns; and
- duplicate instruction/region pairs.

The frozen prompt also requires the output array to be ordered from the
candidate Qwen judges most likely to complete the final task to the least
likely, under the same 300-step DIRECT versus 100+200-step OPEN contract used
by Method V1. This ordinal order defines the `frozen_vlm` baseline and the
post-OPEN DIRECT tie-break. Qwen does not emit a confidence number, and its
rank is never substituted for the learned outcome probability used by the
Method-V1 selector.

The proposer is not given a hidden route label or the instruction “explore.”
Its fixed system prompt defines the JSON schema, the meanings of DIRECT and
OPEN, visible-evidence-only grounding, and the cardinality limits. The
instance-specific user message has exactly this information pattern:

    Task: <original final-task prompt>
    <completed public action history, or "none">
    Public camera: <camera>; public frame: <frame_id>.
    Processed image width=<w>, height=<h>.
    Return candidate JSON under the system schema.

The current public image follows as the only image item. The literal frozen
system prompt and proposer-identity hash are defined in
`src/grounded_interaction/proposals.py`; this document does not duplicate that
string as an independently editable source of truth.

The Qwen coordinate contract is
qwen25-smart-resized-absolute-xyxy-v1. Coordinates are absolute pixels in the
image dimensions produced by the pinned slow Qwen processor. They are
explicitly transformed back to normalized coordinates in the original public
RGB. The code never assumes a legacy 0–1000 coordinate range.

A candidate fingerprint binds the primitive, full instruction, visible
referent, current camera/frame, image digest, and exact normalized box. A
coarse phrase such as “the object near the centre” is therefore not the
candidate identity.

The decision freeze also writes `proposal_audit.json`. It preserves the raw
Qwen response and its SHA-256, processor-space image dimensions, every
deterministic rejection, accepted candidate identities, and one hashed
grounding overlay per accepted candidate. The overlay draws the candidate box
and the exact current-image patch support used by the outcome model. This is a
human audit view, not an additional policy input. Validation reparses the raw
response instead of trusting the accepted list alone.

## Frozen token construction

Let D be the Qwen language-backbone hidden size read from the real checkpoint.
For one public context, the provider returns:

    Z in R^(N x D)       contextual public input tokens
    m in {0,1}^N         valid-token mask
    p in R^(N x 4)       normalized patch anchors
    current in {0,1}^N   current-image patch mask

Image-token positions come from the actual processor input IDs and
image_grid_thw. The number and layout of patches are recovered using the
pinned patch size and spatial merge size, including non-square images. The
implementation does not infer a square grid from sqrt(N).

The patch boxes are spatial anchors for already-contextualized Qwen states.
They do not imply that an image token contains only local information.

For candidate i, the same frozen Qwen text branch encodes the canonical action
instruction. The last token belonging to the instruction content is pooled as
e_i. The raw candidate feature is:

    C_i = concat(e_i, bbox_i, one_hot(primitive_i))

The one-hot order is persisted as DIRECT, OPEN. Candidate order is not encoded.

The grounding support S_i selects only current image tokens from the
candidate's bound camera and frame. Patch centres inside the candidate box are
used first. If this is empty, the single patch with greatest positive
intersection is used. No positive intersection makes the candidate invalid.
Overlay rendering is available to inspect the exact box and selected patches.

All Qwen tensors are detached, converted to float32 on CPU, and stored in a
content-addressed cache. Two hashes have deliberately different meanings:

- the **logical key** identifies what should have been encoded: provider ID,
  public-context fingerprint, ordered candidate IDs, and full candidate
  fingerprints; and
- the **physical artifact identity** binds what was actually serialized:
  the logical key, SHA-256 of the `.pt` tensor file, and SHA-256 of its JSON
  sidecar.

In compact form:

    k_logical  = H(provider_id, context_fp, candidate_ids, candidate_fps)
    k_artifact = H(cache_schema, k_logical, H(tensor_file), H(sidecar_file))

where `H` is canonical SHA-256 and `fp` denotes a full content fingerprint.

The cache stores the same immutable pair under the physical artifact address
and provides a logical-key hard link for generation-time reuse. A formal
decision manifest records the physical artifact identity, not merely the
logical key. Formal loading re-hashes both files, recomputes the sidecar and
every named tensor, and checks model/provider, context, candidate, primitive,
camera, and frame identities. Consequently two encodings of the same nominal
input cannot be silently substituted just because their logical keys match.

The full cache identity also binds the model revision,
processor/Transformers contract, templates, primitive order, and tensor byte
digests. Future frames and outcome labels are absent.

## Trainable forward pass

Only the small outcome model is trained. With hidden width d=256:

    Z_h = context_projection(Z)
    C_h = candidate_projection(C)

The first attention stage reads only the candidate's declared spatial support:

    L_i = LocalAttention(query=C_h[i], key=Z_h[S_i], value=Z_h[S_i])
    G_i = LayerNorm(C_h[i] + output_projection(L_i))

The second attention stage lets the grounded candidate read every valid public
context token:

    A_i = GlobalAttention(query=G_i, key=Z_h[m], value=Z_h[m])
    H_i = LayerNorm(G_i + A_i)
    H_i = LayerNorm(H_i + FFN(H_i))
    logit_i = Linear(H_i)

If the 9-D public state is present, it is normalized with train-split-only
statistics, projected by a trainable Linear plus LayerNorm, and appended as
one key/value token to the global attention stage. It is never used in local
grounding. Missing state has an explicit validity mask.

G_i and H_i are internal predictive states. The implementation does not rename
them as uncertainty tokens. The decision probability is:

    p_i = sigmoid(logit_i)

An optional positive scalar temperature T may be fitted on a disjoint
calibration split:

    p_i_cal = sigmoid(logit_i / T)

Temperature scaling changes probability reliability but not candidate order.
The checked-in Method-V1 configuration disables calibration, so the current
canonical prediction exporter uses identity temperature `T = 1`. The fitting
utility alone is not calibration evidence; a later calibrated experiment must
freeze a disjoint calibration split and an identity-bound calibrator artifact
before any test or deployment decision.

## Supervision and split protocol

Candidate proposal occurs once per exact reset and the entire candidate set is
frozen. The simulator is then restored to the same initial state for every:

    candidate x two preregistered MolmoAct2 model seeds

Each schedule row executes one candidate. It is either:

- OUTCOME_EVALUATED, with one Boolean final-task label backed by either
  completed execution receipts or a sealed policy-failure trace; or
- INFRASTRUCTURE_FAILURE, with diagnostics and no label.

Unexecuted candidates remain unknown. They are not assigned zero. Training
uses the existing executed-candidate Bernoulli negative log likelihood:

    L = -[Y log p_i + (1-Y) log(1-p_i)]

only at the actually executed candidate position.

The split firewall keeps a reset and all of its candidate branches in one of
train, validation, calibration, or test. Scene/layout and paired hidden-family
isolation are enforced from the registered `split_group_id`; deriving that
family from assets remains an upstream dataset-authoring responsibility.
Random frame-level splitting is invalid. Each checkpoint identity records all
train/validation split-group IDs, and learned selection or formal test replay
fails closed if a target manifest overlaps them or changes experiment,
configuration, proposer, token-provider, or outcome-contract identity.

## Fixed execution and continuation

Every candidate is evaluated under the same maximum 300-control-step horizon
and 10-step VLA replanning interval.

DIRECT:

    selected DIRECT -> MolmoAct2 for 300 steps -> final evaluation

OPEN:

    selected OPEN -> MolmoAct2 for 100 steps
                  -> real public RGB/state reobservation
                  -> frozen Qwen proposes at most three DIRECT candidates
                  -> execute Qwen's first valid DIRECT for 200 steps
                  -> final evaluation

The trainable scorer is called once, before the initial action. It is never
called after OPEN. The switch occurs after the fixed 100-step budget and
cannot read a private “drawer is open” predicate. If Qwen proposes no valid
post-OPEN DIRECT candidate, the episode stops after OPEN and is evaluated
under the same final-task contract.

MolmoAct2 receives its actual public RGB, 8-D public state, and precise-text
serialized instruction. Each action chunk must be exactly 10 by 7 and finite.
The serializer names the visible referent and renders its normalized centre
and `(x0, y0, x1, y1)` box into text, with x measured from the left and y from
the top. These coordinates are **not** a native MolmoAct2 point/box channel;
whether the frozen executor reliably follows this textual referent is exactly
the interface limitation isolated by E1a, not an assumed capability.

For every 10-step call, the public trace stores the exact agentview and wrist
RGB as content-addressed files, their frame metadata, the eight state values
and state digest, instruction, derived model seed, request identity, latency,
and the exact actions applied to LIBERO. Stage receipts and real post-action
frames link the entire 100/200/300-step trajectory. A model-output failure is
recorded with the physically executed prefix and receives no invented future
actions.

Byte-level tree hashing is followed by semantic replay validation. Before an
outcome-bearing attempt is admitted, `method_v1_trace.py` resolves every saved
RGB file, recomputes every public-state digest and MolmoAct2 request ID, checks
the stage/session seed derivation, verifies each completed chunk contains ten
finite 7-D actions, re-forms the receipt chain and public context transitions,
and requires exactly the registered DIRECT or OPEN-to-DIRECT budget topology.
An infrastructure failure remains unlabelled even if it contains a valid
completed physical prefix.

The initial Qwen raw proposal, parser rejections, processed-image dimensions,
candidate boxes, patch supports, and overlays are preserved in the decision
freeze. The post-OPEN proposal is a separate online public-RGB event and must
remain distinguishable from the initial frozen candidate set in the execution
trace. Evaluator predicate names and values are written only to the private
sidecar after execution and never enter either model request.

## Data-generation stages

The collector deliberately separates operations that need incompatible model
environments. The reset population is frozen before Qwen is allowed to
propose candidates, and the executable branch population is frozen before any
outcome:

1. freeze-inventory reopens every registered scene spec and freezes the exact
   reset denominator, including split, information stratum, prompt, asset
   hashes, init-state index, environment seed, reset digest, and paired model
   seeds; it copies no evaluator predicate into a policy input;
2. render runs in the pinned LIBERO environment and exports an exact public
   reset context;
3. freeze runs in the Qwen environment, verifies that the reset is an exact
   inventory member, proposes once, extracts frozen features, and emits one
   terminal status: a valid DIRECT/OPEN choice set, a valid single-primitive
   set, or an immutable proposal failure;
4. freeze-plan requires exactly one terminal for every inventory row. It seals
   the full pre-proposal denominator, proposal and choice coverage, all valid
   paired schedules, and the scorer-verifier authentication-key identity while
   all schedule rows are unused;
5. run returns to LIBERO and executes exactly one single-use schedule row;
   MolmoAct2 drives every physical branch, while the Qwen HTTP service is used
   only for a post-OPEN continuation proposal;
6. finalize validates byte integrity, semantic execution traces, evaluator
   separation, and schedule coverage, then produces a self-contained
   diagnostic snapshot. Canonical training and formal evaluation do not trust
   that snapshot: they reopen the freeze receipts and every source
   `attempt.json` through the collector validators and require the exact group
   set named by the pre-outcome plan.

The evaluator-only final predicate is stored in the private scene
specification and queried only after policy execution. It cannot affect
proposal, scoring, VLA input, switching, or termination.

Every reset is also assigned, before any outcome is observed, to exactly one
evaluator-only information stratum:

- `INFORMATION_NECESSARY`: direct observation is intentionally insufficient
  and an enabled information action can reveal task-relevant evidence;
- `INFORMATION_SUFFICIENT`: the public observation already supports direct
  task execution; and
- `INFORMATION_ACTION_NO_HELP`: an information action is executable but does
  not reveal evidence needed by the final task.

The stratum appears only at the top level of the decision manifest and global
collection plan. It is included in their hashes and count audits but is absent
from `DecisionGroupManifest.model_input()`. The main config pre-registers
split-by-stratum counts. After collection, the training entry point separately
requires observed support for all four `DIRECT/OPEN x success/failure` cells in
train and validation without consulting held-out test outcomes. Test coverage
is reported only during final evaluation. This is a data-quality gate, not a
counterfactual label: only actually executed branches contribute outcomes.

The canonical file boundary is:

| Command | Outcome-free inputs | Canonical outputs |
| --- | --- | --- |
| `freeze-inventory` | source config and every preregistered scene-reset spec with its BDDL/init-state assets | immutable pre-Qwen reset inventory and per-row identities |
| `render` | scene-reset spec, BDDL, exact init-state file/index | `prepared_public_context.json` and content-addressed initial public RGB |
| `freeze` | reset inventory, prepared context, Method-V1 config, frozen executor identity, Qwen checkpoint | inventory-bound valid decision freeze and schedule, or an inventory-bound proposal-failure terminal |
| `freeze-plan` | reset inventory, source config, every valid decision freeze, every proposal-failure directory, private verifier key | immutable plan containing the full pre-proposal denominator, proposal/choice coverage, valid schedules, and only the verifier key ID |
| `selection_provenance` | one frozen manifest/schedule row, physical cache, real scorer checkpoint and identity, optional frozen calibrator | one immutable pre-execution selection artifact, accepted only after checkpoint replay |
| `run` | global plan, source config, one member freeze, verifier key, one unused execution index, live MolmoAct2/Qwen services, and—for a learned decision—the authenticated scorer-verifier plus frozen selection | plan-bound claim and started receipts; per-chunk public RGB/state/action trace; public observed branch plus private final evaluator sidecar, or an unlabelled infrastructure-failure attempt |
| `finalize` | global plan, source config, its exact freeze set and every immutable attempt | one self-contained diagnostic `MethodV1OutcomeDataset` snapshot |
| `train_outcomes` | global plan, source config, its exact outcome-free freeze set, only the immutable train/validation attempts, physical Qwen cache roots | split-scoped receipt-admitted checkpoints, identity sidecars, histories, training report; calibration/test outcomes are never opened |
| `evaluate_policy export-predictions` | global plan, source config, its exact freeze set, every immutable attempt, physical caches, real checkpoints | receipt-admitted, checkpoint-derived canonical prediction artifact |

No command copies evaluator fields into a decision manifest, Qwen request,
cached tensor, selection artifact, MolmoAct2 request, or continuation proposal.

Every `run` first verifies the global collection plan and writes its digest and
verifier-key ID into the immutable execution claim. Canonical data admission
later rejects a missing, substituted, duplicated, or differently split whole
freeze group, not merely a missing candidate row inside one group. The plan
must be archived or committed with its digest before the first execution;
local hashes establish consistency but do not provide an external timestamp.

The inventory-aware plan is the required schema for new formal experiments.
The legacy plan remains readable only so historical software artifacts do not
break; because it was assembled after proposal, its report is explicitly
labelled `POST_PROPOSAL_CONDITIONAL_LEGACY_PLAN` and cannot establish proposal
coverage over the registered reset population. Probability metrics are always
conditional on a valid proposal set because no branch probability exists when
proposal itself fails. The report therefore carries both that conditioning
statement and the independent pre-proposal proposal/choice coverage counts.

`run` without `--selection` executes the candidate already named by the
schedule row. That is the data-collection path used to obtain the full paired
branch matrix; it is **not** evidence that the learned scorer selected the
action. A learned pre-execution decision must first be created by the real
checkpoint through `selection_provenance`, then passed to `run --selection`.
Before touching the simulator, the legacy LIBERO collector sends the frozen
selection, manifest, and schedule file identities to an isolated
modern-PyTorch verifier. That service re-hashes the files, reloads the exact
checkpoint and physical Qwen cache on CPU, and recomputes every candidate
probability and argmax. The collector accepts the response only if its service
source, complete replay-source digest, runtime identity, request/response
digests, selected schedule row, fresh nonce, and HMAC tied to the plan's key ID
all agree. Recomputing public JSON hashes without the frozen secret is
therefore insufficient to authorize execution, while the legacy PyTorch 1.11
simulator process never imports the checkpoint loader. This is authenticated
process separation, not hardware attestation. The verified response preserves
the checkpoint's training-dataset, receipt-admission and training-plan digests,
and the collector rejects a training key different from the current verifier
key. Formal offline evaluation recomputes these fields from collector sources;
the single-branch physical verifier does not independently prove that an
actively malicious host trained the supplied weights as declared. Compromise
of the shared host or key remains outside the guarantee.

## Configuration and environments

The human-readable, identity-bound defaults are in:

- experiments/method_v1.yaml
- env/method_v1-qwen.txt
- env/README.md

The Qwen environment is intentionally separate from both the legacy LIBERO
process and the MolmoAct2 server. A minimal setup is:

~~~bash
python3.11 -m venv .venv-qwen
source .venv-qwen/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -r env/method_v1-qwen.txt
python -m pip install -e . --no-deps
~~~

The frozen checkpoint is Qwen/Qwen2.5-VL-3B-Instruct at revision
66285546d2b821cf421d4f5eb2576359d3770cd3 with Transformers 4.57.6, the slow
processor, and SDPA. The MolmoAct2 identity remains the separately pinned
E1-compatible identity.

Start the Qwen proposal service only after the model environment is ready:

~~~bash
python -m grounded_interaction.qwen_service \
  --host 127.0.0.1 --port 8004 --device-map auto
~~~

Start the learned-selection verifier from the modern learned-model environment
(PyTorch 2.11 or newer), without loading Qwen or MolmoAct2:

~~~bash
mkdir -p runs/method_v1/secrets
# Run this subshell exactly once; noclobber refuses an existing experiment key.
(umask 077; set -C; openssl rand 32 > runs/method_v1/secrets/scorer-verifier.key)
python -m grounded_interaction.scorer_verification_service \
  --host 127.0.0.1 --port 8005 \
  --auth-key-file runs/method_v1/secrets/scorer-verifier.key
~~~

The verifier and LIBERO runner currently need a shared filesystem: absolute
paths embedded in the selection artifact must resolve to the same checkpoint,
cache, manifest, and schedule bytes in both processes. The service binds its
own source plus every local source file used for replay into its runtime
identity. The standard-library-only legacy client authenticates fresh-nonce
requests and responses with the same private HMAC-SHA256 key before a
single-use branch can be claimed. Only the key's SHA-256 identifier enters the
pre-outcome plan; the secret remains an ignored local file. This boundary
rejects an unkeyed endpoint that merely fabricates internally consistent
hashes, but does not provide hardware attestation or protect a compromised
host or leaked key.

The MolmoAct2 server remains on localhost port 8003 as documented in
env/README.md.

## Collection commands

Create all strict scene-reset specifications from the provided example. Their
prompts must state only the final task, not the desired exploratory action.
Paths, reset digests, evaluator predicates, and both model seeds must be filled
before use. Then freeze the complete reset population **before the first Qwen
proposal**. Repeat `--scene-spec` once for every registered reset; the two rows
below are only an abbreviated illustration, and the command fails unless the
counts exactly match the source config:

~~~bash
python -m grounded_interaction.collect_outcomes freeze-inventory \
  --scene-spec runs/method_v1/scene_specs/group_000.json \
  --scene-spec runs/method_v1/scene_specs/group_001.json \
  --config experiments/method_v1.yaml \
  --inventory-id method-v1-main-reset-population \
  --output runs/method_v1/reset_inventory.json
~~~

Export the exact public reset:

~~~bash
python -m grounded_interaction.collect_outcomes render \
  --scene-spec runs/method_v1/scene_specs/group_000.json \
  --output-dir runs/method_v1/prepared/group_000
~~~

In the Qwen environment, freeze candidates, features, and all paired branches:

~~~bash
python -m grounded_interaction.collect_outcomes freeze \
  --scene-spec runs/method_v1/scene_specs/group_000.json \
  --prepared-dir runs/method_v1/prepared/group_000 \
  --config experiments/method_v1.yaml \
  --executor-identity experiments/e1_referent_ceiling/pilot_state0_v1.json \
  --reset-inventory runs/method_v1/reset_inventory.json \
  --output-dir runs/method_v1/frozen/group_000
~~~

Repeat `render` and `freeze` for every inventory row. A valid Qwen result leaves
a decision freeze; a proposal failure leaves `proposal_failure.json` in that
row's output directory. Neither may be replaced after inspecting the result.
Before executing any branch, freeze the **complete** terminal population into
one global collection plan. Repeat `--freeze-dir` for every valid result and
`--proposal-failure-dir` for every failure; the abbreviated command below shows
one of each:

~~~bash
python -m grounded_interaction.collect_outcomes freeze-plan \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --proposal-failure-dir runs/method_v1/frozen/group_137 \
  --reset-inventory runs/method_v1/reset_inventory.json \
  --config experiments/method_v1.yaml \
  --plan-id method-v1-dev-plan \
  --scorer-auth-key-file runs/method_v1/secrets/scorer-verifier.key \
  --output runs/method_v1/collection_plan.json
~~~

The plan must match the exact split and information-stratum counts in the
source config and must contain exactly one terminal per inventory row. A reset
with only one primitive remains proposal-covered but is not choice-eligible. A
proposal failure remains in the denominator. If every reset fails proposal,
`freeze-plan` accepts zero `--freeze-dir` arguments and still emits a valid 0%
coverage plan; training and branch evaluation then correctly remain
unavailable. For a small integration diagnostic, first freeze a separate
experiment config and its complete smaller population; do not silently use a
subset of the main study plan.

Back in the LIBERO environment, run one previously unconsumed schedule index:

~~~bash
python -m grounded_interaction.collect_outcomes run \
  --scene-spec runs/method_v1/scene_specs/group_000.json \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --collection-freeze-dir runs/method_v1/frozen/group_000 \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --scorer-auth-key-file runs/method_v1/secrets/scorer-verifier.key \
  --execution-index 0 \
  --molmo-endpoint http://127.0.0.1:8003 \
  --qwen-endpoint http://127.0.0.1:8004 \
  --output-root runs/method_v1/outcomes
~~~

Before the single-use claim is written, the runner checks the MolmoAct2
identity/health endpoint and calls Qwen `/ready`, which forces the frozen Qwen
weights to load. A missing model or model-load OOM therefore does not consume
an index after the robot has already altered the scene. When `--selection` is
present, it additionally requires the identity-bound scorer verifier on port
8005 to replay and approve that exact learned decision before the claim.

There is no overwrite or rerun path. A service-readiness failure before the
claim leaves the row unused. After the claim, an infrastructure failure
consumes the attempt identity but produces no training target.

Every `run` invocation must repeat `--collection-freeze-dir` for the complete
freeze set named by the plan, even though `--freeze-dir` identifies the one
group containing the selected row. This makes omission of an entire group a
pre-execution error rather than a post-hoc dataset choice.

After collecting the preregistered rows, optionally export a diagnostic
snapshot:

~~~bash
python -m grounded_interaction.collect_outcomes finalize \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --require-complete \
  --output runs/method_v1/dataset.json
~~~

`ENTRY_ID` is the immutable 64-character `entry_id` in
`branch_schedule.json`, and it is also the directory printed by `run`.
Repeat both `--freeze-dir` and `--attempt` for every decision group and
consumed row. With `--require-complete`, any missing candidate/seed branch or
infrastructure failure prevents the snapshot from being emitted. This JSON is
convenient for inspection only; it cannot enter canonical training or formal
evaluation because its source traces and evaluator sidecars are no longer
self-authenticating in isolation.

## Training and evaluation

Training defaults are AdamW, learning rate 3e-4, weight decay 1e-2, effective
batch size 64, gradient clipping at 1.0, at most 30 epochs, validation-NLL
patience 5, and independent seeds 0, 1, and 2. Frozen Qwen features remain on
disk/CPU while the small scorer trains.

The canonical training entry point is:

~~~bash
python -m grounded_interaction.train_outcomes \
  --collection-plan runs/method_v1/collection_plan.json \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --cache-root runs/method_v1/frozen/group_000/feature_cache \
  --config experiments/method_v1.yaml \
  --output-dir runs/method_v1/checkpoints \
  --device cuda
~~~

Repeat `--freeze-dir` for every group in the global outcome-free plan and
`--cache-root` for every freeze group's `feature_cache` directory. Repeat
`--attempt` **only** for train and validation schedule rows. The training
loader rejects a calibration/test path before opening the attempt, evaluator
sidecar, or public trace. Admission requires exact consumption of the
train/validation schedule scope; an infrastructure failure in that scope
remains explicitly unlabelled under the frozen outcome contract and cannot be
silently omitted. Add `--require-complete` when the run contract requires every
admitted train/validation row to carry an outcome rather than an
infrastructure-failure receipt. The checkpoint identity binds only the
train/validation receipt evidence while the immutable global plan hash still
commits to the held-out population. Training also fails closed
if any required split lacks an actually observed `DIRECT` or `OPEN` success or
failure; this prevents a degenerate primitive-prior dataset from silently
becoming the main method experiment.

### Provenance-bound learned selection on a fresh plan

A deployment choice is not accepted from a loose probability list. More
importantly, do **not** point this step back at the paired branch-collection
plan used for training: those rows have already been consumed, and replaying a
selected row would not be a fresh policy episode. First freeze a distinct,
outcome-free policy-evaluation population after the scorer checkpoint is
fixed. Its split groups must be disjoint from every checkpoint training group,
and its execution/representation identity must remain compatible with the
checkpoint. The current generic runner can execute one such single-use
integration branch; the dedicated multi-episode policy-evaluation plan and
aggregator are still listed as unimplemented.

On one fresh evaluation freeze, choose one pre-registered repetition seed. The
command scores the complete candidate set once, takes the deterministic
argmax, uniquely resolves that candidate and seed to the frozen schedule, and
freezes the decision together with its evidence:

~~~bash
python -m grounded_interaction.selection_provenance \
  --manifest runs/method_v1/policy_eval/frozen/group_000/decision_manifest.json \
  --schedule runs/method_v1/policy_eval/frozen/group_000/branch_schedule.json \
  --model-seed "$MODEL_SEED" \
  --cache-root runs/method_v1/policy_eval/frozen/group_000/feature_cache \
  --checkpoint runs/method_v1/checkpoints/seed_0/best.pt \
  --checkpoint-identity runs/method_v1/checkpoints/seed_0/identity.json \
  --expected-checkpoint-sha256 "$CHECKPOINT_FILE_SHA256" \
  --expected-checkpoint-identity-sha256 "$CHECKPOINT_IDENTITY_SHA256" \
  --temperature 1.0 --device cuda \
  --output runs/method_v1/policy_eval/selections/group_000-seed_${MODEL_SEED}.json
~~~

The resulting immutable artifact binds the manifest and schedule entry,
scorer probabilities and deterministic argmax, checkpoint bytes and identity
sidecar, the checkpoint's training-dataset, receipt-admission and training-plan
digests, cache logical key and physical tensor/sidecar identities, outcome
contract, and calibration mode. A non-unit temperature additionally requires
a separately frozen calibrator and its expected file digest. The command
prints the uniquely resolved `execution_index`; use that value as `INDEX`
below. The legacy `--execution-index` mode remains available, but fails unless
that row is already the scorer's argmax.

Pass that artifact into the physical branch runner:

~~~bash
python -m grounded_interaction.collect_outcomes run \
  --scene-spec experiments/method_v1/policy_eval_scene_reset_spec.json \
  --freeze-dir runs/method_v1/policy_eval/frozen/group_000 \
  --collection-freeze-dir runs/method_v1/policy_eval/frozen/group_000 \
  --collection-plan runs/method_v1/policy_eval/collection_plan.json \
  --config experiments/method_v1.yaml \
  --scorer-auth-key-file runs/method_v1/secrets/scorer-verifier.key \
  --execution-index "$INDEX" \
  --selection runs/method_v1/policy_eval/selections/group_000-seed_${MODEL_SEED}.json \
  --scorer-endpoint http://127.0.0.1:8005 \
  --molmo-endpoint http://127.0.0.1:8003 \
  --qwen-endpoint http://127.0.0.1:8004 \
  --output-root runs/method_v1/policy_eval/outcomes
~~~

The current global-plan schema requires the fresh evaluation plan to freeze the
complete population registered by its source config, even when only one row is
used for an integration check. A smaller diagnostic therefore needs its own
small config **and a checkpoint trained under that same resolved config**; a
main-study checkpoint cannot be paired with an incompatible config hash.
Non-selected candidate rows in this diagnostic plan are not retroactively
called policy outcomes and must not be sent to the paired-branch `finalize` or
training commands. Until the dedicated held-out episode runner is implemented,
this path is an integration check rather than the RSS policy result.

Training success is not policy evidence. Formal probability and paired-branch
reports cannot consume a user-authored JSONL of claimed probabilities. First
run the real checkpoint(s) over the identity-bound dataset/cache and freeze one
canonical prediction artifact:

~~~bash
python -m grounded_interaction.evaluate_policy export-predictions \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --cache-root runs/method_v1/frozen/group_000/feature_cache \
  --checkpoint runs/method_v1/checkpoints/seed_0/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_1/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_2/best.pt \
  --device cuda \
  --output runs/method_v1/evaluation/canonical_predictions.json
~~~

Repeat all source and cache arguments exactly as for training. The artifact
binds the admitted dataset evidence, outcome contract, Qwen provider/cache,
checkpoint files, checkpoint identities, and the train/validation admission
evidence stored in each checkpoint. Verify it by reopening all source
artifacts and re-running the same checkpoints before reporting any metric:

~~~bash
python -m grounded_interaction.evaluate_policy verify-predictions \
  --artifact runs/method_v1/evaluation/canonical_predictions.json \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --cache-root runs/method_v1/frozen/group_000/feature_cache \
  --checkpoint runs/method_v1/checkpoints/seed_0/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_1/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_2/best.pt \
  --device cuda
~~~

Formal reports consume the checkpoint-regenerated probabilities returned by
verification, not the numbers declared in the artifact. This prevents a
tampered value—or a tolerated tiny device-level replay difference—from
changing an argmax while retaining otherwise valid metadata.

Then compute probability quality and one registered paired-branch baseline:

~~~bash
python -m grounded_interaction.evaluate_policy formal-probabilities \
  --artifact runs/method_v1/evaluation/canonical_predictions.json \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --cache-root runs/method_v1/frozen/group_000/feature_cache \
  --checkpoint runs/method_v1/checkpoints/seed_0/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_1/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_2/best.pt \
  --split test --prediction-source deep_ensemble --device cuda \
  --output runs/method_v1/evaluation/test_probability_metrics.json

python -m grounded_interaction.evaluate_policy formal-branch-matrix \
  --artifact runs/method_v1/evaluation/canonical_predictions.json \
  --collection-plan runs/method_v1/collection_plan.json \
  --config experiments/method_v1.yaml \
  --freeze-dir runs/method_v1/frozen/group_000 \
  --attempt "runs/method_v1/outcomes/${ENTRY_ID}/attempt.json" \
  --dataset-id method-v1-dev \
  --cache-root runs/method_v1/frozen/group_000/feature_cache \
  --checkpoint runs/method_v1/checkpoints/seed_0/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_1/best.pt \
  --checkpoint runs/method_v1/checkpoints/seed_2/best.pt \
  --split test --baseline deep_ensemble --device cuda \
  --output runs/method_v1/evaluation/test_branch_matrix.json
~~~

The formal commands require complete candidate-by-seed coverage on the held-out
split, immutable receipt/claim coverage, and replayed checkpoint outputs. They
do **not** require the observed test set to contain both successes and failures
for each primitive: all-success or all-failure is a scientific result and is
reported as `DEGENERATE_ALL_SUCCESS` or `DEGENERATE_ALL_FAILURE`, not suppressed.
The test coverage table is embedded in both reports; it is opened only for
final reporting and never changes training. It also states that probability
quality is conditional on valid proposal sets and separately reports the full
pre-proposal proposal/choice coverage. `diagnostic-probabilities` and
`diagnostic-branch-matrix` intentionally accept unverified JSONL for debugging;
their reports are explicitly non-formal and must never populate a paper table.

Even the formal branch-matrix estimate reuses already collected candidate
outcomes. The paper's policy result must still come from new, held-out
closed-loop episodes and be reported with reset/split-group bootstrap
intervals.

## Baselines and required diagnostics

All policies use the same candidate set, executor, serializer, continuation,
and maximum horizon. An OPEN branch may stop at 100 steps only when the frozen
continuation proposer returns no valid DIRECT candidate:

- frozen VLM candidate order;
- always DIRECT, using the lowest frozen-Qwen rank among feasible DIRECT
  candidates;
- always OPEN, using the lowest frozen-Qwen rank among feasible OPEN
  candidates and falling back to the lowest-rank DIRECT only if no OPEN exists;
- uniform random valid candidate;
- one GroundedOutcomeModel;
- a deep ensemble of independently trained copies.

Required ablations are full-image support instead of the candidate region,
removing history, removing task text, and shuffling descriptor-to-region
binding. The first two are the initial priority.

Probability quality is reported with NLL, Brier score, ECE, and reliability
bin data; a separate plotting step may render those bins.
Decision quality is reported with held-out final task success. Efficiency is
reported with information-action frequency, control steps, VLM calls, VLA
calls, latency, and proposal coverage.

## Interpretation and failure localization

Method V1 can support the intended claim only if it outperforms the same
executor and candidate set without learning a fixed primitive prior. Useful
diagnoses are:

- manual candidates work but Qwen candidates fail: proposal/grounding problem;
- well-calibrated outcomes but no policy gain: candidate coverage or ranking
  problem;
- OPEN dominates every state: missing information-sufficient negative cases;
- OPEN helps physically but semantic masking has no effect: physical
  affordance selection, not demonstrated information use;
- serialized candidate identity is preserved but the wrong object is touched:
  Stage-1-to-VLA referent interface problem; and
- low training NLL with poor unseen layouts: split leakage or representation
  shift, not successful uncertainty modelling.

The first empirical milestone is one real, sealed branch through the complete
Qwen-to-MolmoAct2-to-LIBERO path. The first scientific milestone is a
held-out, reset-group-paired advantage over the fixed baselines.
