# RSS 2027 method redesign: prompt-conditioned latent interaction routing

## Status

This document is a design freeze candidate, not an empirical claim.

- Base revision: `origin/main@e987c680a71a140a4108a696202d3ace5b9a3795`.
- Working branch: `rss2027/latent-router`.
- The historical Heuristic V0, S02 certificate, and S03 v1--v3 artifacts remain immutable.
- No result in this document implies that the new router has been trained, calibrated,
  or evaluated.
- The primary online-policy boundary remains prompt-conditioned interactive
  manipulation and object information enrichment. Active viewpoint change is out of
  scope.

## 1. Decision

The RSS 2027 line will not continue the online chain

```text
Grounding DINO -> SAM -> DINOv2/SigLIP
-> hand-designed uncertainty factors
-> Qwen JSON labels -> manual utility -> rule selector -> pi0.5
```

and it will not continue the later chain

```text
five explicit belief heads -> six explicit effect heads
-> several conformal sets -> typed singleton rules -> box-to-text bridge.
```

The new method will use one learned, token-level decision stage and one frozen
continuous-control stage:

```text
public RGB history + user prompt + public action history
                         |
                  frozen VLM tokens
                         |
        prompt-conditioned evidence queries
                         |
      +------------------+------------------+
      |                  |                  |
   DIRECT              OPEN              ROTATE ...
      |                  |                  |
 predicted post-interaction evidence for each primitive
      +------------------+------------------+
                         |
             unified route + grounding decoder
                         |
              calibrated primitive set Gamma
                         |
                 singleton only: subtask
                         |
                  frozen MolmoAct2
                         |
               action chunk -> new RGB -> loop
```

The internal representation is implicit. The statistically testable uncertainty
interface is the calibrated primitive set, not the norm of a latent token, an attention
heatmap, or a VLM's verbal confidence.

## 2. Research question and falsifiable hypothesis

### Research question

Given a user's task, current and recent first-person observations, and the public action
history, can a robot predict which physically executable interaction will provide the
missing task-relevant evidence, while avoiding unnecessary exploration when the current
observation is already sufficient?

### Core hypothesis

A prompt-conditioned latent predictor trained on matched pre-action/action/post-action
transitions will route information-seeking primitives more accurately and generalize
better than:

1. direct frozen-VLA execution;
2. prompted VLM reasoning;
3. scalar entropy or self-confidence thresholds;
4. the repository's explicit Heuristic V0 and factorized-effect router; and
5. a route-only token decoder without future-evidence prediction.

The hypothesis is rejected if removing the future-evidence objective does not reduce
information-action selection, downstream task success, or object/layout/prompt OOD
generalization.

## 3. Novelty boundary after the September 2026 audit

The paper must not claim any of the following in isolation:

- first token-level uncertainty mechanism;
- first latent belief in a VLA;
- first active-perception VLA;
- first uncertainty-to-action system;
- first latent future prediction for robot actions;
- first multi-skill information-gain planner; or
- first uncertainty-guided VLA action selection.

Those spaces are already occupied by, among others:

- [CoMe-VLA](https://arxiv.org/abs/2602.04600), which learns a cognitive
  token/head and temporal memories for autonomous subtask transitions;
- [UAOR](https://arxiv.org/abs/2602.18020) and
  [SCALE](https://arxiv.org/abs/2602.04208), which use internal action uncertainty
  to alter observation use or action decoding;
- [LaWAM](https://arxiv.org/abs/2606.15768),
  [VLA-JEPA](https://arxiv.org/abs/2602.10098), and
  [StageWAM](https://arxiv.org/abs/2608.10780), which predict future or stage-level
  latent representations for robot control;
- [FabriMAE](https://arxiv.org/abs/2608.16697), which derives action-generation
  reliability from internal attention and uses it for branch selection; and
- [MS-MEM](https://arxiv.org/abs/2609.02493), which already compares viewpoint,
  push, and grasp using a common uncertainty- and disturbance-aware information
  objective.

The defensible contribution is the conjunction:

> prompt-conditioned latent evidence, counterfactual comparison across physically
> executable information primitives, calibrated set-valued routing, and execution by
> a frozen open VLA, evaluated with a benchmark that causally separates information
> need, primitive choice, information effect, and motor execution.

The closest architectural sources are deliberately acknowledged rather than hidden:

- [BLIP-2](https://arxiv.org/abs/2301.12597) and
  [Perceiver IO](https://arxiv.org/abs/2107.14795) motivate a small learned query set
  over a large frozen multimodal token field;
- [DETR](https://arxiv.org/abs/2005.12872) motivates parallel learned output queries
  instead of a hand-authored serial proposal pipeline;
- CNABU and [Dengler et al.](https://arxiv.org/abs/2506.02286) motivate predicting
  an action-conditioned post-interaction belief before choosing an information action;
- VLA-JEPA motivates a future-target branch that is visible during training only and
  cannot leak into deployment input;
- [KnowNo](https://proceedings.mlr.press/v229/ren23a.html) motivates converting
  uncalibrated model scores into a set-valued decision interface; and
- [MolmoAct2](https://arxiv.org/abs/2605.02881) supplies the primary frozen
  flow-matching executor and a stronger direct-execution baseline than pi0.5.

## 4. Minimal notation

The main text should use only the following symbols.

| Symbol | Definition | Shape or type |
| --- | --- | --- |
| `q` | original user instruction | text |
| `o_{<=t}` | current and recent public RGB observations, optionally public proprioception | sequence |
| `h_{<t}` | public executed-action and observation history | sequence |
| `H_t` | full frozen-VLM multimodal token field | `N x D_vlm` |
| `B_t` | learned prompt-conditioned evidence tokens | `Q x D` |
| `a_k` | one registered primitive, such as DIRECT, OPEN, REMOVE, ROTATE, BRING-CLOSE, STOP | discrete |
| `Bhat_{t+1}^k` | predicted evidence tokens after executing `a_k` | `Q x D` |
| `s_t^k` | unified route logit for primitive `a_k` | scalar |
| `g_t^k` | grounding distribution over current visual patches | vector |
| `Gamma_alpha(x_t)` | split-conformal primitive prediction set | set |
| `tau_t` | structured natural-language subtask sent to the executor | text |
| `A_{t:t+L}` | continuous action chunk returned by the frozen VLA | array |

There is intentionally no single online scalar `u`. A scalar entropy may be reported
as a diagnostic, but it does not control the robot.

## 5. What uncertainty means here

A latent named `belief` or `uncertainty` is not automatically probabilistic. `B_t` is
therefore described conservatively as a **task-conditioned evidence representation**.
It earns an uncertainty-related interpretation only if it supports all four tests:

1. **Prompt causality:** keeping the image fixed while changing the requested target or
   relation changes the appropriate primitive.
2. **Perceptual intervention:** keeping the prompt fixed while revealing the target
   reduces calibrated ambiguity and changes the route from an information action to
   DIRECT.
3. **Action-effect prediction:** `Bhat_{t+1}^k` agrees with the evidence encoding of the
   real post-action observation for the executed primitive.
4. **Language invariance and sensitivity:** paraphrases preserve the decision, whereas
   changing the target or spatial relation changes it.

Four failure sources must remain conceptually separate:

| Source | Meaning | Appropriate response |
| --- | --- | --- |
| task-information uncertainty | the observation lacks evidence that a physical interaction can reveal | choose a positive-value information primitive |
| instruction ambiguity | the prompt itself admits multiple goals | ask or stop; do not disturb the scene by default |
| Stage-1 epistemic/OOD uncertainty | the router has not learned this object, layout, or phrasing | abstain or collect data |
| executor uncertainty | the VLA may fail to realize an already selected primitive | stop, replan, or collect motor data |

Only the first source is the paper's primary target. Flow-VLA disagreement and internal
action-reliability methods are Stage-2 diagnostics, not substitutes for prompt-relevant
information uncertainty.

## 6. Stage 1: latent interaction router

### 6.1 Frozen multimodal token field

The policy-visible input is

```text
x_t = (q, o_{<=t}, h_{<t}).
```

The frozen VLM produces

```text
H_t = E_VLM(x_t).
```

`H_t` retains every valid prompt and visual patch token, plus public camera, time, and
patch-position metadata. It does not pool the input into four global vectors. The online
path does not require Grounding DINO, SAM, DINOv2, SigLIP, an object list, a semantic
map, or simulator state.

### 6.2 Evidence queries

A lightweight query transformer extracts a small set of evidence tokens:

```text
B_t = Q_theta(Q_evidence, H_t).
```

The prompt is already in the key/value field, so the same image can produce different
evidence for different tasks. Learned queries are a representation mechanism, not a
probability claim.

### 6.3 Primitive-conditioned future evidence

Each registered primitive has an encoded capability description. All primitives query
the same evidence state in parallel:

```text
Bhat_{t+1}^k = P_theta(B_t, Emb(a_k)).
```

During training, an EMA/stop-gradient target encoder processes the actual post-action
observation:

```text
Bbar_{t+1} = stopgrad(Q_target(E_VLM(x_{t+1}))).
```

Only the primitive that was actually executed receives a future-target loss. If the
simulator is reset to the same initial state and several primitives are executed, those
records form a counterfactual group; no unexecuted future is fabricated.

This is narrower than a general world model. It predicts the prompt-conditioned evidence
needed for interaction routing, not a future RGB frame and not the whole scene state.

### 6.4 Unified route and grounding

One shared decoder reads current evidence, primitive identity, and predicted future
evidence:

```text
r_t^k = D_theta(Emb(a_k), [B_t ; Bhat_{t+1}^k])
s_t^k = Linear(r_t^k)
g_t^k = Pointer(r_t^k, current_visual_patches(H_t)).
```

The decoder learns which predicted evidence changes matter for the prompt. There is no
online six-factor effect vector, uncertainty-weight sum, Qwen adjective-to-number map,
or second selector. Grounding points only to current visual tokens and is learned from
training labels that remain outside the policy-visible input.

### 6.5 Structured subtask output

The Stage-1 output is a constrained record:

```text
primitive: one registered capability
referent: an open-vocabulary referring expression
grounding: current-camera point or region when supported
```

The natural-language executor instruction is serialized from this record. The
serializer is an interface contract, not an uncertainty model. DIRECT passes the
original task; STOP does not call the executor.

## 7. Training objective

The first implementation uses three losses only:

```text
L = L_route + lambda_J * L_JEPA + lambda_G * L_ground.
```

### Route loss

For an admissible primitive set `Y_b`, use multi-positive likelihood:

```text
L_route = -mean_b log sum_{k in Y_b} softmax(s_b)_k.
```

`Y_b` is constructed from a frozen task contract and real counterfactual rollouts. It
must not be inferred from the model's own confidence.

### Future-evidence loss

For the actually executed primitive `a_b`:

```text
L_JEPA = mean_{b,q} [1 - cosine(Bhat_{t+1}^{a_b,q}, Bbar_{t+1}^{q})].
```

The target branch is stop-gradient. The mandatory `lambda_J = 0` ablation determines
whether future-evidence prediction contributes anything beyond ordinary routing.

### Grounding loss

For a set of valid target patches `M_{b,k}`:

```text
L_ground = -log sum_{p in M_{b,k}} softmax(g_b^k)_p.
```

Masks or simulator geometry may generate training/evaluation labels, but they are never
tokenized as online inputs.

Additional effect heads, EDL, entropy penalties, semantic-entropy sampling, an explicit
VOI regressor, and ensemble disagreement are baselines or later ablations, not part of
the minimal method.

## 8. The decision-theoretic bridge

The theory uses task-conditioned value of information to define the desired behavior,
not as another hand-coded runtime score. Let `R_q(B_t)` be the Bayes risk of making the
task decision implied by prompt `q`. The information value of primitive `a` is

```text
VOI_q(a | x_t)
  = R_q(B_t) - E[R_q(B_{t+1}) | x_t, do(a)].
```

Only under log loss is this exactly conditional mutual information. Under task-success,
wrong-object, or safety loss, it is expected Bayes-risk reduction. The benchmark uses
counterfactual outcomes to define admissible actions; the neural router learns the
ranking directly instead of manually estimating and weighting identity, occlusion,
resolution, cost, and risk terms at inference.

DIRECT, information primitives, and STOP inhabit the same route space:

- DIRECT wins when current evidence is sufficient for the task;
- OPEN, REMOVE, ROTATE, or BRING-CLOSE win only when their observed counterfactual
  consequences improve the task decision;
- STOP wins when no admissible physical action can resolve the remaining issue.

## 9. Calibration and abstention

After all trainable parameters are frozen, a separate calibration split computes a
split-conformal primitive set. With canonical action label `y_i` and route probability
`p_i`:

```text
R_i = 1 - p_i(y_i)
qhat = the ceil((n + 1) * (1 - alpha))-th ordered calibration score
Gamma_alpha(x) = {k : 1 - p(k | x) <= qhat}.
```

Under exchangeability, the claim is marginal inclusion of the canonical acceptable
primitive at the requested coverage. It is not a per-trial success guarantee, a task
success guarantee, or protection against arbitrary distribution shift.

The decision rule has exactly one branch:

```text
|Gamma| == 1 -> execute the sole primitive
|Gamma| != 1 -> abstain
```

There is no post-conformal utility selector. ID and object/layout/prompt OOD coverage,
mean set size, singleton rate, abstention, and selective risk are reported separately.

## 10. Stage 2: frozen executor boundary

The primary executor is the released MolmoAct2-LIBERO policy. Its stock inference
contract accepts camera images, instruction text, and proprioceptive state and returns a
continuous action chunk. It does not expose a documented input for arbitrary external
belief tokens or K/V tensors.

Therefore the first valid implementation is:

```text
Stage-1 router -> constrained natural-language subtask
stock frozen MolmoAct2 -> continuous action chunk.
```

If a future experiment projects evidence tokens into MolmoAct2's per-layer K/V pathway,
that system must be named a trained token/KV adapter over a frozen backbone. It is not
the stock frozen executor and is not the primary claim.

The existing pi0.5 OPEN certificate (`122/124`) remains valid only for its frozen
pi0.5 checkpoint, serializer, prompt, and middle-drawer context. It cannot be transferred
to MolmoAct2. Every MolmoAct2 primitive used by the paper requires a new executor-specific
qualification.

## 11. Benchmark V1

### 11.1 Required causal axes

Each scene family should expose matched variants across four axes:

1. **information need:** clear, hidden, partially occluded, label-resolution, wrong
   orientation, and exhausted/not-found;
2. **prompt dependence:** the same physical scene paired with prompts that require
   DIRECT, an information primitive, or STOP;
3. **action effect:** reset-identical branches for valid, irrelevant, failed, and
   unnecessarily disruptive primitives; and
4. **executor competence:** route correctness, primitive execution, post-action evidence,
   and terminal task success scored separately.

Core scenarios should include drawers, refrigerators/cabinets, inverted containers,
partial clutter, bring-close label reading, rotation for label exposure, clear direct
execution, and exhaustive not-found. `BRING_CLOSE` moves an object for inspection and is
not active camera viewpoint planning.

### 11.2 Dataset record

Each immutable record contains a public and private half:

```text
public:
  prompt, RGB history, public state, public action history, registered primitives

private evaluator/training labels:
  executed primitive, admissible primitive set, grounding region,
  execution success, post-action evidence outcome, downstream task outcome
```

Private fields may supervise training and score evaluation but must fail the online-input
firewall before tokenization.

### 11.3 Splits

Use initial-state-group-disjoint train, development, calibration, and sealed-test splits.
Report additional held-out object, layout, and prompt-composition splits. Prompt
paraphrases of one scene group must never be scattered across train and test.

## 12. Main baselines and ablations

All routing baselines should use the same frozen executor wherever their interface permits.

### Baselines

1. frozen MolmoAct2 direct execution;
2. prompted VLM/chain-of-thought router, representing ZS-IP-style reasoning;
3. CoMe-VLA-style binary sufficiency token/head;
4. action softmax entropy and VLM semantic entropy;
5. KnowNo-style conformal action set without learned future evidence;
6. EDL Dirichlet route head;
7. two-head or lightweight-adapter ensemble disagreement;
8. Heuristic V0 from this repository;
9. the current explicit factorized-effect router;
10. CNABU/Dengler/MS-MEM-inspired explicit-map specialist, labelled as a reference
    rather than a directly comparable baseline if faithful porting is impossible.

Flow-VLA velocity-field disagreement and FabriMAE/SAFE-style signals are executor-risk
baselines. They must not be presented as measures of task-information uncertainty.

### Mandatory ablations

- no prompt;
- no history;
- route-only (`lambda_J = 0`);
- no current-vs-predicted-future pairing;
- no counterfactual grouping;
- no grounding supervision;
- no conformal set;
- scalar entropy in place of latent evidence;
- pi0.5 versus MolmoAct2 executor, with separate qualification; and
- text-only hard interface versus any later trained soft/KV adapter.

## 13. Primary metrics

The paper must not collapse the pipeline into one success number.

| Layer | Primary measurements |
| --- | --- |
| information need | DIRECT / information-action / STOP routing, unnecessary-exploration rate |
| primitive choice | top-1, macro F1, NDCG or ranking regret over counterfactual actions |
| information effect | pre/post task-risk reduction, target evidence acquired, empty/exhausted conclusion |
| grounding | current-patch recall, point/region coverage, wrong-object selection |
| calibration | marginal/conditional coverage, mean set size, singleton rate, abstention, selective risk |
| execution | primitive success, wrong contact, disturbance, action steps and latency |
| full loop | terminal task success and calibrated failure decomposition |

The same-image/different-prompt test is the most important differentiator from global
map uncertainty. Global map uncertainty is identical, while task-relevant information
value changes with the prompt.

## 14. Repository migration

### Preserve

- full frozen-VLM patch/prompt token extraction;
- camera, time, patch-coordinate, and public-history metadata;
- public/private input firewall;
- group-disjoint split and immutable artifact machinery;
- calibration/statistical test ideas;
- primitive qualification protocol; and
- all historical S02 evidence.

### Freeze as baselines or negative evidence

- `src/interaction_uncertainty/` and its Grounding-DINO/SAM/SigLIP/Qwen chain;
- `src/calibrated_interaction/` route/effect pilot;
- the present five-belief/six-effect successor;
- object-sidecar checkpoints;
- pi0.5-specific box-to-English bridge; and
- S03 v1--v3 execution trees and diagnostics.

### Do not continue

- fixed target phrases or object/container priors;
- SAM-area resolution uncertainty;
- fixed uncertainty weights and temporal smoothing;
- adjective-to-number mappings;
- manual expected-utility selection;
- route-after-route singleton rule chains;
- `NEXT_BEST_VIEW`; or
- another S03 amendment for the RSS method.

The new active code lives in `src/latent_interaction/` and must not import legacy method
packages. Historical artifacts remain readable and immutable.

## 15. Go/no-go gates

1. **Representation gate:** full tokens and prompt/history perturbation tests pass; no
   privileged field reaches the token provider.
2. **JEPA toy gate:** on pre/post OPEN, ROTATE, and REMOVE pairs, the executed-action
   future prediction beats a current-state and shuffled-action baseline.
3. **Routing gate:** the full model beats route-only on held-out scene groups. Failure
   here removes the JEPA contribution.
4. **Prompt-causality gate:** matched same-image/different-prompt routing is materially
   above prompted-VLM and global-map baselines.
5. **Calibration gate:** ID coverage reaches its registered target without unusable set
   size; OOD degradation is reported rather than hidden.
6. **Executor gate:** every claimed MolmoAct2 primitive is separately qualified. The old
   pi0.5 OPEN result is not reused.
7. **Closed-loop gate:** improvement survives decomposition into route, grounding,
   execution, information effect, and final task success.

## 16. Immediate order of work

1. land contracts, a synthetic CPU router, losses, conformal set, and frozen-executor
   protocols without any empirical claim;
2. export public pre/post records into a new immutable schema without importing legacy
   method code at training time;
3. implement and validate a real frozen-VLM full-token provider;
4. run the smallest route-only versus future-evidence toy experiment;
5. qualify MolmoAct2 OPEN, DIRECT/PICK, and PLACE under version-pinned instructions;
6. expand counterfactual branches and prompt-paired BenchV1 records;
7. freeze train/development/calibration/test manifests before main evaluation; and
8. run the main study only after the toy and executor gates pass.

## 17. Reviewer-killing tests

The strongest likely review is: “this is only a high-level action classifier with a
conformal wrapper.” The answer must be empirical, not rhetorical:

- future-evidence prediction must improve route accuracy or OOD generalization over the
  identical route-only model;
- predicted primitive effects must agree with real post-action evidence;
- swapping the primitive token while holding the scene fixed must change the predicted
  future appropriately;
- swapping only the prompt must change information value without changing global scene
  uncertainty; and
- gains must persist with MolmoAct2 while motor failures are reported separately.

If these tests fail, the honest paper is a benchmark and negative-result study, not a
new uncertainty model.

## 18. Backbone selection is an experiment, not a hidden assumption

The method is defined over frozen multimodal tokens rather than a specific commercial
VLM. The first controlled screen should compare only two Stage-1 providers while keeping
the learned router, data, and frozen executor identical:

1. a compact Qwen3-VL checkpoint, because CoMe-VLA establishes a directly relevant
   embodied-use precedent and its token interface is practical for adapter training; and
2. the released Molmo embodied-reasoning VLM, because it is architecturally aligned with
   MolmoAct2 and is a strong open spatial-reasoning candidate.

The selection criteria are held-out prompt-paired routing, grounding, latency, peak
memory, and deterministic access to full visual and language hidden states. The paper
must report the screen rather than choosing a backbone from reputation. The frozen
MolmoAct2 executor remains fixed during this comparison.

## 19. Dated RSS 2027 work programme

The official submission date must be replaced when the RSS 2027 call for papers is
published. Until then, use 18 January 2027 as an internal freeze date, leaving a buffer
before the usual late-January submission period.

| Dates | Deliverable | Go/no-go result |
| --- | --- | --- |
| 7--20 September 2026 | contracts, synthetic router, losses, conformal wrapper, frozen-executor boundary | focused unit tests pass; no legacy import |
| 21 September--11 October | frozen full-token providers and immutable public-transition export | prompt/history perturbations affect evidence; private fields fail closed |
| 12 October--1 November | OPEN/ROTATE/REMOVE paired toy set and route-only versus JEPA screen | future prediction beats current-state and shuffled-action controls |
| 2--22 November | MolmoAct2-LIBERO adapter and primitive-specific qualification | only qualified primitives enter the closed loop |
| 23 November--13 December | BenchV1 prompt pairs, counterfactual branches, and group-disjoint split freeze | no scene/prompt leakage; data card and hashes complete |
| 14--27 December | main baselines, calibration, OOD evaluation, and mandatory ablations | primary hypothesis passes or the method claim is reduced |
| 28 December--10 January 2027 | closed-loop runs, demos, figures, and failure taxonomy | route, information effect, execution, and task success remain separable |
| 11--18 January | sealed-test opening, statistical audit, paper freeze | no post-test tuning; claim table is evidence-bound |

## 20. Expected research assets

The redesign is complete only when the repository contains all of the following, with
versioned identities and immutable manifests:

- the method contract and a paper-level architecture figure;
- a frozen Stage-1 token-provider identity and trainable-router checkpoint;
- public/private BenchV1 schemas, a data card, group-disjoint split manifests, and prompt
  pair/counterfactual-group identifiers;
- pre-action/action/post-action transition data for every claimed primitive;
- route-only, future-evidence, prompted-VLM, entropy, EDL, KnowNo, Heuristic V0, and
  explicit-factor baseline results;
- independent conformal calibration artifacts and ID/OOD risk--coverage reports;
- executor-specific qualification certificates for MolmoAct2;
- decomposed closed-loop results and synchronized policy-view/evaluator-view videos; and
- a claim ledger mapping each abstract/table sentence to a frozen result artifact.

At the present scaffold stage, only the first method contract and synthetic interfaces
exist. There is no Stage-1 checkpoint, MolmoAct2 adapter, BenchV1 dataset, calibrated
router, or new empirical result yet.
