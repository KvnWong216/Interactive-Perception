# Prompt-conditioned Interaction Uncertainty (PIU)

We study which prompt-relevant information a frozen VLA is missing, where that
uncertainty is located, and which executable interaction is expected to reduce
it before the task is resumed.

## Method

The base policy is the public `pi05_libero` checkpoint. Its weights and action
decoder remain frozen.

```text
public RGB + prompt + history
  → structured task-fact belief + node uncertainty
  → typed hard-valid candidates
  → learned action-effect and future-belief distributions
  → explicit expected information/task utility
  → semantic subtask → frozen π0.5
  → six public observations → conformal outcome → belief update → replan
```

Policy inputs are the stock LIBERO `agentview`, wrist RGB, robot state, and
prompt. Simulator joints, segmentation, hidden poses, task predicates, and BEV
images are evaluator-only.

The current executable development path uses Grounding DINO + SAM + DINOv2 to
build public object proposals, frozen SigLIP to form the pre-VLM
prompt-conditioned belief and uncertainty field, and Qwen2.5-VL only to predict
registered action effects and produce a grounded semantic subtask. A
deterministic expected-utility selector chooses among `ACT`, `MOVE_CLOSER`,
`NEXT_BEST_VIEW`, `OPEN_CONTAINER`, `REMOVE_OCCLUDER`, and controlled `STOP`.
There is no online confidence threshold. The selected physical subtask is
executed by the frozen stock `pi05_libero` dual-camera policy.

Agentview and wrist observations are fused as complementary fields of view:
a singleton reveal in either camera is positive evidence, while one camera's
`NOT_REVEALED` is not counterevidence against the other camera. Local `EMPTY`
requires singleton-negative target evidence from both cameras plus singleton
completed search coverage. This v13 composition is development-only until it
passes a fresh clean validation; it does not alter the frozen v12b audit.

We define action-resolvable uncertainty as the expected task risk removed by
an information action after its cost and measured failure rate are included:

\[
U_{\mathrm{res}}(a\mid b)=V_{\mathrm{stop}}(b)-
\left[c(a)+\mathbb{E}_{y}V(b^{a,y})\right].
\]

PIU V0 freezes the π0.5 PaliGemma prefix as its prompt-conditioned visual
encoder and trains small belief, effect, outcome, future-belief, and ranking
heads. Its current `target_location` domain is `visible_workspace`,
`middle_drawer`, `other_unsearched_region`, and `absent`; only the first two
have V0 initial-belief training data. Control memory is deterministic
`(searched_regions, attempt_counts, public outcomes)`. No confidence threshold
decides whether to open the drawer.

## Current evidence

| Test | Result |
|---|---:|
| Stock π0.5 reproduction | 100/100 |
| T01 drawer opening (joint only) | 97/100 |
| Hidden-target belief | 97/100 |
| Same RGB, visible-target prompt | 100/100 |
| Same prompt, target already visible | 99/100 |
| Target-observability risk routing | 296/300 |
| Temporal outcome critic v8 | NOT-GO: FAILED 7/7, REVEALED 7/9, EMPTY 4/5 |
| Patch-resolvable v9 clean development | NOT-GO: critic passed; EMPTY physical 36/40, lower 0.786 |
| v10 fresh clean development | NOT-GO on 1400–1439: 118/120 singleton correct; one false EMPTY |
| v11 three-head RGB cascade model selection | 120/120 on contaminated 1400–1439; non-claim |
| v11 fresh clean development | NOT-GO on 1440–1479: 118/120 singleton correct; two false REVEALED |
| v12b fresh clean development | **GO** on 1900–1939: 120/120 truth retained; 119/120 singleton; zero false singleton EMPTY/REVEALED |
| v12b sealed outcome audit | **GO** on 900–999: 299/300 truth retained; 294/300 singleton; zero false singleton EMPTY/REVEALED |
| v4 OPEN_AND_OBSERVE disposable smoke | 3/3: REVEALED / EMPTY / FAILED |
| PIU V0 learned-head diagnostics | belief 27/30; outcome 20/21 contaminated diagnostic |
| PIU V0 + v12b end-to-end disposable behavior | 5/5 routes; 3/3 singleton outcomes; non-paper |
| PIU V1 object-sidecar calibration | location/action argmax 77/80; conformal truth retained 80/80; development only |
| PIU V1 scene-disjoint clean | **NOT-GO**: visible/hidden truth retained 24/40 and 40/40; 6 false singleton routes |
| PIU V2 development refit | calibration 80/80; new clean block 1740–1759 frozen but unopened |
| Object-level PIU mechanism loop | disposable seed 1399: singleton OPEN then singleton REVEALED; information 1/1; final task 0/1 |
| Qwen PIU + fresh π0.5, original cluttered scenario | disposable seed 1399: OPEN_CONTAINER → REVEALED → MOVE_CLOSER; information 1/1; final task not executed |
| Original-prompt physical continuation | information 1/1; final task 0/1; non-paper diagnostic |
| Final retrieval after reveal | 0/5 |

The PIU smoke covers hidden butter, a same-RGB visible cream-cheese prompt,
already-visible butter, local EMPTY, and a constructed FAILED control. Hidden
butter routes to real π0.5 `OPEN_AND_OBSERVE`, obtains singleton `REVEALED`,
updates the public belief, and replans to `DIRECT_ACT`.
EMPTY excludes only the middle drawer and FAILED preserves the hypothesis.
In a separate one-case diagnostic, `DIRECT_ACT` physically continued for a
fixed 400 steps under the original prompt. Information acquisition succeeded,
but butter placement failed. This is an information-loop success and a final-
task failure, never a combined system success.

The v7 outcome head is invalidated. It checked only the final frame: debug seed
770 saw the target during opening and lost it after return. Version 3 therefore
checks all six public history points and requires same-camera-pose visibility
coverage before calling a result `EMPTY`. Its 180-example development set is
now frozen. The v8 head failed development and the sealed audit remains
untouched.

Version 3 also exposed a second evaluator problem: true middle-layer reveals
occupy at least 311 target pixels, while three supposed reveals in the empty
layer contain only 5--7 leaked pixels from the upper layer. Version 9 therefore
uses one pi0.5 visual-token footprint (256 pixels) as an architecture-derived
resolvability definition. The v9 critic passed its class gates on seeds
660--699, but the old endpoint incorrectly required the drawer to remain open
and the wrist to return to one exact pose; the resulting EMPTY physical branch
was only 36/40 (one-sided 95% lower 0.786), so v9 is NOT-GO.

Version 10 separates two questions. A per-frame public-agentview RGB head with
a temporal OR asks whether the butter was ever prompt-resolvable. Only when it
returns singleton `NOT_REVEALED` does a second public-history head classify the
observation effect as `FAILED` or `COMPLETED`; `COMPLETED` then means local
`EMPTY`. This preserves acquired information after later reclosure or
reocclusion. The frozen combination is 120/120 on now-contaminated seeds
660--699, which is model-selection evidence only. On fresh seeds 1400--1439,
v10 retained FAILED 40/41, REVEALED 39/40, and EMPTY 39/39, but produced one
false singleton EMPTY and is therefore NOT-GO. Those seeds were then consumed
to diagnose two distinct causes: a wrist-only reveal and an uncertified search
view.

Version 11 uses an agentview target head, a high-precision wrist positive
rescue, and an independent public-agentview RGB coverage head. Its deterministic
cascade was 120/120 on the already-contaminated 1400--1439 model-selection
block, but fresh 1440--1479 failed with two false singleton REVEALED outputs.
That block was then consumed for composition-rule diagnosis.

Version 12b freezes a stricter evidence rule before opening new data: only a
singleton wrist-positive versus singleton agentview-negative conflict is
ambiguous; a multi-label wrist set is abstention rather than negative evidence.
On fresh clean seeds 1900--1939, FAILED and REVEALED are 40/40 singleton each;
EMPTY retains 40/40 correct labels and is singleton-correct 39/40 (lower
0.887). Mean set size is 1.008, one case SAFE_STOPs, and both physical
information endpoints are 40/40 (lower 0.928). The frozen one-time sealed audit
on 900--999 then obtained FAILED 100/100 singleton, REVEALED 100/100 singleton,
and EMPTY 99/100 coverage with 94/100 singleton-correct (lower 0.885). It has
zero false singleton EMPTY/REVEALED, mean set size 1.017, and 5/300 ambiguous
SAFE_STOPs. Physical REVEALED and local EMPTY are both 100/100 (lower 0.970),
which also passes the original 0.90 standard. Complete motor return is 199/200
and is reported separately from information acquisition. Six-frame history
retains 100/100 reveals versus 96/100 for final-frame-only.

Because frozen π0.5 itself varies substantially across LIBERO seeds, the
canonical same-state mechanism loop is the immediate engineering priority and
cross-seed executor robustness is a secondary characterization. This priority
does not turn same-seed smoke into held-out evidence or relax the clean/sealed
gate definitions.

The first object-level PIU sidecar used public RGB, public robot state, the full
prompt, Grounding DINO/SAM proposals, DINOv2 region features, and a frozen π0.5
prefix. Its first clean scene-disjoint block failed primarily on unseen visible
assets: it produced 19 SAFE_STOPs and six false singleton routes. That block is
now development-only. V2 adds those failures to development training without
adding an online signal, action type, or oracle input. Its original calibration
is 80/80, but that is not a clean result. The replacement V2 clean protocol is
frozen on seeds 1740–1759 and remains unopened at this commit; the PIU sealed
audit remains closed.

A disposable same-environment mechanism run now connects the public object
frontend and frozen V2 sidecar to the real `OPEN_AND_OBSERVE` option. Given the
original prompt, it produced singleton `closed_container` and
`OPEN_TO_INSPECT`, selected the registered middle-drawer option by explicit
utility, obtained singleton `REVEALED` from the six-frame v12b RGB critic, and
replanned to direct action. Information acquisition succeeded (1/1), while the
subsequent original-prompt manipulation failed (0/1). This establishes wiring
for one registered option, not held-out performance, final-task success, or
validated exact drawer-layer localization.

The latest disposable run uses the original cluttered
`T01D_hidden_butter_retrieval` scene rather than the simplified scene. Starting
from the complete prompt `Place the butter in the basket`, the corrected Qwen
pipeline selected `OPEN_CONTAINER`; a freshly started frozen π0.5 server then
sampled and executed `OPEN_AND_OBSERVE`; the six-frame complementary public-RGB
critic returned singleton `REVEALED`; and the updated Qwen pipeline replanned to
`MOVE_CLOSER`. Evaluator-only replay measured 303 agentview pixels during the
option and 818 wrist pixels after return. These values were read only after the
controller and critic terminated. This proves one development information-loop
execution, not final placement, clean generalization, or v13 calibration.

The action reliability lower bound is 0.80; the original 0.90 result is always
reported. Conformal error is 0.05. These are evidence gates, not online
confidence triggers. Open-drawer retrieval remains 0/5, so target evidence is
not reported as final task success.

| Milestone | Decision |
|---|---:|
| PIU V0 wiring / inference smoke | GO (non-claim) |
| T01 v12b outcome clean development | GO |
| T01 v12b outcome sealed audit | GO |
| Scene-disjoint PIU belief/effect development | V1 NOT-GO; V2 clean unopened |
| RSS method paper | NOT-GO |
| Final product | NOT-GO |

`PARTIAL` always counts as `NOT-GO`.

## Seed and data provenance

The single machine-readable authority is
[`benchmarks/rss_v1/seed_registry.yaml`](benchmarks/rss_v1/seed_registry.yaml).
Historical manifests remain immutable, but their old “future audit” text does
not override this registry.

| Seeds | Current role | State |
|---|---|---|
| 90–189 | legacy prompt router / joint-open test | consumed; not RGB reveal evidence |
| 220–269 | prompt-state train/calibration/development | consumed |
| 280–379 | prompt-state audit + precision extension | consumed |
| 400–459 | legacy action-effect development | consumed |
| 500–599 | obsolete v1 audit allocation | quarantined; not reused |
| 600–619 | outcome prototype train | consumed |
| 620–652 | conformal calibration | consumed |
| 653–659 | v8/v9 diagnosis | contaminated; diagnostic only |
| 660–699 | v9 clean development; later v10 diagnosis | consumed, NOT-GO, contaminated |
| 700–799 | final-frame-bug debugging | whole block quarantined |
| 800–899 | later closed-loop validation | frozen; audit-gated; non-paper |
| 900–999 | one-time v12b sealed outcome audit | consumed, GO, immutable |
| 1300, 1399 | wiring smoke | disposable; never claim-bearing |
| 1400–1439 | v10 clean development; v11 diagnosis/model selection | consumed, NOT-GO, contaminated |
| 1440–1479 | v11 clean; then v12b model selection | consumed, NOT-GO, contaminated |
| 1500–1519 | PIU V1 prototype train | consumed |
| 1520–1599 | PIU prototype reserve | untouched |
| 1600–1619 | PIU V1 conformal calibration | consumed |
| 1620–1699 | PIU calibration reserve | untouched |
| 1700–1719 | invalid first clean protocol | quarantined; target visibly leaked |
| 1720–1739 | PIU V1 clean NOT-GO; then V2 development data | consumed, contaminated |
| 1740–1759 | PIU V2 replacement clean development | frozen, authorized, unopened |
| 1760–1799 | PIU clean-development reserve | untouched |
| 1800–1899 | future scene-disjoint sealed audit | sealed |
| 1900–1939 | v12b fresh clean development | consumed, GO, immutable |

The three-class collection and label contract is frozen in
[`outcome_data_protocol_v1.yaml`](benchmarks/rss_v1/outcome_data_protocol_v1.yaml).

## Reproduce

Reproduce the current original-scene development loop with a new immutable run
identifier:

```bash
EXPERIMENT_GPU_INDEX=0 EXPERIMENT_ALLOW_LOCAL_RUSTDESK=1 \
  bash scripts/run_piu_original_demo.sh my_run_id
```

The recorded run, exact commands, inputs, limitations, and zero-online-oracle
contract are summarized in
[`PIU_ORIGINAL_FRESH_DEMO.md`](docs/PIU_ORIGINAL_FRESH_DEMO.md). Its public
[Demo](results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition.mp4),
[contact sheet](results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition_contact_sheet.png),
and [machine-readable trace](results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition_trace.json)
are development assets and must not be reported as final-task success.

Run the dependency-gated experiment:

```bash
bash scripts/run_rss_experiment_ladder.sh preflight
bash scripts/run_rss_experiment_ladder.sh smoke
bash scripts/run_rss_experiment_ladder.sh development
bash scripts/run_t01_open_and_observe_pipeline.sh v12b-clean
bash scripts/run_t01_open_and_observe_pipeline.sh v12b-audit
```

Run the local PIU V0 path:

```bash
../.conda/envs/ipu/bin/python scripts/build_piu_v0_dataset.py
../.conda/envs/ipu/bin/python scripts/train_piu_v0.py --help
../.conda/envs/ipu/bin/python scripts/run_piu_v0_smoke.py --help
```

The latest recorded five-case behavior run used
`EXPERIMENT_GPU_INDEX=0 bash scripts/run_piu_v0_smoke.sh --output
results/smoke/piu_v0_v12b_full_pipeline_v1_seed1399.json --asset-dir
results/assets/piu_v0_v12b_full_pipeline_v1_seed1399/raw`.

Its post-terminal presentation assets are indexed by
[`assets_manifest.json`](results/assets/piu_v0_v12b_full_pipeline_v1_seed1399/visualizations_v1/assets_manifest.json).
The left demo panel is replayed wrist RGB; the right global panel is explicitly
evaluator-only and was never available to the controller.

The matching machine-readable privileged-input audit is
[`piu_v0_v12b_full_pipeline_v1_privileged_input_audit.json`](results/audits/piu_v0_v12b_full_pipeline_v1_privileged_input_audit.json).

The original-prompt physical continuation is recorded in
[`piu_v0_v12b_physical_act_v1_seed1399.json`](results/smoke/piu_v0_v12b_physical_act_v1_seed1399.json)
with its [combined demo](results/assets/piu_v0_v12b_physical_act_v1_seed1399/visualizations_v1/piu_v0_full_pipeline_seed1399.mp4)
and [separate privileged-input audit](results/audits/piu_v0_v12b_physical_act_v1_privileged_input_audit.json).

The object-level disposable mechanism trace is
[`piu_object_t01_seed1399_v1.json`](results/mechanism/piu_object_t01_seed1399_v1.json).
Its [asset manifest](results/assets/piu_object_t01_seed1399_v1/visualizations_v1/assets_manifest.json)
indexes the wrist/evaluator replay, public-history storyboard, belief update,
action utility, effect forecast, and uncertainty visualizations. The matching
[privileged-input audit](results/audits/piu_object_t01_seed1399_v1_privileged_input_audit.json)
passes with zero controller oracle inputs. The global replay is evaluator-only
and all visualizations were rendered after controller termination.

This PC has one physical GPU at index 0. The generic preflight rejects unknown
compute processes and permits the current-user RustDesk process only when
explicitly enabled. `LAB_SERVER_MODE=1` still hard-requires physical GPU1 and
the lab Server User Guide.

The v12b sealed audit was run once after development passed. Re-running it is
forbidden; the historical exact command required explicit authorization:

```bash
ALLOW_SEALED_AUDIT=1 bash scripts/run_rss_experiment_ladder.sh audit
```

See [method](docs/RSS_METHOD_V1.md), [RSS gates](benchmarks/rss_v1/gates.yaml),
[experiment](benchmarks/rss_v1/experiment_v1.yaml),
[PIU training protocol](docs/PIU_TRAINING_V0.md),
[PIU data protocol](benchmarks/piu_v0/data_protocol.yaml),
[PIU main matrix](benchmarks/piu_v0/main_experiment_matrix.yaml),
[PIU strict gates](benchmarks/piu_v0/gates.yaml),
[PIU code ownership](benchmarks/piu_v0/code_ownership.yaml),
[seed/data provenance registry](benchmarks/rss_v1/seed_registry.yaml),
[threshold register](docs/THRESHOLD_REGISTER.md), and
[temporal-label failure](results/T01_TEMPORAL_LABEL_BUG.md).
