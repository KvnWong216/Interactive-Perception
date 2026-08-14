# Handoff

## 2026-08-14 G4-v1 calibration

The owner froze alpha=0.1 and minimum executor reliability=0.9. A new
oracle-free LIBERO dataset contains 100 observations and 800 independent π0.5
action chunks for the binary intent scope `ACT` vs `REMOVE_OCCLUDER`. Seeds
0–19 per class learn prototypes, 20–39 calibrate, and 40–49 are held-out
validation. G4-v1 reaches 20/20 coverage with mean set size 1.0. Dataset hash:
`6873f44e46c903ebecf2b5aa10b8f91ff740dea3629e189f3ac81e1a2dd5db86`.

Do not broaden this result: three primitives remain uncalibrated, and the
separate executor gate still fails because T01 drawer opening is only 3/5
(one-sided 95% lower bound 0.189, required reliability 0.9). Calibration-only
scenes and samples are permanently excluded from the paper test set. The
policy server was stopped after collection and GPU model memory was released.

## 2026-08-12 pure-policy debug result

The wrapper confound has been removed from the capability test. The new
`scripts/run_pure_pi05_scenario_sr.py` uses `OffScreenRenderEnv` and the stock
openpi observation/action contract; only the BDDL scene and instruction change.
On T01 (five seeds), direct middle-layer opening reaches the joint endpoint in
3/5, explicit search-and-retrieve succeeds in 0/5 (drawer endpoint 1/5), and
final-goal-only retrieval succeeds in 0/5. Stock drawer opening remains 5/5.

Conclusion: earlier failures included prompt/geometry/context and evaluation
path defects, but the corrected policy is still context- and
composition-sensitive. Say "consistent with shortcut dependence or
overfitting", not "proves π0.5 is overfit". G4 calibration is NOT-GO because no
frozen public VLM or held-out calibration split exists. Do not publish router
numbers until that changes. The policy server was stopped after the run and
GPU memory was released. See `results/PURE_PI05_DEBUG.md` and the two new T01
demos in `results/demos/`.

G4 now has an executable replacement for the raw VLM-confidence threshold:
`src/interactive_perception/semantic_conformal.py` and
`scripts/calibrate_semantic_intents.py` conformalize the distribution over
decoded action intents. The CLI requires an explicit policy ID, split ID, and
at least 30 held-out examples. Do not lower that requirement merely to pass the
gate; collect a disjoint calibration split instead.
Online intent decoding must also filter out every `role: task_target` anchor
before resolution. Existing trace-time decoding is evaluator-only and cannot
be copied into the controller unchanged without leaking the hidden pose.

## Update after the 8269d65 evidence snapshot

Pipeline v0.4 now keeps pi0.5 on stock `agentview` and uses the horizontal pose
only for temporarily rendered demo frames. It adds three executable arms:
`monolithic`, `fixed-rule`, and `uncertainty-router`. The last arm calls an
RGB-only public-evidence service, emits bounded primitive sub-prompts, observes
again after execution, updates searched/cleared state, and can terminate with
`NOT_FOUND` or a low-confidence safe stop. See `docs/PIPELINE_V04.md`.

The original T01 and T06 were replaced. `T01_multi_drawer_search` has three
candidate drawers with the highest prior deliberately assigned to the wrong
drawer; its 360-pose certificate is `NBV_INSUFFICIENT` (0 -> 1235 px).
`T06_severe_clutter_occlusion` has nine objects and three declared occluders;
its certificate is `NBV_SUFFICIENT` (2060 -> 2064 px). Both replacements pass
scene validation on seeds 0, 1, and 2.

No new GPU result exists yet. Before interpreting one, freeze and record the
public vision model/checkpoint, run the reproduction gate, and require T04 to
recover under the stock policy camera. The RGB evidence service must never be
implemented with simulator segmentation or `task_target` anchors.

**Calibration blocker:** the current closed-loop defaults (`confidence=0.5`,
initial absence mass `0.05`, false-absence loss `4.0`, clutter split
`1.9/0.1`, and the fixed-rule 120-step prefix) were introduced to test control
flow. They are not justified method parameters. Do not run or publish the main
three-arm experiment until they are estimated on a disjoint calibration set,
replaced by task-measured costs, or covered by a preregistered sensitivity
analysis. A smoke test may be used only to validate wiring.

**Corrected drawer reference:** direct inspection of LIBERO's
`open_the_middle_drawer_of_the_cabinet` task places the cabinet at x=0.02--0.04,
y=-0.25---0.23 and uses the instruction "Open the middle layer of the drawer".
The earlier handoff statement that stock geometry was y=-0.30 was too broad and
is not valid for this capability control.

Paste the block at the bottom into a fresh agent session. Everything above it
is the evidence behind that block, kept so the claims can be checked rather
than trusted.

---

## What this project is testing

Whether a monolithic VLA (π0.5), given an ambiguous prompt on a scene where the
target is hidden, *elects* to take the information-gathering action. The claim
under test is a **decision** failure, not a skill gap, which is why every
scenario carries a `capability` prompt rung that names the information step
outright. If the policy fails even when instructed, the scenario measures a
skill gap and can support no claim about information seeking.

## Repositories

| Repo | Path | Branch | Head | Role |
|---|---|---|---|---|
| Interactive-Perception | `~/InteractivePerception_yg/Interactive-Perception` | `main` | `8269d65` | scenes, rollout, metrics, experiment runner |
| Interaction-Uncertainty | `~/InteractivePerception_yg/Interaction-Uncertainty` | `main` | `8205e47` | the method: belief, router, primitives |

Environment: one conda env at `~/InteractivePerception_yg/.conda/envs/ipu`,
pinned in `env/`. The policy server runs in openpi's own `uv` venv at
`~/InteractivePerception_yg/openpi/.venv`. See `env/README.md`.

**Every command needs `env -u PYTHONPATH`** — a ROS install on `PYTHONPATH`
shadows the conda packages. LIBERO is vendored at `third_party/LIBERO` and put
on `sys.path` by `scripts/_bootstrap.py`; `import libero` failing in a bare
interpreter is expected.

## Running it

```bash
bash scripts/run_full_experiment.sh                    # full sweep
QUICK=1 bash scripts/run_full_experiment.sh            # smoke test
TASKS="T01_drawer_retrieval" SEEDS="0 1 2" MAX_STEPS=500 \
  bash scripts/run_full_experiment.sh                  # subset
```

Seven stages; stage 4 is a hard gate (reproduction on stock `libero_object`,
last measured **100/100** against a published 98.2%). If it fails, nothing
downstream is evidence of anything. `scripts/collect_results.py --clean` then
`scripts/format_results_tables.py` curate `outputs/` into tracked `results/`.

## State of the evidence

**The claim is not currently supported, and the reason is measured.**

Last run (T01, T06, plus T04 as control; 5 seeds; 500 steps):

| Scene | success | endpoint | note |
|---|---|---|---|
| T04 visible (control) | ~35% | 100% | **was 100% before the camera change** |
| T01 drawer | 0% | 0% | all four rungs, including `capability` |
| T06 clutter | 0% | 0–40% | `explicit` reaches 40% |

`capability - implicit` on endpoint rate: **+0.000, CI [0, 0]**.

T04 sees the target (endpoint 100%) and cannot grasp it. That is a visuomotor
mapping tied to the training camera, so the 20.6° camera pose is
off-distribution for π0.5 and **T01/T06's zeros cannot be read as a decision
failure**. The control is the only reason this is known.

## Bugs already found — do not rediscover these

1. **Reset wrappers were never applied in the rollout.** Only the validators and
   the NBV certifier called them, so T03 ran with the bowl *beside* the butter
   and T06 with no clutter. Both scored endpoint 1.00 under every prompt and
   looked like easy successes. Fixed: `run_episode` applies them, and a
   declared-but-unregistered wrapper is now a hard failure.
2. **`motion_scale` unit error.** LIBERO `OSC_POSE` commands are normalised to
   `[-1,1]`, not metres. At 0.05 every sample saturated, vacuity became a
   constant of the sampling budget, and the curve still plotted. Now 6.0, with
   `saturated_fraction` reported per episode.
3. **Vacuity is `W/S`, not `K/S`.** Only equal under one pseudo-count per
   category; otherwise `u` exceeds 1.
4. **Scene geometry was off-distribution.** The cabinet sat at yaw 2.66–2.72,
   y=−0.17; stock LIBERO uses yaw=π, y=−0.30. Fixed, and T01 now certifies
   `NBV_INSUFFICIENT` at **pre=0** — zero recoverable pixels from 360 poses.
5. **Stock LIBERO has no fridge task**, so T02 is off-distribution by
   construction; its numbers say nothing about decision-making.
6. **The NBV certifier once resolved a missing fixture to a hardcoded centre**,
   which would have fabricated certificates. It raises now.
7. **T04's four prompt rungs are textually identical**, so it contributes a
   constant zero to the paired contrast and dilutes it. Exclude it from
   `capability_minus_implicit` or give it a single prompt.

## The method

`interaction_uncertainty/v2/target_belief.py` holds a Dirichlet over *where the
target is*. Hypotheses are typed by how evidence could ever be obtained:

`OBSERVED` · `VIEWPOINT_BLOCKED` · `MANIPULATION_ONLY` · `ABSENT`

    alpha_k = W a_k + omega_k e_k [k is OBSERVED]
    u = W / S

Splitting hidden mass three ways is what lets one belief choose among moving
closer, opening a container, and declining. `select_action` compares three
expected losses (commit / decline / pay for information) and returns the risks
alongside the choice.

Two modelling errors found by running it: `ABSENT` accepted no evidence, so
`NOT_FOUND` degenerated into a timeout; and once it did, the router declared
absence with 34% of mass behind an unopened door. Absence evidence is now
scaled by how exhaustive the search actually was.

**Anchor discipline:** benchmark anchors with `role: task_target` resolve to the
hidden object's pose. `build_hypotheses` refuses that role and never calls the
resolver for it. Violating this invalidates every downstream result.

## Open decisions

1. **Decouple the cameras.** Policy should consume stock `agentview`
   (in-distribution); demos and uncertainty curves should render from the
   horizontal pose. Not yet implemented; needs the owner's sign-off because it
   changes what the policy sees.
2. **The benchmark may not need uncertainty.** A fixed rule — "target not
   visible → interact with the highest-prior container; visible → grasp" —
   solves five of six scenes. Until an uncertainty-blind ablation (arm C) is
   run and lost, the method's contribution is unproven. Extend the benchmark
   with multi-container ambiguity and graded visibility, where ordering and
   sufficiency actually matter.
3. **Closed loop is unbuilt.** `EpisodeController` exists with tests but has
   never run against the live policy. Missing: an `ActionOutcomeCritic`, a
   primitive→sub-prompt executor on the frozen π0.5, and a runner.

---

## Prompt for a fresh agent

```
You are continuing a robotics research project in ~/InteractivePerception_yg.
Read Interactive-Perception/HANDOFF.md first; it is current as of
Interactive-Perception 8269d65 and Interaction-Uncertainty 8205e47, both on main.

Context in one paragraph: we are testing whether pi0.5, given an ambiguous
prompt on a scene whose target is hidden, elects to take the information-
gathering action. Six LIBERO-derived scenes; a four-rung prompt ladder ending
in a `capability` rung that names the information step outright. The claim is
currently NOT supported, and the reason is measured rather than guessed: a
camera-pose change put the policy off-distribution, and the T04 visible-target
control fell from 100% to ~35% success while still seeing the target.

Working rules, which are not negotiable:
- One conda env at ~/InteractivePerception_yg/.conda/envs/ipu. Work only inside
  ~/InteractivePerception_yg. Prefix every command with `env -u PYTHONPATH`.
- Never let the controller read anchors with `role: task_target`; that is the
  hidden object's pose and reading it invalidates all results.
- The reproduction gate (stage 4, currently 100/100) is a hard gate. If it
  fails, no downstream number is evidence.
- Any scene change requires re-running scene validation AND the NBV
  certification, because certificates must describe the configuration actually
  evaluated. This has been violated once and voided 40 episodes.
- Report negative results plainly. The most valuable output so far is a control
  that invalidated a run.

Next task, in order of value:
1. Decouple the policy camera (stock `agentview`) from the demo camera
   (horizontal, pos [0.80,0,1.20], quat [0.5813,0.4025,0.4025,0.5813]), then
   re-run T01/T06/T04 and check T04 returns to ~100%.
2. Build the uncertainty-blind ablation (arm C). If the method does not beat a
   fixed "not visible -> open highest-prior container" rule, say so; the
   benchmark then needs multi-container ambiguity before any method claim.
3. Close the loop: wire interaction_uncertainty.v2.target_belief's router to
   the live policy via a primitive->sub-prompt executor.

Ask before spending more than an hour of GPU time on a run.
```
