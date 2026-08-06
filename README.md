# Interactive Perception

Minimal LIBERO scenarios for studying **when a robot should manipulate the
environment to obtain task-relevant information before acting**.

## Motivation

Vision-language-action policies are usually evaluated in scenes where the
target is already visible and semantically identifiable. Household
manipulation is often different: the instruction can be unambiguous while the
observation is not. A target may be inside a closed container, under a cover,
partly occluded by clutter, too small to identify, oriented away from the
camera, or absent altogether.

This project isolates that gap. The policy receives an ordinary final-goal
instruction without being told whether exploration is necessary. It must
eventually choose between direct task execution and information-producing
manipulations such as opening, clearing, bringing an object closer, rotating it
to expose a label, or terminating with `NOT_FOUND`.

### Why viewpoint change is permitted, not forbidden

Version 0.2 forbade active viewpoint change by fiat. That made the benchmark
unfalsifiable: a reader could always object that the robot merely needed to look
from somewhere else, and the benchmark had no answer.

Version 0.3 permits viewpoint change and discharges the objection with evidence
instead. `scripts/certify_nbv_insufficiency.py` sweeps a hemisphere of camera
poses around the workspace, renders instance segmentation from each, and records
the best target visibility any viewpoint achieves without touching the scene. It
then applies the oracle manipulation and sweeps again. A scene earns
`NBV_INSUFFICIENT` when no viewpoint reveals the target but manipulation does.

Scenes that fail that certificate are kept as controls rather than deleted. The
contrast between *needs a better view* and *needs a different world* is the
distinction this benchmark exists to measure, so both sides of it have to be
present. Closed containers and covers are expected to certify as
`NBV_INSUFFICIENT`; partial clutter occlusion and low resolution are expected to
certify as `NBV_SUFFICIENT`.

## Current status

### New in v0.3

- viewpoint change permitted and certified rather than forbidden
  (`scripts/certify_nbv_insufficiency.py`);
- a four-rung prompt ladder per scenario -- `implicit`, `hinted`, `explicit`,
  `capability` -- all terminating in the same task goal;
- rollout tooling for π0.5 (`pi05_libero`) over openpi's websocket policy
  server, with a reproduction gate that checks the wiring before any challenge
  result is trusted;
- per-step uncertainty instrumentation: repeated sampling on a frozen
  observation, converted into subjective-logic evidence
  (vacuity / dissonance / predictive entropy / Dirichlet mutual information);
- decoding of continuous actions into the coarse action space
  `A = {ACT, NOT_FOUND, ROTATE, MOVE_CLOSER, REMOVE_OCCLUDER}`;
- graded metrics that stay interpretable when success rate is zero;
- figures and demo videos with the uncertainty curve composited over the
  policy view.

**Not yet run.** The v0.3 code paths are written but have not been executed
end-to-end: this machine has no GPU and no simulator install. Treat every script
below as unverified until the reproduction gate passes on your hardware.

### From v0.2

This release contains scene definitions and oracle endpoint validators, not a
trained policy. We have completed:

- six Interactive Manipulation scenarios covering containers, removable
  covers, visible controls, target absence, and partial clutter occlusion;
- two controlled Information Enrichment scenarios isolating label resolution
  and label-surface orientation;
- final-goal-only prompts that do not disclose the required information action;
- deterministic seeds and benchmark metadata;
- evaluator-only segmentation/state checks for validating the intended
  pre-action and post-action observations;
- a natural oblique wrist-camera reset for the Information Enrichment branch.

The clean repository snapshot was regression-tested on the configuration below:

```text
Interactive Manipulation: 18 / 18 cases passed (6 tasks × 3 seeds)
Information Enrichment:     6 /  6 cases passed (2 tasks × 3 seeds)
```

## Scenario inventory

| ID | Final-goal instruction | Hidden difficulty | Expected primitive | Expected NBV verdict |
|---|---|---|---|---|
| `T01_drawer_retrieval` | Place the butter in the wicker basket. | Target is in a closed drawer | `REMOVE_OCCLUDER` | `NBV_INSUFFICIENT` |
| `T02_fridge_retrieval` | Place the orange juice in the wicker basket. | Target is in a closed refrigerator | `REMOVE_OCCLUDER` | `NBV_INSUFFICIENT` |
| `T03_inverted_bowl_retrieval` | Place the butter in the wicker basket. | Target is covered by an inverted bowl | `REMOVE_OCCLUDER` | `NBV_INSUFFICIENT` |
| `T04_visible_direct` | Place the alphabet soup can in the wicker basket. | Target is already clear and identifiable | `ACT` | `NBV_SUFFICIENT` |
| `T05_exhaustive_not_found` | Place the alphabet soup can in the wicker basket. | Target is absent; drawers must be exhausted | `NOT_FOUND` | n/a |
| `T06_dense_clutter_partial_occlusion` | Place the ketchup in the wicker basket. | Target is partially occluded by dense clutter | `REMOVE_OCCLUDER` | `NBV_SUFFICIENT` |
| `IE02_resolution_only` | Place the package labeled Macaroni and Cheese in the wicker basket. | Correct label surface is visible but too small | `MOVE_CLOSER` | `NBV_SUFFICIENT` |
| `IE03_orientation_only` | Place the bottle labeled Tomato Ketchup in the wicker basket. | Candidate is close, but its label faces away | `ROTATE` | `NBV_INSUFFICIENT` |

The expected behavior is benchmark metadata, not part of the language
instruction given to a policy. The expected NBV verdict is design intent; the
certificate produced by the certifier is the measurement, and where the two
disagree the certificate wins.

Two scenarios carry special roles and should be included in almost every run:

- **`T04_visible_direct` is the uncertainty reference.** A flat vacuity trace on
  a hidden-target scene proves nothing by itself. It becomes a result only
  against a scene where the evidence genuinely was available. The plotting
  script refuses to draw a challenge curve without it.
- **`T05_exhaustive_not_found` is the abstention probe.** It is the only scene
  whose correct answer is to stop, and therefore the only place where a policy's
  inability to decline can be measured rather than assumed.

### Coarse action space

All methods are scored in one shared vocabulary:

```text
A = { ACT, NOT_FOUND, ROTATE, MOVE_CLOSER, REMOVE_OCCLUDER }
```

Members are separated by *what information the action produces*, not by how it
is executed. Container opening therefore folds into `REMOVE_OCCLUDER`: a drawer
front is an occluder that happens to be mounted on a joint. The finer execution
vocabulary (`PULL_DRAWER`, `UNCOVER`, `PUSH_ASIDE`, …) is retained in
`interaction_uncertainty.v2.primitives.PrimitiveKind` and projects onto `A` via
`to_coarse`.

`NOT_FOUND` has no decoding from continuous actions. That is a finding, not a
gap: a policy with no abstention channel cannot abstain, so its evidence for
that member is structurally zero and is plotted as an empty bar.

## Prompt ladder

Each scenario carries four prompts that all terminate in the **same** task goal
and differ only in how much of the information step the language gives away:

| Rung | Variant | Example (`T01`) |
|---|---|---|
| L0 | `implicit` | Place the butter in the wicker basket. |
| L1 | `hinted` | Place the butter in the wicker basket. You may need to interact with the scene before the task is possible. |
| L2 | `explicit` | Open the top drawer, then place the butter in the wicker basket. |
| L3 | `capability` | Open the top drawer of the wooden cabinet. |

Goal identity is the point. Comparing *"place the butter in the basket"* against
*"find the butter in the drawer"* would compare two different tasks and measure
nothing.

**`capability` versus `implicit` is the load-bearing comparison.** If the policy
performs the information step when instructed and never performs it otherwise,
the failure is a decision failure. If it cannot perform the step even when
instructed, the scenario is measuring a missing skill and cannot support any
claim about information seeking.

That distinction is not hypothetical here. `pi05_libero` is trained on
`libero_spatial`, `libero_object`, `libero_goal` and `libero_10`. Drawer opening
appears in `libero_goal` and `libero_10`; refrigerators do not appear in any of
them. `T02_fridge_retrieval` is therefore marked `primary_condition: false` and
carries an explicit `skill_confound` field. Any claim drawn from it must report
its `capability` result alongside.

## How the benchmark is constructed

Each task has two components:

1. a LIBERO BDDL file defining the language instruction, objects, regions,
   initial predicates, and task-success predicates;
2. a benchmark YAML entry defining the target, observation contract, expected
   information endpoint, seeds, and validation thresholds.

The construction protocol is deliberately simple:

1. write a normal final-task prompt without words such as *search*, *open*,
   *inspect*, *rotate*, or *move closer*;
2. create one controlled source of uncertainty in the initial state;
3. define an oracle information endpoint, such as an opened drawer, cleared
   visibility corridor, closer package, or label-facing orientation;
4. render the initial and endpoint observations under fixed seeds;
5. use instance segmentation and simulator state only in the evaluator to
   verify that the endpoint causally improves the intended information;
6. keep these oracle signals out of the policy observation.

The validators may directly write simulator state to check endpoint validity.
Those writes are **not policy actions** and do not claim that low-level robot
control has already been solved. A learned policy must eventually reach the
same endpoints through Panda `OSC_POSE` actions.

## Repository layout

```text
Interactive-Perception/
├── README.md
├── pyproject.toml
├── requirements-macos-sim.txt
├── requirements-linux-gpu.txt
├── benchmarks/
│   ├── interactive_manipulation_v0/benchmark.yaml
│   └── information_enrichment_v0/benchmark.yaml
├── scenarios/
│   ├── interactive_manipulation_v0/       # T01–T06
│   └── information_enrichment_v0/         # IE02–IE03
├── src/interactive_perception/
│   ├── policy_client.py                   # openpi websocket client + CPU stub
│   ├── anchors.py                         # evaluator-private scene readout
│   ├── primitive_decoder.py               # continuous actions -> coarse space A
│   ├── metrics.py                         # graded outcomes, bootstrap CIs
│   └── rollout.py                         # episode loop + uncertainty probe
├── tests/
└── scripts/
    ├── setup_libero_config.py
    ├── check_install.py
    ├── list_scene_handles.py              # verify anchor refs resolve
    ├── certify_nbv_insufficiency.py       # viewpoint-sufficiency certificates
    ├── download_pi05_libero.py            # mirror the public checkpoint
    ├── serve_pi05.sh                      # GPU policy server
    ├── run_repro_gate.py                  # wiring check on stock LIBERO
    ├── run_challenge_rollout.py           # prompt ladder on the challenge scenes
    ├── plot_uncertainty_curves.py         # figures + demo video overlay
    ├── run_interactive_manipulation_v0.py
    ├── validate_interactive_manipulation_v0.py
    ├── run_information_enrichment_v0.py
    └── validate_information_enrichment_v0.py
```

Generated images, reports, environments, caches, and the LIBERO checkout are
gitignored.

## Uncertainty instrumentation

A feed-forward VLA exposes no belief, so the question this benchmark asks --
*does the policy register that it lacks the evidence it needs?* -- cannot be put
to it directly. The instrument is repeated sampling.

openpi's policy server splits its PRNG on every `infer` call, so querying the
**same frozen observation** `n` times draws `n` independent flow-matching
samples. The spread of those chunks over task-relevant anchors is an observable
proxy for what the policy has committed to.
`interaction_uncertainty.v2.vla_bridge` converts that spread into
subjective-logic evidence, which makes vacuity, dissonance, predictive entropy
and Dirichlet mutual information computable for a baseline that reports none of
them.

Three properties matter when reading the resulting curves.

- **Evidence total is not the sample count.** Each sample contributes evidence
  proportional to how decisively it moves. Were evidence simply a count,
  `S = n + K` would be constant and vacuity would carry no information about the
  observation at all.
- **Anchors are evaluator-private.** Anchor positions come from simulator state
  and are used to *read* the policy, never to inform it. Nothing in
  `anchors.py` may reach a policy observation.
- **The belief is a diagnostic, not the policy's confidence.** Low vacuity means
  the policy is committed, not that it is correct. The hypothesis under test is
  precisely that a monolithic VLA reports low vacuity while holding no evidence.

Probing dominates cost: a probe step is `probe_samples` inferences instead of
one. `probe_every` trades curve resolution against wall-clock and should be set
from the GPU budget. The first sample of each probe is the chunk that gets
executed, so probing changes what is measured, not what is done.

## Evaluating π0.5

`pi05_libero` is the openpi checkpoint reported at 98.8 / 98.2 / 98.0 / 92.4 on
LIBERO spatial / object / goal / 10.

### 1. Fetch the checkpoint (~12.4 GB, no credentials needed)

```bash
python scripts/download_pi05_libero.py --dest ../checkpoints
python scripts/download_pi05_libero.py --dest ../checkpoints --verify-only
```

### 2. Serve it on the GPU host

```bash
scripts/serve_pi05.sh ../openpi ../checkpoints/checkpoints/pi05_libero
```

The client is pure CPU, so the simulator and the model may live on different
machines; point the runners at the server with `--host/--port`.

### 3. Pass the reproduction gate before anything else

```bash
python scripts/run_repro_gate.py --strict
```

This runs one suite at reduced trial count. Its purpose is to catch a silent
wiring bug -- a missing 180-degree image rotation, a mis-ordered state vector, a
stale checkpoint -- which would depress success on the challenge scenes and be
indistinguishable from the failure the benchmark is trying to measure. **A
challenge-scenario result obtained before this gate passes is not evidence of
anything.**

### 4. Certify viewpoint sufficiency

```bash
python scripts/certify_nbv_insufficiency.py --strict
```

### 5. Run the prompt ladder

```bash
python scripts/run_challenge_rollout.py \
  --task-ids T01_drawer_retrieval T04_visible_direct T05_exhaustive_not_found \
  --variants implicit explicit capability \
  --seeds 0 1 2 \
  --save-frames
```

`--policy stub` swaps in a CPU stand-in that exercises the loop, trace schema,
metrics and plots without a model server. Stub output is a plumbing check and
must never be reported as a result.

### 6. Draw the figures and demo video

```bash
python scripts/plot_uncertainty_curves.py --figures --metric vacuity
python scripts/plot_uncertainty_curves.py --video \
  --task-id T01_drawer_retrieval --variant implicit --seed 0
```

Before running any of this against a new or edited scene, confirm the anchor
references resolve:

```bash
python scripts/list_scene_handles.py --task-ids T02_fridge_retrieval
```

### Reporting

Success rate alone is close to useless on these scenarios: a policy that never
opens the drawer and one that opens it and fumbles the grasp both score zero,
yet only one supports the claim under test. Every episode therefore also records
`information_endpoint_reached` -- whether the needed information ever became
visible, whatever the policy did next -- alongside premature-commit and
terminal-decision rates. Confidence intervals bootstrap over episodes, never
over adjacent frames within an episode.

## Environment setup

### Tested configuration

- macOS on Apple silicon (tested on M2);
- Python `3.10`;
- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- MuJoCo `2.3.7`;
- robosuite `1.4.0`;
- Panda with `OSC_POSE`;
- off-screen rendering through `MUJOCO_GL=cgl`.

The scenario files are platform-independent, but the provided dependency lock
and full regression commands have currently been validated only on native
Apple silicon. For Linux/RTX systems, start from the official
[LIBERO installation](https://github.com/Lifelong-Robot-Learning/LIBERO) and
reuse the `scenarios` and `benchmarks` directories.

### 1. Clone this repository and the pinned LIBERO source

```zsh
git clone https://github.com/KvnWong216/Interactive-Perception.git
cd Interactive-Perception

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  third_party/LIBERO
git -C third_party/LIBERO checkout \
  8f1084e3132a39270c3a13ebe37270a43ece2a01
```

### 2. Create the simulation environment

The tested setup uses a repository-local Conda environment:

```zsh
conda create \
  --prefix "$PWD/.conda/envs/libero-mac" \
  python=3.10 \
  -y

conda activate "$PWD/.conda/envs/libero-mac"
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-sim.txt
```

### 3. Activate and verify

```zsh
source scripts/activate_env.sh
python scripts/check_install.py
```

The activation script:

- activates the repository-local environment;
- adds the pinned LIBERO source to `PYTHONPATH`;
- selects the macOS CGL renderer;
- creates repository-local cache/output directories;
- generates `.libero/config.yaml` with paths for the current clone.

A successful check ends with:

```text
OK: imports, architecture, and benchmark registry are ready.
```

## Reproduce the existing scenarios

Run all six Interactive Manipulation tasks under seeds `0, 1, 2`:

```zsh
source scripts/activate_env.sh

python scripts/run_interactive_manipulation_v0.py \
  --seeds 0 1 2 \
  --jobs 3 \
  --strict
```

The command creates:

```text
outputs/interactive_manipulation_v0/
```

Run the two Information Enrichment tasks:

```zsh
python scripts/run_information_enrichment_v0.py \
  --seeds 0 1 2 \
  --strict
```

The command creates:

```text
outputs/information_enrichment_v0/
```

Run one task and one seed:

```zsh
python scripts/run_interactive_manipulation_v0.py \
  --task-ids T01_drawer_retrieval \
  --seeds 0 \
  --jobs 1 \
  --strict

python scripts/run_information_enrichment_v0.py \
  --task-ids IE03_orientation_only \
  --seeds 0 \
  --strict
```

Each case directory contains RGB observations, evaluator-only masks where
applicable, and a JSON report describing the initial state, oracle information
endpoint, camera contract, and pass/fail checks.

## Render or inspect one raw BDDL scene

Off-screen render:

```zsh
python scripts/render_scenario.py \
  --bddl scenarios/interactive_manipulation_v0/T04_visible_direct.bddl \
  --target alphabet_soup_1 \
  --seed 0 \
  --output outputs/T04_single_render
```

Open a MuJoCo window:

```zsh
python scripts/launch_gui.py \
  --bddl scenarios/interactive_manipulation_v0/T04_visible_direct.bddl \
  --camera agentview \
  --observe-only
```

For `T03`, `T06`, `IE02`, and `IE03`, use the benchmark runners for canonical
reproduction because their validators apply calibrated reset wrappers or
information endpoints that are not encoded solely in the raw BDDL file.

## Observation and action boundary

- `T01–T06` retain the frozen fixed `agentview` protocol used for the first
  benchmark branch.
- `IE02–IE03` use `robot0_eye_in_hand` as the policy camera and `agentview`
  only for evaluation.
- The wrist branch begins from a deterministic, approximately 30-degree
  downward oblique sensing pose.
- Manipulating or holding an object may naturally alter what the wrist camera
  sees; this is treated as manipulation-driven information acquisition.
- Viewpoint change is no longer prohibited. Scenes are instead certified for
  whether any camera pose could substitute for manipulation, and a method that
  solves an `NBV_SUFFICIENT` scene by looking from elsewhere has solved it
  legitimately. The claim the benchmark defends is narrower and stronger than a
  prohibition: on `NBV_INSUFFICIENT` scenes, no viewpoint suffices.
- Evaluator-private state -- object poses, instance segmentation, joint angles,
  scenario metadata -- may be read by the evaluator and by the uncertainty
  instrumentation, and may never enter a policy observation.

## Next milestone

The instruction-conditioned belief model over:

```text
target identity
target location
target existence
target observability
grasp feasibility
action outcome
```

For each candidate action, the model should predict the post-action belief and
rank the coarse action space `A` using task-relevant expected information gain,
expected task progress, action cost, and risk. Typed contracts, the belief
filter, and the candidate proposer live in the companion repository
[Interaction-Uncertainty](https://github.com/KvnWong216/Interaction-Uncertainty);
this repository supplies the scenes, the rollout loop, and the baseline
measurements those components have to beat.

The measurement path is already shared: `vla_bridge.py` in that repository turns
a baseline VLA's action samples into the same `TaskBelief` the planner consumes,
so the baseline and the method are scored by identical code rather than by two
implementations that happen to agree.

## Primary methodological references

- [Map Space Belief Prediction for Manipulation-Enhanced Mapping, RSS 2025](https://www.roboticsproceedings.org/rss21/p039.html)
- [Efficient Manipulation-Enhanced Semantic Mapping With Uncertainty-Informed Action Selection, 2025](https://arxiv.org/abs/2506.02286)
- [Interactive Learning of Physical Object Properties Through Robot Manipulation, 2024](https://arxiv.org/abs/2404.07344)

## Acknowledgment

The environments are built on
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[MuJoCo](https://mujoco.org/). This repository does not vendor LIBERO assets or
source code.
