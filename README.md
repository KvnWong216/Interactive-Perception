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
to expose a label, or terminating with `NOT_FOUND`. Deliberate active
viewpoint-change actions are outside the current scope.

## Current status

This first release contains scene definitions and oracle endpoint validators,
not a trained policy. We have completed:

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

| ID | Final-goal instruction | Hidden difficulty | Expected high-level behavior |
|---|---|---|---|
| `T01_drawer_retrieval` | Place the butter in the wicker basket. | Target is in a closed drawer | `OPEN_CONTAINER` |
| `T02_fridge_retrieval` | Place the orange juice in the wicker basket. | Target is in a closed refrigerator | `OPEN_CONTAINER` |
| `T03_inverted_bowl_retrieval` | Place the butter in the wicker basket. | Target is covered by an inverted bowl | `CLEAR_OCCLUDER` |
| `T04_visible_direct` | Place the alphabet soup can in the wicker basket. | Target is already clear and identifiable | `DIRECT_ACT` |
| `T05_exhaustive_not_found` | Place the alphabet soup can in the wicker basket. | Target is absent; drawers must be exhausted | `NOT_FOUND` |
| `T06_dense_clutter_partial_occlusion` | Place the ketchup in the wicker basket. | Target is partially occluded by dense clutter | `CLEAR_OCCLUDER` |
| `IE02_resolution_only` | Place the package labeled Macaroni and Cheese in the wicker basket. | Correct label surface is visible but too small | `BRING_CLOSER` |
| `IE03_orientation_only` | Place the bottle labeled Tomato Ketchup in the wicker basket. | Candidate is close, but its label faces away | `ROTATE_TO_LABEL` |

The expected behavior is benchmark metadata, not part of the language
instruction given to a policy.

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
├── requirements-macos-sim.txt
├── benchmarks/
│   ├── interactive_manipulation_v0/benchmark.yaml
│   └── information_enrichment_v0/benchmark.yaml
├── scenarios/
│   ├── interactive_manipulation_v0/       # T01–T06
│   └── information_enrichment_v0/         # IE02–IE03
└── scripts/
    ├── setup_libero_config.py
    ├── check_install.py
    ├── run_interactive_manipulation_v0.py
    ├── validate_interactive_manipulation_v0.py
    ├── run_information_enrichment_v0.py
    └── validate_information_enrichment_v0.py
```

Generated images, reports, environments, caches, and the LIBERO checkout are
gitignored.

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
- The evaluated policy is not allowed to choose a standalone camera-motion or
  next-best-view action.
- Manipulating or holding an object may naturally alter what the wrist camera
  sees; this is treated as manipulation-driven information acquisition.

## Next milestone

The next implementation step is an instruction-conditioned belief model over:

```text
target identity
target location
target existence
target observability
grasp feasibility
action outcome
```

For each candidate action, the model should predict the post-action belief and
rank:

```text
DIRECT_ACT
BRING_CLOSER
ROTATE_TO_LABEL
CLEAR_OCCLUDER
OPEN_CONTAINER
NOT_FOUND
```

using task-relevant expected information gain, expected task progress, action
cost, and risk.

## Primary methodological references

- [Map Space Belief Prediction for Manipulation-Enhanced Mapping, RSS 2025](https://www.roboticsproceedings.org/rss21/p039.html)
- [Efficient Manipulation-Enhanced Semantic Mapping With Uncertainty-Informed Action Selection, 2025](https://arxiv.org/abs/2506.02286)
- [Interactive Learning of Physical Object Properties Through Robot Manipulation, 2024](https://arxiv.org/abs/2404.07344)

## Acknowledgment

The environments are built on
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[MuJoCo](https://mujoco.org/). This repository does not vendor LIBERO assets or
source code.
