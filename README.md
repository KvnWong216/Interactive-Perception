# Prompt-conditioned Interaction Uncertainty

PIU asks which prompt-relevant fact is missing, where its evidence may be, and
which executable interaction should acquire it before a frozen VLA resumes the
task.

```text
public agentview/wrist RGB + prompt + public history
  -> Grounding DINO + SAM + DINOv2 object scene packet
  -> frozen SigLIP prompt-conditioned belief/uncertainty field
  -> Qwen2.5-VL registered action-effect predictions
  -> explicit information/task utility
  -> typed semantic option -> frozen pi05_libero
  -> six public observations -> conformal outcome -> update -> replan
```

Online code never consumes segmentation, simulator depth, hidden object poses,
articulated joints, task predicates, semantic IDs, BEV, or global cameras.
Those fields are permitted only in data builders and evaluator replays after
controller termination.

## Repository

```text
configs/                  action priors and scenario parameters
scripts/
  infra/                  installation, GPU preflight, pi0.5 server
  perception/             object scene-packet construction
  pipeline/               one-step inference and semantic-option execution
  data/                   collection, paired states, frozen features
  training/               belief/effect and public-RGB outcome training
  evaluation/             clean evaluation, outcome scoring, manifests
  visualization/          four-panel demo renderer
src/
  interaction_uncertainty/ belief, VLM reasoning, learned sidecar
  interactive_perception/  pi0.5 bridge, observe return, RGB critic
benchmarks/               current machine-readable protocols and gates
results/                  only the current demo and frozen evidence
```

Scenario details are data, not Python branches. For example,
[`configs/scenarios/original_drawer.yaml`](configs/scenarios/original_drawer.yaml)
defines the BDDL, prompt, semantic options, seed, and evaluator targets used by
the current cluttered drawer case. Every runner also accepts direct CLI
overrides.

## Main entry points

Check the local installation and GPU:

```bash
python scripts/infra/check_install.py
bash scripts/infra/check_gpu.sh
```

Run public-RGB PIU inference:

```bash
python scripts/pipeline/infer.py \
  --agentview results/assets/piu_messy_fresh_e2e_seed1399_v1/public_keyframes/00_before_agentview.png \
  --wrist results/assets/piu_messy_fresh_e2e_seed1399_v1/public_keyframes/00_before_wrist.png \
  --prompt "Place the butter in the basket" \
  --asset-dir runs/example/inference_assets \
  --output runs/example/inference.json
```

Execute any registered semantic option and return to the public initial pose:

```bash
python scripts/pipeline/execute.py \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --role OPEN_CONTAINER \
  --assets runs/example/option_assets \
  --work runs/example/work \
  --output runs/example/option.json
```

Collect and train with arbitrary protocol/split paths:

```bash
python scripts/data/collect_snapshots.py --help
python scripts/perception/build_scene_packets.py --help
python scripts/data/extract_features.py --help
python scripts/training/train.py --help
python scripts/training/train_outcome.py --help
python scripts/evaluation/evaluate.py --help
```

All generated outputs are immutable: choose a new output directory for a new
run.

## Evidence status

| Evidence | Result | Meaning |
|---|---:|---|
| Six-frame RGB outcome, clean development | GO, 119/120 singleton-correct | Outcome component only |
| Six-frame RGB outcome, sealed audit | GO, 294/300 singleton-correct | Outcome component only |
| Object PIU scene-disjoint clean | NOT-GO, 6 false singleton routes | Initial belief/action model not paper-ready |
| Current cluttered-drawer demo | OPEN_CONTAINER -> REVEALED -> MOVE_CLOSER | One disposable information trace |
| Final butter retrieval | 0/5 | Final task remains NOT-GO |

The demo proves wiring and one real information acquisition, not held-out
closed-loop performance or final manipulation success. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for claim boundaries.

## Current assets

- Demo: `results/demos/piu_original_fresh_seed1399_v1/`
- Initial maps: `results/assets/piu_messy_corrected_initial_seed1399_v1/`
- Post-action maps: `results/assets/piu_messy_corrected_post_open_seed1399_v1/`
- Six-frame public option: `results/assets/piu_messy_fresh_e2e_seed1399_v1/`
- Machine trace: `results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition_trace.json`

## Claim discipline

Target observability and final-task success are always reported separately.
`EMPTY` excludes only the inspected region. Any non-singleton outcome is
`SAFE_STOP`; the controller never selects a convenient label from a conformal
set. PARTIAL is NOT-GO.
