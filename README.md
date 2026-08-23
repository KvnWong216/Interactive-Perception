# Prompt-Conditioned Interaction Belief for VLA

This repository studies whether a robot can recognize that the current
observation is insufficient for a user's prompt, acquire the missing evidence
through physical interaction, bind the new evidence to the requested target,
and use it in the next manipulation.

The current falsifiable method line is deliberately restricted to one unchanged
hidden-butter drawer scenario:

```text
closed drawer -> OPEN -> new butter evidence
  -> full frozen PaliGemma patch tokens + prompt + public OPEN history
  -> learned prompt-conditioned spatial binding
  -> target-conditioned PICK -> PLACE
```

The spatial binder and full-prefix cache contract are implemented but not yet
trained or validated. The evaluator-only oracle intervention must first show a
positive causal executor ceiling on a prospectively sized paired experiment.
Action-effect prediction, information-value weighting, conformal control, and
scenario expansion are deferred until that gate.

The former Grounding DINO/SAM/DINOv2/SigLIP/Qwen/manual-utility pipeline is
frozen as [`B1 Heuristic V0`](baselines/heuristic_v0/README.md). It remains an
engineering baseline and is not the proposed method.

**Current decision (2026-08-22):** the broad effect-aware full-loop method is
rejected for the fixed drawer experiment. The six-stage rebuild gives OPEN
`9/10`, acquisition `8/9` conditional on OPEN, and target contact `0/8`
conditional on acquisition; wrong-object contact occurs in `3/8` of those
acquisition-success groups. Executed-effect supervision adds 0.00 route macro
F1 over route-only in all five grouped development folds. This is a controlled
failure decomposition, not a successful ICRA/RSS method claim.

The next falsifiable repair test is implemented as an evaluator-only visual
prompt that gives frozen pi0.5 the exact target region after OPEN. This is an
oracle upper bound, never the proposed method. Its five disjoint groups are now
an independent feasibility pilot, not an automatic `4/5` pass/fail rule. A
separate prospective test will be sized from the pilot before any public-RGB
target-binding method is claimed or rejected.

## Research artifacts

- [Literature lineage](docs/literature_lineage.md), audited through 2026-08-22
- [Novelty gate](docs/novelty_audit.md)
- [Architecture decision](docs/adr/0001_candidate_conditioned_calibrated_interaction.md)
- [Research question, math, data, experiments, compute, and go/no-go plan](docs/research_plan.md)
- [Internal ICRA-cadence schedule](docs/icra_cadence_plan.md)
- [Paper research execution plan and external pi0.5 setup](docs/paper_research_execution_plan.md)
- [Method and threshold provenance audit](docs/method_provenance_audit.md)
- [Spatial-prefix successor contract](docs/spatial_prefix_successor_contract.md)
- [Frozen PIU research charter](docs/research_charter.md)
- [Drawer-binding preregistration](docs/preregistration.md)
- [Claim--evidence ledger](docs/claim_evidence_ledger.md)
- [Original-drawer method cycle and negative-result report](docs/original_drawer_experiment_report.md)
- [Executed counterfactual effect-label policy](docs/executed_effect_dataset.md)
- [Submission-shaped internal paper draft](paper/main.md)
- [Learned-package contracts](src/calibrated_interaction/README.md)
- [中文旧系统总览与复现教程](docs/TAKEAWAY_AND_TUTORIAL_CN.md), retained as
  Heuristic V0 documentation

## Repository

```text
baselines/heuristic_v0/       immutable legacy baseline lock
configs/capabilities/         real executor primitive registry
configs/experiments/          one-GPU learned-method configuration
configs/scenarios/            scenario data; no Python scenario branches
src/calibrated_interaction/   candidate schema, decoder, calibration, controller
src/piu/                      current leakage firewall, evaluator, spatial binder
src/interaction_uncertainty/  legacy heuristic implementation
src/interactive_perception/   frozen pi0.5 execution bridge and legacy critics
scripts/pipeline/             learned replay and legacy live runners
scripts/data/                 counterfactual/snapshot builders
tests/                        unit, leakage, split, model-shape, integration tests
benchmarks/                   protocols and gates
results/                      retained component evidence and public RGB assets
```

The current retrospective sprint dataset stores public transitions and
privileged evaluator sidecars in separate JSONL files under
`data/piu/drawer_binding_sprint_v1/`. It is development evidence only and is
forbidden for training, calibration, or formal testing.

Online learned-method inputs are limited to the complete prompt, agentview/wrist
RGB history, and public action/observation history. Simulator segmentation,
semantic IDs, hidden object poses, articulated joint truth, depth by default,
and task predicates are forbidden policy inputs. Privileged state is allowed
only for offline labels, evaluator metrics, and oracle upper bounds.

## Install and validate

The core contracts and calibration tools require only NumPy and PyYAML. Run
their focused tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev \
  pytest -q tests/test_calibrated_interaction.py
```

Run the full repository suite (legacy tests need PyTorch and Pillow too):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra learned --extra vlm --extra dev \
  pytest -q
```

Run the CPU-only, no-training closed-loop replay on the same original drawer
scenario:

```bash
uv run python scripts/pipeline/run_calibrated_replay.py \
  --scenario configs/scenarios/original_drawer.yaml \
  --replay tests/fixtures/original_drawer_calibrated_replay.json \
  --output runs/original_drawer_calibrated_replay.json
```

This replay verifies candidate validation, calibrated set arbitration, the
short text adapter, history updates, and `OPEN -> reobserve -> DIRECT` control
flow. Its probabilities are explicitly fixture-only: it is not trained-model
accuracy or robot task-success evidence.

Check the full simulator/GPU installation and frozen π0.5 server separately:

```bash
python scripts/infra/check_install.py
bash scripts/infra/check_gpu.sh
bash scripts/infra/serve_pi05.sh
```

### Oracle target-binding qualification

The 1.5 GB local GPU contract prohibits loading pi0.5 on this workstation, so
the qualification runner accepts only an identified external frozen-policy
server. The recommended `host:port` is an SSH tunnel:

```bash
ssh -N -L 8002:127.0.0.1:8002 USER@REMOTE_GPU_HOST
```

This makes the local endpoint `127.0.0.1:8002` without exposing a public port.
Complete remote-server commands and hardware expectations are in the
[paper execution plan](docs/paper_research_execution_plan.md). First
reproduce the policy-free preflight in the simulator environment:

```bash
python scripts/evaluation/preflight_oracle_target_prompt.py \
  --output runs/preflight/original_drawer_oracle_prompt.json
```

Then inspect and run the nine screening jobs against an external service:

```bash
python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host <external-pi05-host> --dry-run

python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host <external-pi05-host>

python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_v1.json
```

Run the selected style on the five disjoint confirmation seeds with
`--phase confirmation --style <selected-style>`, then summarize with the same
phase and style. Every oracle report declares two online privileged inputs and
uses the claim scope `EVALUATOR_ONLY_ORACLE_UPPER_BOUND`.

## Legacy baseline entry points

Public-RGB Heuristic V0 inference:

```bash
python scripts/pipeline/infer.py \
  --agentview results/assets/piu_messy_fresh_e2e_seed1399_v1/public_keyframes/00_before_agentview.png \
  --wrist results/assets/piu_messy_fresh_e2e_seed1399_v1/public_keyframes/00_before_wrist.png \
  --prompt "Place the butter in the basket" \
  --asset-dir runs/heuristic_v0/inference_assets \
  --output runs/heuristic_v0/inference.json
```

Execute a legacy registered semantic option:

```bash
python scripts/pipeline/execute.py \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --role OPEN_CONTAINER \
  --assets runs/heuristic_v0/option_assets \
  --work runs/heuristic_v0/work \
  --output runs/heuristic_v0/option.json
```

Do not add rules or scores to these legacy entry points.

## Evidence status

| evidence | result | claim boundary |
|---|---:|---|
| full repository test suite | 71 passed | software and retained-artifact integrity |
| oracle visual-prompt preflight v2 | 8 eligible, 2 excluded; no policy calls | evaluator-only rendering/packet and identified-server contract, not method performance |
| same-RGB prompt router, held-out | B6 95.83% vs B7 93.75% mean accuracy | pilot only; proxy effects |
| calibrated B7 pilot | 93.75% coverage, 68.75% abstain, 6.25% wrong execute | 16 held-out samples |
| fresh 10-seed OPEN qualification | drawer 9/10; hidden-target evidence 8/10 | fixed scenario; physical acquisition works |
| DIRECT after actual OPEN | butter visible initially 8/10; pick 0/10; task 0/10 | information-utilization failure |
| visible-object executor control | pick 10/10; terminal placement 3/10 | placement gate fails |
| executed-effect grouped development CV | B6/B7 route macro F1 both 100%; effect accuracy 94.17% | CPU baseline; 10 inspected seed groups; no formal calibration claim |
| [original-drawer calibrated replay](results/diagnostics/original_drawer_calibrated_replay_v1.json) | OPEN -> reobserve -> DIRECT | wiring only; fixture probabilities |
| six-frame RGB legacy outcome, clean development | 119/120 singleton-correct | legacy outcome component only |
| six-frame RGB legacy outcome, sealed audit | 294/300 singleton-correct | legacy outcome component only |
| legacy object PIU scene-disjoint | NOT-GO, 6 false singleton routes | Heuristic V0 is not paper-ready |
| legacy current cluttered-drawer demo | OPEN_CONTAINER -> REVEALED -> MOVE_CLOSER | one disposable information trace |
| legacy five-seed final butter retrieval | 0/5 | historical Heuristic V0 result |

Current assets include the original frontend under
`results/assets/original_drawer_frontend_v1/`, its public Scene Packet under
`results/diagnostics/original_drawer_scene_packet_v1.jsonl`, and the legacy demo
under `results/demos/piu_original_fresh_seed1399_v1/`.

## Claim discipline

Hidden representations are not called uncertainty. Entropy is a diagnostic,
not the final decision definition. Conformal coverage is marginal under
exchangeability, not a single-trial success probability. Primitive execution,
information acquisition, post-action recognition, rerouting, and final task
success are always reported separately. Any non-actionable calibrated set
produces `ABSTAIN`; it is never collapsed to a convenient top-1.
