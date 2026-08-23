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
  -> calibrated current-frame patch set -> structured target-conditioned PICK
  -> separately qualified PLACE
```

The full-prefix binder, candidate-conditioned action-effect model, isolated
calibration, set-valued controller, current-patch-to-text bridge, and
certificate-gated external dispatcher are implemented but not yet trained or
validated on new real groups. The evaluator-only oracle intervention must first
show a positive causal executor ceiling on a prospectively sized paired
experiment. Real action-effect training, controller rollout, and scenario
expansion remain empirically gated. There is no information-value weighting:
multi-task scales are learned and decisions use typed singleton prediction-set
conditions.

The former Grounding DINO/SAM/DINOv2/SigLIP/Qwen/manual-utility pipeline is
frozen as [`B2 Heuristic V0`](baselines/heuristic_v0/README.md). It remains an
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
- [Target-binding train/calibration pipeline](docs/piu_binding_pipeline.md)
- [Action-effect and calibrated-control pipeline](docs/piu_action_effect_pipeline.md)
- [Claim--evidence ledger](docs/claim_evidence_ledger.md)
- [Original-drawer method cycle and negative-result report](docs/original_drawer_experiment_report.md)
- [Executed counterfactual effect-label policy](docs/executed_effect_dataset.md)
- [Submission-shaped internal paper draft](paper/main.md)
- [Automatically generated evidence/readiness tables](paper/generated/piu_evidence_tables_v1.md)
- [Automatically generated method and evidence-boundary figures](paper/generated/piu_method_pipeline_v1.svg)
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
The online binder/effect/controller path accepts no evaluator-label argument.
PICK/DIRECT can execute only with a nonempty current-frame conformal patch set,
which is converted to exact normalized boxes in a deterministic pi0.5 subtask.
Live dispatch additionally requires a prospective exact-binomial primitive
qualification certificate; the existing retrospective registry authorizes none.
Its minimum reliability is not a repository constant: an external task owner
must freeze an episode failure budget and a power-design alternative using
`configs/templates/piu_external_execution_risk_budget_template.yaml`. The code
derives the per-dispatch null as `1-delta/8`, freezes new state/controller
groups, and recomputes every outcome from a scheduled execution receipt. A
certificate covers the exact public candidate payload and spatial-serializer
mode, not merely a primitive name.
The same-state collector first creates a hash-bound public execution plan from
calibrated binder sets. Candidates outside their typed execution context remain
in the route matrix but are not physically forked and receive no invented
effect labels.

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

Create the CPU-only hash lock for the offline mainline, or verify a checkout
against the retained lock:

```bash
python scripts/repro/check_piu_offline_pipeline.py \
  --output runs/piu_offline_repro_check.json \
  --reference results/diagnostics/piu_offline_repro_preflight_v1.json
```

This command reports the external pi0.5/oracle/real-data gates as pending; it
does not reinterpret software readiness as empirical readiness.

Regenerate a new version of the paper tables from admissible reports, or verify
the retained v1 snapshot byte-for-byte:

```bash
python scripts/evaluation/build_piu_paper_tables.py --verify
```

The v1 table intentionally shows real successor rows as `PENDING`. Missing
artifacts are never printed as zero, development ablations stay separate from
sealed evidence, and B6/B7 remain oracle upper bounds.

Verify that public prose has not restored a retired threshold or relabeled a
compound DIRECT endpoint as a separately qualified PICK/PLACE primitive:

```bash
python scripts/evaluation/build_piu_claim_audit.py --verify
```

The retained JSON hashes the paper, README, status, and results surfaces. This
is a claim-semantics check, not performance evidence.

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
python scripts/infra/check_external_pi05.py \
  --host <external-pi05-host> --port 8002 \
  --identity results/diagnostics/pi05_libero_checkpoint_identity_v1.json \
  --probe-report runs/paper_cycle_executor_v2/seed1400/open_butter/report.json \
  --output results/diagnostics/external_pi05_endpoint_check_v1.json

python scripts/evaluation/build_oracle_target_prompt_schedule.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_schedule_v1.json

python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host <external-pi05-host> \
  --schedule results/method/original_drawer_oracle_prompt_screen_schedule_v1.json \
  --dry-run

python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host <external-pi05-host> \
  --schedule results/method/original_drawer_oracle_prompt_screen_schedule_v1.json \
  --endpoint-check results/diagnostics/external_pi05_endpoint_check_v1.json

python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_v2.json
```

Freeze the confirmation order from that screen result, then run the selected
style on the five disjoint confirmation seeds:

```bash
python scripts/evaluation/build_oracle_target_prompt_schedule.py \
  --phase confirmation \
  --screen-result results/method/original_drawer_oracle_prompt_screen_v2.json \
  --output results/method/original_drawer_oracle_prompt_confirmation_schedule_v1.json

python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase confirmation --style <selected-style> \
  --host <external-pi05-host> \
  --schedule results/method/original_drawer_oracle_prompt_confirmation_schedule_v1.json \
  --endpoint-check results/diagnostics/external_pi05_endpoint_check_v1.json
```

Then summarize with the same phase and style. Every oracle report declares two
online privileged inputs and uses the claim scope
`EVALUATOR_ONLY_ORACLE_UPPER_BOUND`.

### B1 prompted-VLM baseline

B1 uses a separate identified external VLM router; it is not silently replaced
by π0.5 prefix similarity or a local heuristic. The service exposes `GET
/metadata` and `POST /route`, and its frozen identity is supplied as a
`piu.prompted-vlm-router-identity.v1` artifact. The client sends only public RGB,
prompt, public history, and registered candidate descriptions. A response is
hash-bound to the exact request and may contain only one candidate ID; anything
outside the registered set is retained as ABSTAIN.

```bash
python scripts/pipeline/run_piu_prompted_vlm_closed_loop.py \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --candidate-set PATH/candidates.jsonl --initial-sample-id SAMPLE --seed SEED \
  --router-identity PATH/router_identity.json \
  --router-host ROUTER_HOST --router-port ROUTER_PORT \
  --pi05-host PI05_HOST --pi05-port 8002 \
  --qualification-map PATH/qualified_executor_map.json \
  --output-dir PATH/b1_episode --dry-run
```

Remove `--dry-run` only after both external identities and every physical
candidate certificate validate. No identified prompted-VLM router is currently
available, so B1 has software coverage but no empirical result.

### B2 frozen Heuristic V0

B2 is executed from the immutable tag, never reconstructed on the current
branch. Its tag contains only one inference decision, so the adapter does not
grant it later replanning. First create the detached worktree, then generate the
external-GPU plan from a paired initial capture:

```bash
git worktree add ../Interactive-Perception-heuristic-v0 baseline/heuristic-v0

python scripts/pipeline/run_piu_heuristic_v0_inference.py \
  --worktree ../Interactive-Perception-heuristic-v0 \
  --capture-report PATH/capture.json \
  --output-dir ../Interactive-Perception-heuristic-v0/runs/B2_GROUP --dry-run

python scripts/pipeline/run_piu_heuristic_v0_once.py \
  --attestation PATH/attestation.json \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --seed SEED --host PI05_HOST --port 8002 \
  --output-dir PATH/b2_episode --dry-run
```

Actual legacy perception/Qwen inference is prohibited on this workstation by
the 1500 MiB cap. The attestation hashes all five legacy model trees and the
exact frozen commit.

### B7 same-source oracle target-binding upper bound

B7 is not the existing conditional post-OPEN pilot. It starts from the same
paired hidden-target state as B0. The selected evaluator marker is a pixel-exact
no-op while the target mask is empty, then activates if the frozen policy makes
the target visible. A uniquely selected development screen artifact is required:

```bash
python scripts/pipeline/run_piu_oracle_binding_full_loop.py \
  --scenario-config configs/scenarios/original_drawer.yaml \
  --style-selection PATH/oracle_screen_result.json \
  --initial-state PATH/source_state.npz --initial-state-group GROUP \
  --split sealed_test --seed SEED --host PI05_HOST --port 8002 \
  --output-dir PATH/b7_episode --dry-run
```

The runner records target identity and instance segmentation as online oracle
inputs and cannot support a public-method claim.

### Main paired pilot and formal schedule

B8 and B0 pilot episodes must use the development split, the same opaque source
state and simulator seed within each pair, and one registered pi0.5 identity.
The public closed-loop aggregator accepts development episodes for this purpose;
the formal-row exporter still accepts sealed episodes only.

```bash
python scripts/evaluation/aggregate_piu_closed_loop_episode.py \
  --manifest PATH/b8_group/closed_loop_manifest.json \
  --output PATH/b8_group/episode.json

python scripts/evaluation/plan_piu_formal_paired_test.py \
  --treatment-episodes PATH/B8_*/episode.json \
  --comparator-episodes PATH/B0_*/episode.json \
  --output results/method/piu_fixed_drawer_b8_vs_b0_formal_plan_v1.json
```

The planner does not turn a small pilot p-value into a gate. It reports paired
effect/discordance/variance and uses joint 95% lower exact-binomial bounds for a
conservative exact-power operating point. A nonpositive or insufficiently
identified directional effect produces no sample size, not a round-number
fallback. Before loading the episodes it verifies every hash in the retained
offline reproduction lock. After allocating exactly the planned number of new
sealed groups, first freeze the exact opaque state that every method will load:

```bash
python scripts/evaluation/build_piu_formal_initial_states.py \
  --split-manifest data/piu/mainline_v1/split_manifest.json \
  --state GROUP_1 PATH/state_1.npz \
  --state GROUP_2 PATH/state_2.npz \
  --output data/piu/mainline_v1/formal_initial_states_v1.json
```

The state files are validated numeric NPZ transport artifacts and never policy
features. Then freeze the outcome-independent B0--B8 order:

```bash
python scripts/evaluation/build_piu_formal_schedule.py \
  --formal-plan results/method/piu_fixed_drawer_b8_vs_b0_formal_plan_v1.json \
  --split-manifest data/piu/mainline_v1/split_manifest.json \
  --initial-state-manifest data/piu/mainline_v1/formal_initial_states_v1.json \
  --output results/method/piu_fixed_drawer_formal_schedule_v1.json
```

Formal matrix authorization must bind this schedule hash in addition to row and
split hashes. Matrix assembly rejects pilot-group reuse, cohort-size drift,
missing B0--B8 cells, seed drift, and policy-identity drift.
It also rejects any row whose source-state hash differs from the state frozen
before outcome collection.

Sealed cells are run one at a time in the frozen order. Issue the next ticket,
pass that exact ticket to the scheduled B0--B8 runner, aggregate its episode,
and close it before requesting the next ticket:

```bash
python scripts/evaluation/begin_piu_formal_attempt.py \
  --schedule results/method/piu_fixed_drawer_formal_schedule_v1.json \
  --ledger-dir runs/piu_formal_v1/ledger \
  --run-output-dir runs/piu_formal_v1/ENTRY_OUTPUT

# Run the scheduled method with --formal-attempt-ticket
# runs/piu_formal_v1/ledger/NNNNN.started.json, producing episode.json.

python scripts/evaluation/close_piu_formal_attempt.py \
  --ticket runs/piu_formal_v1/ledger/NNNNN.started.json \
  --episode runs/piu_formal_v1/ENTRY_OUTPUT/episode.json
```

The ledger is deliberately fail-closed: an issued ticket without its bound
episode and close receipt blocks later cells. Do not delete partial output or
silently rerun it; retain the interruption for independent protocol review.

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
| full repository test suite | see current CI/preflight report | software and retained-artifact integrity |
| oracle visual-prompt preflight v2 | 8 eligible, 2 excluded; no policy calls | evaluator-only rendering/packet and identified-server contract, not method performance |
| same-RGB prompt router, held-out | legacy route-only 95.83% vs legacy route+effect 93.75% mean accuracy | pilot only; proxy effects; old artifact names B6/B7 |
| calibrated B7 pilot | 93.75% coverage, 68.75% abstain, 6.25% wrong execute | 16 held-out samples |
| fresh 10-seed OPEN qualification | drawer 9/10; hidden-target evidence 8/10 | fixed scenario; physical acquisition works |
| DIRECT after actual OPEN | nonempty butter mask initially 8/10; butter grasp contact 0/10; task 0/10 | retrospective information-utilization failure |
| visible-object executor control | compound DIRECT contact 10/10; terminal destination predicate 3/10 | endpoint diagnostic, not separate PICK/PLACE qualification |
| executed-effect grouped development CV | legacy route-only/route+effect macro F1 both 100%; effect accuracy 94.17% | CPU baseline; 10 inspected seed groups; old artifact names B6/B7; no formal calibration claim |
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
