# Paper research execution plan

This plan targets a successful robotics-method paper while keeping the current
T01D drawer scenario fixed for method development. The existing negative result
is immutable: OPEN acquires butter evidence, but the frozen policy does not use
it. The next method is allowed to proceed only after a causal executor gate.

## Proposed main-method hypothesis

The working hypothesis is **action-causal evidence binding**: a robot should
bind evidence revealed by its own interaction to the user's target and pass a
calibrated spatial reference—not only new RGB or a rewritten sentence—to the
frozen VLA. The eventual public-input method may use the prompt, pre/post-action
agentview and wrist RGB, public proprioception, and action history. Simulator
instance masks remain offline labels only.

The paper contribution is not “draw a box.” It is the measured bridge from
`information action -> acquired evidence -> target correspondence -> frozen-VLA
execution`, with calibration and abstention at each failed conversion.

## Blocking decision sequence

| stage | experiment | decision statistic | consequence |
|---|---|---|---|
| G0, complete | raw post-OPEN DIRECT | target pick | 0/8; freezes the utilization failure |
| G1 | oracle target prompt screen | target grasp contact, destination, wrong contact, then minimum changed RGB pixels | freeze one intervention without a hand-written style preference |
| G2 | oracle target prompt pilot | paired grasp-contact effect and continuous lift on five disjoint groups | estimate feasibility and prospectively size a separate formal test; no automatic branch |
| G2F | formal oracle mechanism test | exact paired test at predeclared alpha on new groups | a positive causal ceiling motivates binder development; null/negative evidence motivates a primitive study |
| G3 | learned binder development | development effect curves and failure modes on new groups | select architecture without touching calibration/test groups |
| G4 | placement decomposition | grasp contact, continuous lift, destination entry, and terminal placement | localize failure; do not impose an arbitrary percent gate |
| G5 | isolated calibration/test | risk, coverage, stage conversion, final task success | only sealed evidence enters the main table |

The oracle experiments are upper bounds and always report two online privileged
inputs. They can select a research branch but cannot appear as learned-method
performance.

## Two-week execution cadence

| day | immutable deliverable |
|---|---|
| 0 | identified external pi0.5 endpoint and one finite action probe |
| 1 | nine G1 reports: 3 prompt styles x 3 development groups |
| 2 | frozen style choice and five G2 pilot reports; prospective G2F sample-size calculation |
| 3-5 | collect formal oracle groups, then develop the supported executor-repair branch |
| 6 | explicit placement gate and destination-binding diagnosis |
| 7-9 | new same-scenario training/development seed groups; no reuse of G1/G2 for final claims |
| 10 | isolated calibration split, temperature/conformal thresholds frozen |
| 11-12 | sealed paired test: raw DIRECT, text rewrite, learned bridge, oracle ceiling |
| 13 | automatically generated stage-conversion table, confidence intervals, and failure taxonomy |
| 14 | paper method/results revision and clean-checkout reproduction |

The fixed scenario can establish a narrow causal mechanism, not broad
generalization. A true ICRA/RSS main-method claim ultimately requires additional
tasks/scenes; those are deferred until G2-G4 pass so compute is not spent scaling
a broken executor.

## Supplying the external `host:port`

Use a remote NVIDIA machine with at least 16 GB VRAM (24 GB preferred), about
20 GB free disk, the OpenPI checkout, and the exact official checkpoint. On the
remote machine, clone/pull this repository and run:

```bash
cd /path/to/Interactive-Perception
EXPERIMENT_GPU_INDEX=0 PORT=8002 \
  bash scripts/infra/serve_pi05.sh \
  /path/to/openpi \
  /path/to/checkpoints/checkpoints/pi05_libero
```

The server hashes the complete checkpoint before loading it and exposes the
identity in WebSocket metadata. The expected identity is retained in
`results/diagnostics/pi05_libero_checkpoint_identity_v1.json`.

The recommended connection is an SSH tunnel. Run this locally and keep it open:

```bash
ssh -N -L 8002:127.0.0.1:8002 USER@REMOTE_GPU_HOST
```

The requested `host:port` is then simply `127.0.0.1:8002`. No public firewall
rule is needed. Validate identity and one action sample from a retained policy
frame:

```bash
/path/to/simulator-python scripts/infra/check_external_pi05.py \
  --host 127.0.0.1 --port 8002 \
  --identity results/diagnostics/pi05_libero_checkpoint_identity_v1.json \
  --probe-report runs/paper_cycle_executor_v2/seed1400/open_butter/report.json \
  --output results/diagnostics/external_pi05_endpoint_check_v1.json
```

Do not expose port 8002 directly to the public internet. If direct networking is
unavoidable, restrict the firewall to the local machine's source IP and use a
private network or authenticated reverse proxy.

## Oracle screen and confirmation

After the endpoint check passes:

```bash
/path/to/simulator-python \
  scripts/evaluation/build_oracle_target_prompt_schedule.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_schedule_v1.json

/path/to/simulator-python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host 127.0.0.1 --port 8002 \
  --schedule results/method/original_drawer_oracle_prompt_screen_schedule_v1.json \
  --endpoint-check results/diagnostics/external_pi05_endpoint_check_v1.json

/path/to/simulator-python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_v2.json
```

Read `screen.selected_style`, then run the disjoint confirmation:

```bash
/path/to/simulator-python \
  scripts/evaluation/build_oracle_target_prompt_schedule.py \
  --phase confirmation \
  --screen-result results/method/original_drawer_oracle_prompt_screen_v2.json \
  --output results/method/original_drawer_oracle_prompt_confirmation_schedule_v1.json

/path/to/simulator-python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase confirmation --style SELECTED_STYLE \
  --host 127.0.0.1 --port 8002 \
  --schedule results/method/original_drawer_oracle_prompt_confirmation_schedule_v1.json \
  --endpoint-check results/diagnostics/external_pi05_endpoint_check_v1.json

/path/to/simulator-python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase confirmation --style SELECTED_STYLE \
  --output results/method/original_drawer_oracle_prompt_pilot_v2.json
```

The pilot summary reports paired effects and explicitly sets `passed: null`.
It never makes a method branch automatically. The formal group count is frozen
after a power calculation based on this independent pilot; no prompt patching
or threshold change is permitted after screening.

```bash
/path/to/simulator-python scripts/evaluation/plan_oracle_paired_test.py \
  --pilot results/method/original_drawer_oracle_prompt_pilot_v2.json \
  --output results/method/original_drawer_oracle_formal_plan_v1.json
```

For reference, even five intervention-only discordances give a two-sided exact
`p=0.0625`; six give `p=0.03125`. The calculator uses the full pilot-estimated
discordant-pair probabilities rather than treating `4/5` as evidence.

When the plan status is `PROSPECTIVE_GROUP_COUNT_FROZEN`, create a new immutable
`oracle_formal_split_manifest.json` containing exactly that many
`oracle_formal` groups, excluding every oracle preflight/pilot and
OPEN-qualification group/seed, and freeze their pre-OPEN opaque states. Do not
append them to a qualification or later sealed-test manifest: those counts are
known at different prospective decision points, and changing one hash-bound
split would invalidate earlier schedules. G2F itself requires a formal OPEN
executor certificate: every
group attempts that exact candidate once, regardless of whether OPEN succeeds,
then starts the target-marker and raw-DIRECT arms from the same resulting state
in a hash-keyed order. Failed or interrupted source/arm executions remain false
in the complete intention-to-treat denominator.

```bash
python scripts/data/build_piu_planned_split_manifest.py \
  --purpose oracle_formal \
  --plan results/method/original_drawer_oracle_formal_plan_v1.json \
  --seed-start EXTERNALLY_RESERVED_SEED_START \
  --group-prefix oracle-formal \
  --exclude-split PATH/open_qualification_split.json \
  --output data/piu/mainline_v1/oracle_formal_split_manifest.json

python scripts/evaluation/build_oracle_formal_initial_states.py \
  --split-manifest data/piu/mainline_v1/oracle_formal_split_manifest.json \
  --state GROUP_1 PATH/state_1.npz --state GROUP_2 PATH/state_2.npz \
  --output data/piu/mainline_v1/oracle_formal_initial_states_v1.json

python scripts/evaluation/build_oracle_formal_schedule.py \
  --formal-plan results/method/original_drawer_oracle_formal_plan_v1.json \
  --split-manifest data/piu/mainline_v1/oracle_formal_split_manifest.json \
  --initial-state-manifest data/piu/mainline_v1/oracle_formal_initial_states_v1.json \
  --open-certificate results/method/piu_open_primitive_certificate_v1.json \
  --output results/method/original_drawer_oracle_formal_schedule_v1.json

python scripts/pipeline/run_oracle_formal_group.py \
  --schedule results/method/original_drawer_oracle_formal_schedule_v1.json \
  --execution-index INDEX \
  --endpoint-check results/diagnostics/external_pi05_endpoint_check_v1.json \
  --host 127.0.0.1 --port 8002 --execution-location external_simulator

python scripts/evaluation/analyze_oracle_formal_experiment.py \
  --schedule results/method/original_drawer_oracle_formal_schedule_v1.json \
  --output results/method/original_drawer_oracle_formal_result_v1.json
```

The result uses the predeclared exact two-sided paired test and reports a
conservative paired-risk-difference interval. It establishes at most a
privileged target-binding mechanism in this fixed scenario; it cannot serve as
learned-method performance.

## Paper experiment table after executor qualification

The main paired comparison will contain raw frozen DIRECT, best text-only
rewrite, route-only interaction, learned action-causal binding, no-calibration,
no-action-history, and oracle upper bound. Report target visibility, correct
contact, lift, wrong-object contact, destination reached/final, complete task
success, abstention, prediction-set size, and interaction cost separately.

Development chooses architecture. A new group-disjoint calibration split fits
all thresholds. A sealed test split is opened once. Report Wilson intervals and
paired exact tests, preserve every action history and source-state hash, and
keep oracle results in a visually separate upper-bound column.

## Main B8-vs-B0 Phase 9 design

After real, group-disjoint training/calibration artifacts and primitive
certificates exist, collect paired B8 and B0 development episodes and aggregate
them into the standard episode schema. The main planner validates identical
source-state hash, simulator seed, and pi0.5 identity within each pair:

```bash
python scripts/evaluation/plan_piu_formal_paired_test.py \
  --treatment-episodes PATH/B8_*/episode.json \
  --comparator-episodes PATH/B0_*/episode.json \
  --output results/method/piu_fixed_drawer_b8_vs_b0_formal_plan_v1.json
```

The formal N is calculated only if the joint 95% lower exact-binomial design
point supports a directional B8 effect and exact two-sided paired power reaches
0.80 within 200 groups. Otherwise the blocked plan is retained. After assigning
exactly N new sealed groups, create a separate immutable
`formal_split_manifest.json` and freeze execution order without reading
outcomes:

```bash
python scripts/data/build_piu_planned_split_manifest.py \
  --purpose sealed_test \
  --plan results/method/piu_fixed_drawer_b8_vs_b0_formal_plan_v1.json \
  --seed-start EXTERNALLY_RESERVED_SEED_START \
  --group-prefix sealed-main \
  --exclude-split PATH/open_qualification_split.json \
  --exclude-split PATH/pick_qualification_split.json \
  --exclude-split PATH/place_qualification_split.json \
  --exclude-split data/piu/mainline_v1/oracle_formal_split_manifest.json \
  --exclude-split data/piu/mainline_v1/learning_split_manifest.json \
  --output data/piu/mainline_v1/formal_split_manifest.json

python scripts/evaluation/build_piu_formal_initial_states.py \
  --split-manifest data/piu/mainline_v1/formal_split_manifest.json \
  --state GROUP_1 PATH/state_1.npz --state GROUP_2 PATH/state_2.npz \
  --output data/piu/mainline_v1/formal_initial_states_v1.json

python scripts/evaluation/build_piu_formal_schedule.py \
  --formal-plan results/method/piu_fixed_drawer_b8_vs_b0_formal_plan_v1.json \
  --split-manifest data/piu/mainline_v1/formal_split_manifest.json \
  --initial-state-manifest data/piu/mainline_v1/formal_initial_states_v1.json \
  --output results/method/piu_fixed_drawer_formal_schedule_v1.json
```

The schedule's SHA-256 permutation randomizes group and within-group B0--B8
order and retains every seed. Sealed matrix authorization binds its hash. The
schedule also binds the pre-outcome source-state hash for every row. Main
interpretation reports full-task/wrong-contact comparisons against B1/B3,
interaction cost, calibration efficiency, the B0-to-B7 oracle gap, and the
B4-versus-B3 effect ablation; no p-value alone creates a success claim.
