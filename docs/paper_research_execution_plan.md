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

| gate | experiment | pass condition | consequence |
|---|---|---|---|
| G0, complete | raw post-OPEN DIRECT | target pick | 0/8; freezes the utilization failure |
| G1 | oracle target prompt screen | best of box/point/spotlight on 3 groups | freeze one style without using confirmation groups |
| G2 | oracle target prompt confirmation | >=4/5 target picks and <=1/5 wrong-object contacts | pass: learn public RGB binder; fail: switch primitive |
| G3 | learned binder development | >=80% pick and <=10% wrong contact on new groups | qualify public-input target binding |
| G4 | placement decomposition | >=80% final placement after successful pick | otherwise split PICK and PLACE or add destination binding |
| G5 | isolated calibration/test | risk, coverage, stage conversion, final task success | only sealed evidence enters the main table |

The oracle experiments are upper bounds and always report two online privileged
inputs. They can select a research branch but cannot appear as learned-method
performance.

## Two-week execution cadence

| day | immutable deliverable |
|---|---|
| 0 | identified external pi0.5 endpoint and one finite action probe |
| 1 | nine G1 reports: 3 prompt styles x 3 development groups |
| 2 | frozen style choice and five G2 confirmation reports |
| 3-5 | public RGB binder if G2 passes, otherwise targeted PICK primitive |
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
/path/to/simulator-python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase screen --host 127.0.0.1 --port 8002

/path/to/simulator-python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase screen \
  --output results/method/original_drawer_oracle_prompt_screen_v1.json
```

Read `screen.selected_style`, then run the disjoint confirmation:

```bash
/path/to/simulator-python scripts/pipeline/run_oracle_target_prompt_gate.py \
  --phase confirmation --style SELECTED_STYLE \
  --host 127.0.0.1 --port 8002

/path/to/simulator-python scripts/evaluation/summarize_oracle_target_prompt_gate.py \
  --phase confirmation --style SELECTED_STYLE \
  --output results/method/original_drawer_oracle_prompt_confirmation_v1.json
```

The confirmation summary makes the branch decision automatically. No prompt
patching or threshold change is permitted after screening.

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
