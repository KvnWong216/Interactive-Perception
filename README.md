# Interactive Perception

Learning prompt-conditioned outcomes of grounded physical interventions for
frozen VLA execution.

> **Status — 2026-09-07.** This repository contains the first software
> implementation of the formal-v1 pipeline. The release is verified only with
> contract tests, synthetic tensors, and a deterministic replay. It does not
> contain a trained formal-v1 checkpoint, an upstream candidate-proposal/
> grounding adapter, a frozen-VLM adapter, a MolmoAct2 integration, a LIBERO
> policy result, or a real-robot result.

## Research objective

A robot may know how to open, move, rotate, inspect, grasp, and place, yet still
fail when the observation is insufficient for the user's request. The question
here is:

> Under a fixed executor and interaction budget, can the robot choose the right
> grounded physical intervention and use the newly revealed evidence to improve
> its next action on the requested target?

Given a public prompt and visual/action history, Stage 1 compares complete
grounded interventions

```text
u = (primitive, referent, parameters, current-frame grounding)
```

by their predicted task outcomes. Stage 2 delegates the selected subtask to a
separately frozen VLA. The robot then reobserves the real scene before deciding
again.

## Method at a glance

```text
prompt + public RGB/action history
                │                         externally supplied
                ▼                       pre-bound interventions
 frozen VLM token field (protocol)               │
                └──────────────┬─────────────────┘
                               ▼
     support-constrained fusion → p(task outcome | context, intervention)
                │
                ▼
       feasible finite-set argmax
                │
                ▼
 referential adapter → frozen VLA
                │
                ▼
      action → real reobservation → repeat
```

- `x_t = (q, o_≤t, h_<t)` is the public prompt, observation, and action history.
- `u_j = (m_j, ρ_j, η_j)` is one complete grounded intervention.
- The primary output is the probability of a preregistered bounded task outcome
  under a fixed serializer, executor, continuation policy, and horizon.
- Internal tokens are predictive representations. They are not called a
  calibrated belief or uncertainty distribution without additional evidence.

The first learned model is deliberately small. `GroundedCandidateEncoder`
receives candidate tokens with **pre-bound** `grounding_support`, attends only
within those current-image patches, and passes the resulting representation to
the outcome scorer. It does not propose candidates, discover referents, or
predict grounding. `OutcomePrediction` contains one bounded task-success logit
per candidate plus masks and immutable identities; it has no branch or
`grounding_logits` output. There is no route head, hand-weighted uncertainty
score, factor ontology, future-latent path, or conformal singleton rule in
formal v1.

## Scope and non-goals

In scope:

- interactive manipulation: `OPEN` and `REMOVE`;
- information enrichment: `ROTATE` and `BRING_CLOSE`;
- direct task execution and safe `STOP` in the same candidate set; and
- receding-horizon decisions based on actual post-action observations.

Outside the current release:

- active viewpoint change;
- continuous trajectory generation in Stage 1;
- VLA weight updates;
- multi-step tree search or a claim of solving a full POMDP;
- calibration, EDL, JEPA, or future-latent prediction; and
- benchmark, OOD, or real-robot performance claims.

Online policy input is restricted to public observations, the complete prompt,
public action history, and optional public proprioception. Simulator semantic or
instance IDs, hidden poses or contents, ground-truth masks, task predicates,
rewards, and evaluator labels are rejected. Post-action semantic branch labels
remain audit/evaluation data and are not appended to policy history.
Concrete adapters must additionally audit provenance; a type checker cannot
detect private state deliberately encoded inside an otherwise public string or
tensor.

## Current release status

| Component | Code | Empirical validation | Boundary |
| --- | ---: | ---: | --- |
| Public-input firewall and frame contract | Yes | No | Contract tests only |
| Immutable grounded-intervention identity | Yes | No | Contract tests only |
| Frozen-token metadata and pre-bound-support contracts | Yes | No | Synthetic tensors only |
| Token-level outcome model and executed-only loss | Yes | No | Forward/backward smoke only |
| Feasible finite-set selector | Yes | No | Deterministic unit tests only |
| Serializer/request/receipt identity chain | Yes | No | Replay only; geometry is audit-only |
| Reobserve-and-repeat runtime | Yes | No | Deterministic replay only |
| Upstream candidate proposal / grounding adapter | No | No | Not implemented |
| Concrete frozen VLM token provider | No | No | Protocol only |
| Concrete MolmoAct2 executor | No | No | Interface only |
| Reset-controlled formal-v1 dataset/checkpoint | No | No | Collection contract only |
| LIBERO main experiment or real robot | No | No | Not released |

## Inputs and outputs

| Boundary | Main fields | Meaning |
| --- | --- | --- |
| Public context | `prompt`, content-addressed RGB frames, public history, optional proprioception | Deployment-available evidence |
| Candidate | `candidate_id`, primitive, referent, parameters, camera/frame/box/point | One complete physical intention |
| Frozen tokens | provider ID, token tensor, valid/current-patch masks, camera/frame IDs, patch boxes | Reversible public token provenance |
| Model output | candidate/context identities, bounded task-success logits, valid mask | No grounding or branch prediction |
| Decision | selected candidate or `ABSTAIN` | Feasible argmax; stable proposal-order tie break |
| Stage-2 request | subtask text plus identity/audit payload | Text is consumed; exact region is not claimed as a native VLA input |
| Receipt | candidate and request digests, status, actual post frames | Byte-matched execution record |
| Next context | prior frames plus actual post-action frames; public subtask and execution status | Input to the next Stage-1 decision; hashes and evaluator labels stay in the audit trace |

The supervised outcome contract has no defaults. Before collecting labels it
must name the primary outcome, continuation policy, horizon, executor,
serializer, and failure handling. Only the candidate actually executed in a
reset-controlled branch receives an outcome label; unexecuted alternatives are
never assigned fabricated counterfactual targets.

Serializer and executor identity belong to this `OutcomeContract` and to the
subsequent request/receipt chain. They are not fields of
`GroundedIntervention`.

## Repository structure

```text
src/grounded_interaction/   active formal-v1 package
  contracts.py             public context and complete intervention schemas
  tokens.py                frozen-token provenance and grounding support
  model.py                 pre-bound-support fusion and outcome scorer
  losses.py                executed-candidate Bernoulli NLL
  selection.py             feasibility filtering and finite-set argmax
  serialization.py         deterministic referential text adapter
  execution.py             frozen-VLA request/receipt boundary and replay double
  loop.py                  execute, reobserve, append history, repeat
  adapters.py              protocols only; concrete providers remain pending
  smoke.py                 canonical software-only smoke run
tests/                      focused formal-v1 verification
docs/formal_pipeline_v1.md  full method and data contract
env/README.md               supported environment boundary
```

The pre-formal repository is intentionally absent from the active tree. It is
recoverable from Git tag `archive/pre-formal-v1-2026-09-07` at commit
`8c4be631`.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/KvnWong216/Interactive-Perception.git
cd Interactive-Perception
uv sync --no-editable --extra learned --extra dev
uv run --no-editable pytest -q
uv run --no-editable ip-smoke --output runs/formal_v1_smoke.json
```

The smoke report must state:

```text
software_verification_only: true
empirical_evidence: false
selected sequence: OPEN → DIRECT → STOP
```

It checks schema validation, token shapes, finite model output/loss, backward
propagation, candidate identity preservation, action execution plumbing, and
post-observation insertion into the next synthetic context. Its scripted scores
and replayed outcomes are fixtures, not model accuracy or robot success.

See [the environment guide](env/README.md) for the exact supported boundary.
The current release supports the CPU/PyTorch software checks on macOS and Linux.
No LIBERO scene, runner, or VLA environment is claimed by this release.

## Reproduction levels

| Level | Reproduces | Availability |
| --- | --- | ---: |
| L0 | Schemas, firewall, identity, selector, serializer tests | Available |
| L1 | One-batch outcome-model forward/backward and synthetic loop | Available |
| L2 | Frozen-VLM extraction plus candidate proposal/grounding | Pending |
| L3 | Qualified frozen-VLA execution in LIBERO | Pending |
| L4 | Reset-controlled outcome training/evaluation | Pending |
| L5 | Full hidden-result closed loop and benchmark table | Pending |
| L6 | Real-robot transfer | Not part of this release |

## Roadmap

Dates are internal planning targets, not conference deadlines.

| Milestone | Target | Required asset | Exit condition |
| --- | --- | --- | --- |
| M0 — formal-v1 software | 2026-09-07–09-14 | Current package, tests, smoke, docs | All released software checks pass |
| M1 — execution-interface ceiling | 2026-09-15–09-28 | Same-primitive/different-referent scenes and paired report | Correct referent measurably controls intended first contact |
| M2 — hidden-result branching | 2026-09-29–10-19 | Empty/target/distractor/exhausted reset groups | New evidence changes the second action |
| M3 — outcome supervision | 2026-10-20–11-16 | Result-only checkpoints and matched baselines | Outcome supervision improves held-out routing/task outcome |
| M4 — benchmark study | 2026-11-17–12-21 | Frozen splits, baselines, OOD tests, closed-loop table | Main simulation evidence is complete |
| M5 — paper freeze | 2027-01 | Tables, demo, evidence ledger, draft | Every claim is backed by sealed evidence |

## TODO

- [x] Replace the old multi-pipeline tree with one formal-v1 package.
- [x] Implement typed grounded candidates and candidate-conditioned outcome code.
- [x] Preserve candidate identity through selection, serialization, receipt, and
  reobservation.
- [x] Add a deterministic software-only smoke run.
- [ ] Implement and freeze an upstream candidate-proposal/grounding adapter.
- [ ] Integrate and freeze one concrete VLM token provider.
- [ ] Run the E1 referent-to-executor interface ceiling before large collection.
- [ ] Build randomized hidden-result branching scenes and reset groups.
- [ ] Collect actual candidate outcomes under one frozen outcome contract.
- [ ] Train result-only and matched route/history baselines.
- [ ] Add a future auxiliary only with no-path and equal-capacity controls.
- [ ] Add calibration only after the uncalibrated behavior is useful.
- [ ] Freeze benchmark splits and execute the simulation main study.

## Evidence boundary

The active tree contains no formal-v1 performance result. Passing software
tests establishes interface consistency only. Missing artifacts are never
printed as zero, and no prior primitive qualification transfers to a new
executor, serializer, candidate, or task contract.

The next valid scientific result is not another intermediate confidence score.
It is a paired execution-interface ceiling followed by hidden-content trials in
which identical initial public evidence leads to different second actions only
after a physical intervention reveals different observations.

## Documentation

- [Formal-v1 method and data contract](docs/formal_pipeline_v1.md)
- [Architecture decision record](docs/adr/0002_grounded_intervention_outcome_planning.md)
- [Environment guide](env/README.md)
