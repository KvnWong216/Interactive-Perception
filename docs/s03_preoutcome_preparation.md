# S03 public-input pre-outcome preparation

The immutable scientific design remains
`docs/s03_perception_decision_runbook.md` at SHA-256
`8e11538a64c5bd302c5d915c236862b78e44a8007de52eb33f9012f317c44afa`.
This preparation layer does not revise its prompts, denominators, thresholds,
or failure policy.

## Canonical DAG amendment

`configs/experiments/piu_empirical_stage_dag_v1.yaml` is retained byte-for-byte
because the completed S02 schedule binds the v1 offline reproducibility lock.
The canonical v2 checker composes
`configs/experiments/piu_empirical_stage_dag_v2.yaml`, whose base reference is
hash-bound to v1. The amendment distinguishes two branches:

- `S03_oracle_development_gate` is explicitly a legacy privileged Oracle
  diagnostic and is ineligible for the current public-input S03 claim;
- `S03_public_perception_decision_input_freeze` validates only the sanitized
  manifest and offline schedule;
- `S03_public_perception_decision_gate` cannot complete without the future
  outcome certificate and its three hash-bound result artifacts.

Completing the input-freeze stage is not completing the public S03 outcome
gate. The missing certificate leaves the latter `READY_FOR_EXTERNAL_WORK` and
keeps `paper_claim_ready=false`.

## Frozen inputs and ordering

The input manifest is
`results/method/piu_s03_perception_decision_input_manifest_v1.json`; the exact
offline order is
`results/method/piu_s03_perception_decision_schedule_v1.json`. They contain:

- 124 A Information Effect pre/post pairs;
- 124 B/ACT, 124 B/OPEN, and 124 B/STOP records;
- 124 C hash-linked pre/post transition inputs;
- all S02 execution indices 0 through 123, including 101 and 104;
- direct SHA-256 references to public RGB and, only at post-OPEN phases, the
  public action history;
- opaque provenance hashes for the S02 certificate, outcome index, and ordered
  124-receipt tree, none of which is controller-visible or semantically loaded
  to construct a policy input.

Every record sets `outcome_present=false` and `inference_executed=false`. The
schedule additionally sets `predictions_present=false`, `labels_present=false`,
`certificate_present=false`, and `rollout_executed=false`.

The policy-visible object has an exact closed schema: prompt, public candidate
registry, and a phase-ordered sequence of direct agent/wrist RGB references
plus public action history. Simulator semantic/instance IDs, simulator masks,
object poses, oracle target markers/locations, container membership,
evaluator-only fields, task predicates, reward, and success/failure are
forbidden. Stratum and validator-role metadata remain outside the
policy-visible object.

## Safe verification

The following command rebuilds both artifacts in memory, compares their bytes,
re-hashes every direct input, checks the 620-record denominator and firewall,
and exits without invoking a model, simulator, or policy endpoint:

```bash
python scripts/evaluation/build_piu_s03_preparation.py --verify
```

The canonical v2 amendment can then be checked with:

```bash
python scripts/repro/check_piu_empirical_dag_v2.py
```

The historical `check_piu_empirical_dag.py` remains the byte-locked v1 checker;
changing it would invalidate S02's retained offline lock. Neither checker nor
the preparation verifier is an S03 inference or rollout entrypoint. Do not
create the certificate, run a router, call pi0.5, call `env.step`, join
evaluator-private labels, or execute a selected primitive under this
preparation authorization.
