# S03 public runner execution contract

This additive appendix freezes the canonical public-input S03 runner. It does
not alter the design runbook at
`docs/s03_perception_decision_runbook.md` (SHA-256
`8e11538a64c5bd302c5d915c236862b78e44a8007de52eb33f9012f317c44afa`),
the 620-record schedule, the input manifest, any S02 artifact, or any frozen
threshold. At this freeze point, 0/620 S03 records have been executed and the
canonical output root does not exist.

The frozen runner is
`scripts/pipeline/run_piu_s03_perception_decision.py`. Its execution identity
is `configs/experiments/piu_s03_model_identity_v1.json`; the enclosing source
commit and every inference-critical checkpoint byte are hash-bound. The
identity explicitly records that the V0 public pipeline is uncalibrated. There
is no confidence threshold and no substitute hand-tuned cutoff.

## Safe pre-outcome commands

These modes validate or construct a request in memory. They never load a
model, invoke inference, make a pi0.5 action call, create a receipt, or write an
outcome:

```bash
.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision.py \
  --schedule results/method/piu_s03_perception_decision_schedule_v1.json \
  --execution-index 0 --validate-only

.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision.py \
  --schedule results/method/piu_s03_perception_decision_schedule_v1.json \
  --execution-index 0 --dry-run

.venv/bin/python scripts/repro/validate_piu_s03_runner.py \
  --schedule results/method/piu_s03_perception_decision_schedule_v1.json \
  --execution-index 0
```

The dry-run payload has a separate `policy_request`. It contains only the
frozen prompt, public RGB references, public action history, candidate
registry, and an empty `online_oracle_inputs` list. Subtest, stratum, expected
route, S02 outcome, evaluator labels, and simulator semantic fields are not in
that policy payload.

## Future outcome-bearing command (not run by this freeze)

After separate authorization, exactly one next ordered index is consumed by:

```bash
.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision.py \
  --schedule results/method/piu_s03_perception_decision_schedule_v1.json \
  --execution-index INDEX --allow-outcome-write
```

Omitting `--allow-outcome-write` fails before a receipt or model request. The
output root is fixed to `runs/piu_s03_perception_decision_v1`; overriding it is
rejected. The runner cannot call pi0.5, `env.step`, or dispatch ACT/OPEN/STOP.
It only replays public images through the frozen perception/router backend.

The frozen schedule does not expose the six intermediate public-camera frames
required by the existing calibrated RGB outcome critic. Adding those frames
now would modify the frozen input set. The v1 identity therefore emits the
conservative set `AMBIGUOUS` for that critic output. This is a documented
fail-closed limitation: it makes C nonpositive where singleton evidence is
required, and it is not replaced with a score or pixel threshold. A future
critic-capable design requires a new schedule version and outcome-unseen
inputs.

## Single-use receipts and recovery

The canonical ledger is
`runs/piu_s03_perception_decision_v1/_receipts`. A started receipt is written
immediately at the request boundary and permanently consumes that index. A
closed receipt hash-links the previous close and either an evaluated record or
an infrastructure-failure record. Indices must be 0 through 619 in order;
skip, replacement, duplicate record ID, overwrite, and rerun are rejected.

Failure handling is fixed as follows:

- Before the started receipt: service/model/checkpoint/input preflight failure
  consumes nothing; fix the infrastructure and retry the same not-yet-started
  index.
- After start but before a prediction: close the index as infrastructure
  failure. Do not rerun it.
- After prediction but before outcome assembly: preserve and hash every partial
  public artifact, then close as infrastructure failure. Do not convert it to
  a model/task failure.
- Corrupt or interrupted immutable write: do not overwrite the bytes. Close
  with a distinct infrastructure-failure artifact if possible; otherwise use
  the explicit adjudication command below.
- Service unavailable before the request boundary: no receipt and no consumed
  index. Service loss after the boundary consumes the index as infrastructure
  failure.

The only recovery operation is:

```bash
.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision.py \
  --schedule results/method/piu_s03_perception_decision_schedule_v1.json \
  --execution-index INDEX \
  --adjudicate-interrupted-as-infrastructure-failure \
  --allow-outcome-write
```

It can close only the current started, unclosed index. It cannot resume model
inference, replace a prediction, or produce a task outcome.

The schemas are:

- `configs/schemas/piu_s03_model_identity_v1.schema.json`;
- `configs/schemas/piu_s03_outcome_v1.schema.json`;
- `configs/schemas/piu_s03_receipt_v1.schema.json`.

The DAG v2 checker reports runner readiness separately from the S03 outcome
gate. Runner readiness never marks S03 complete. Until a future validated
620-record certificate exists, the outcome gate remains
`READY_FOR_EXTERNAL_WORK` and `paper_claim_ready=false`.
