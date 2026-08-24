# S03 public execution v2 amendment

This note seals the S03 public-v1 execution history and prospectively defines
the replacement execution identity. It does not change the frozen 620 logical
records, record IDs, ordering, evaluator thresholds, public-input firewall, or
any S02 artifact.

## Sealed v1 history

S03 public v1 is infrastructure-blocked after execution index 0. The index was
started and closed exactly once as an infrastructure failure caused by a
Grounding DINO post-processing API mismatch. It produced no prediction, no
model/task outcome, and no S03 certificate. Its started receipt, closed
receipt, infrastructure-failure artifact, blocker diagnostic, and partial tree
are immutable. Index 0 is consumed and is not eligible for rerun, relabeling,
deletion, or reuse under v1. S03 v1 is not eligible for paper performance
claims.

The canonical blocker is
`results/diagnostics/piu_s03_execution_blocker_v1.json` (SHA-256
`9e742a7a20cc28622da4b95c9e75079e1e8700d707aec4fa7d935952d89e6048`).
The sealed partial execution tree SHA-256 is
`eee3900ad0e490e027f18436f979bc72977c015ae27870e80d7d4ac278b265ad`.

## Prospective v2 execution

S03 public v2 starts a new single-use ledger at
`runs/piu_s03_perception_decision_v2`. It will execute all 620 unchanged
logical records exactly once in their original order under a newly hash-bound
runner identity. The v1 index-0 infrastructure failure is historical audit
evidence and is excluded from v2 outcome statistics; v2 still evaluates the
corresponding logical record once under v2. There is no filtering,
replacement, cherry-picking, or threshold change. S02 indices 101 and 104
remain in every applicable subtest/stratum.

The compatibility wrapper inspects the installed processor signature. It
passes the frozen detector score threshold through `threshold` when that
explicit parameter exists, otherwise through the legacy `box_threshold`
parameter. If neither exists, execution fails closed. The chosen keyword,
callable signature, Transformers version, and wrapper version are recorded in
the v2 runtime identity and future backend reports.

Only the following pre-outcome commands are authorized by this amendment; they
do not load a model, invoke inference, or write an execution receipt:

```bash
.venv/bin/python scripts/repro/validate_piu_s03_v2_amendment.py

.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision_v2.py \
  --execution-plan results/method/piu_s03_perception_decision_execution_plan_v2.json \
  --execution-index 0 --validate-only

.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision_v2.py \
  --execution-plan results/method/piu_s03_perception_decision_execution_plan_v2.json \
  --execution-index 0 --dry-run
```

Outcome-bearing v2 execution requires separate authorization and the explicit
`--allow-outcome-write` flag. Until a complete validated v2 certificate exists,
the public S03 outcome gate remains incomplete and `paper_claim_ready=false`.
