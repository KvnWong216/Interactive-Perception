# S03 public execution v3 amendment

This is a prospective, pre-outcome execution amendment. It does not change the
frozen 620 logical records, their order or IDs, evaluator thresholds, the
public-input firewall, or any S02 artifact.

## Sealed v1 and v2 history

S03 public v1 and v2 are separate infrastructure-blocked histories. In each
version, execution index 0 was started and closed exactly once as an
infrastructure failure. Neither attempt produced a prediction that entered a
scientific result, a model/task outcome, or an S03 certificate. Both index-0
attempts are consumed and are ineligible for rerun, relabeling, deletion, or
reuse under their original execution versions.

The immutable v2 blocker is
`results/diagnostics/piu_s03_v2_execution_blocker_v1.json` (SHA-256
`6bc48e54e0bb114fef8fffc3bcdf56144143c16842fc6b372cd429a341b710b8`).
The v2 partial execution tree SHA-256 is
`017220a2c71d0378f8a6d21cfbab37896ada45e41410fc9bd6c4bb5dea236d25`.
Its A/B/C metrics are unavailable, not passed or failed. The fact that zero
`AMBIGUOUS` predictions were generated is not an interpretable ambiguous rate,
because v2 produced zero outcome-evaluable records.

## Frozen v3 runtime and identity

V3 fixes two infrastructure defects prospectively. First, the VLM dependency
set now locks SentencePiece 0.2.1, Protobuf 6.33.5, and TorchVision 0.28.0 in
addition to the previously frozen packages. Second, processor-only readiness
must initialize the local SigLIP tokenizer and every Grounding DINO, SAM,
DINOv2, SigLIP, and Qwen processor/config without loading model weights or
executing a prediction.

The frozen readiness contract is
`configs/experiments/piu_s03_runtime_readiness_v3.json` (SHA-256
`b0064934ca96a618016495dd89f216d9c9233e49fa491753be36e37db6f8231d`).
The model identity is
`configs/experiments/piu_s03_model_identity_v3.json` (SHA-256
`9a44bfab4e52833ab5257146dc6e279b4df30e3c13935ec7ed5fc6fb3641a433`).
The runner identity is
`configs/experiments/piu_s03_runner_identity_v3.json` (SHA-256
`9fe1357d28b9c2bc80a52c04c05e68e01514aa320226ddba5f53aafd405d0112`).
The 620-record execution plan is
`results/method/piu_s03_perception_decision_execution_plan_v3.json`
(SHA-256
`130bf3f51210519daed115b6dfa675cb9dbe57d4dafc5ae72c54789d25700eca`).

V3 uses all 620 original logical records exactly once in their original order.
Historical v1/v2 attempts are audit evidence excluded from v3 performance
statistics. There is no filtering, replacement, cherry-picking, or threshold
change. S02 indices 101 and 104 remain included in every applicable stratum.

## Lifecycle semantics

The validator distinguishes five states:

1. `FROZEN_READY_BEFORE_OUTCOMES`: zero closed receipts and no certificate;
2. `EXECUTION_IN_PROGRESS`: a valid ordered receipt chain has started but has
   fewer than 620 closed receipts;
3. `EXECUTION_BLOCKED_INFRA`: an append-only infrastructure blocker seals the
   partial chain and no certificate exists;
4. `EXECUTION_COMPLETE_PENDING_CERTIFICATE`: all 620 receipts are valid and no
   certificate exists;
5. `CERTIFIED`: all 620 receipts and the certificate both validate.

An output root is not itself evidence of pre-freeze leakage after execution is
authorized. The validator follows and verifies the receipt chain, rejects
rerun/overwrite/skip/duplicate operations, and rejects any certificate before
620 valid closed receipts. Identity validation remains valid during an
append-only execution; it no longer conflates execution progress with a frozen
identity mutation.

The following commands are the only v3 commands authorized by this amendment:

```bash
.venv/bin/python scripts/repro/preflight_piu_s03_backend_v3.py

.venv/bin/python scripts/repro/validate_piu_s03_v3_amendment.py

.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision_v3.py \
  --execution-plan results/method/piu_s03_perception_decision_execution_plan_v3.json \
  --execution-index 0 --validate-only

.venv/bin/python scripts/pipeline/run_piu_s03_perception_decision_v3.py \
  --execution-plan results/method/piu_s03_perception_decision_execution_plan_v3.json \
  --execution-index 0 --dry-run
```

These commands do not authorize inference. A future outcome-bearing run needs
separate authorization and `--allow-outcome-write`. At this freeze, v3 has no
receipt root, prediction, outcome, blocker, or certificate;
`paper_claim_ready=false`.
