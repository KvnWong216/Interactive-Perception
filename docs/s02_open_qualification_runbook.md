# S02 OPEN qualification runbook

The S02 design and the S02 input allocation are different immutable objects.
The design fixes the exact-binomial test (`n=124`, reject at 122 successes)
without formal data. The input allocation fixes the 124 group IDs, simulator
seeds, reset-only NPZ states, public initial observations, model-free OPEN
stimuli, and hash-permuted execution order. Input freezing performs zero
`env.step` calls, does not call pi0.5, and creates no qualification result.

The canonical pre-outcome artifacts are:

- `results/method/piu_open_primitive_qualification_plan_v1.json`
- `data/piu/mainline_v1/open_primitive_qualification_v1/split_manifest.json`
- `data/piu/mainline_v1/open_primitive_qualification_v1/seed_inventory.json`
- `data/piu/mainline_v1/open_primitive_qualification_v1/groups/`
- `results/method/piu_open_primitive_qualification_schedule_v1.json`

The schedule is immutable after it is written. A runtime failure after a
single-use `STARTED` ticket is created remains in the denominator; seeds,
states, controller decisions, and ordering must never be replaced in response
to execution behavior.

Formal execution is the next, separate phase. Only after explicit authorization
to run the 124 attempts, verify the identified endpoint and execute in the
frozen order:

```bash
for execution_index in $(seq 0 123); do
  .venv/bin/python scripts/pipeline/run_piu_primitive_qualification.py \
    --schedule results/method/piu_open_primitive_qualification_schedule_v1.json \
    --execution-index "${execution_index}" \
    --host 127.0.0.1 --port 8002
done
```

Do not run S03 after schedule creation. S03 remains blocked until
`scripts/evaluation/evaluate_piu_primitive_qualification.py` recomputes all 124
scheduled results and produces a replay-valid `FORMALLY_QUALIFIED` certificate.
An input schedule is not performance evidence, and `paper_claim_ready` remains
false.
