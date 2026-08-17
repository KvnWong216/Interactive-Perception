# Tonight's frozen experiment

The pipeline is now an explicit loop:

```text
prompt + stock policy observations
  -> typed conformal world belief
  -> temporal-progress belief
  -> capability-gated action effects
  -> expected-risk decision
  -> frozen VLA information option
  -> prompt-attended six-point RGB/proprioceptive history
  -> conformal temporal FAILED / REVEALED / EMPTY outcome
  -> update both beliefs or SAFE_STOP
```

`OPEN_AND_OBSERVE` is the first complete information option. It combines the
frozen pi0.5 drawer-opening prompt with a policy-visible return-to-observation
controller. Its physical effect and its visual outcome critic are separate
gates.

Run the CPU-only preflight:

```bash
bash scripts/run_rss_experiment_ladder.sh preflight
```

Then run the three-episode wiring check and development calibration:

```bash
bash scripts/run_rss_experiment_ladder.sh smoke
bash scripts/run_rss_experiment_ladder.sh development
```

Development v3 used seeds 600--659. `REVEALED` checks all six public history
points, while `EMPTY` additionally requires same-camera-pose counterfactual
visibility coverage.
The head is selected only on 600--619,
conformal calibration uses 620--652 (33 samples per outcome class), and
653--659 was the held-out development decision. V8 failed that decision:
FAILED 7/7, REVEALED 7/9, and EMPTY 4/5 coverage. The audit remains sealed.

That failure showed that the five-pixel v3 label counted 5--7 leaked target
pixels as useful evidence, while every real middle-layer reveal occupied at
least 311 pixels. The v9 candidate uses one frozen visual-token footprint
(256 pixels) as the resolvability definition. Seeds 653--659 are now diagnostic
only; untouched 660--699 must pass as a clean extension before v9 can freeze.

The earlier 700--799 run stopped at 77 rows after exposing a final-frame-only
label bug and is permanently debug-only. Seeds 900--999 are the new one-time
audit and remain sealed unless the untouched 660--699 v9 extension passes all
physical and critic gates. After inspecting that frozen v9 artifact, explicitly
authorize the audit:

```bash
ALLOW_SEALED_AUDIT=1 bash scripts/run_rss_experiment_ladder.sh audit
```

The audit is not the paper test. Seeds 800--899 are reserved for the later
oracle-free closed-loop comparison. Final-task success remains a separate hard
gate because post-reveal butter retrieval is currently 0/5.

If and only if the audit passes, start a fresh extended server and verify that
online prefix features reproduce all 21 held-out development conformal sets.
Then run one non-paper seed and render its wrist-left/global-right demo:

```bash
bash scripts/run_t01_closed_loop_v5.sh smoke
```

Only after that wiring check passes, restart the server and consume the frozen
held-out pipeline-validation seeds:

```bash
bash scripts/run_t01_closed_loop_v5.sh heldout
```

T01D is a custom scenario and is calibration-only under the owner's scope
rule. Seeds 800--899 test the frozen pipeline without refitting, but are marked
`paper_eligible: false`; they are not the paper's final benchmark test set.

The owner revised the physical action reliability lower bound from 0.90 to
0.80 on 2026-08-17. Conformal error remains 0.10. Every result reports both
the new 0.80 decision and whether it would pass the original 0.90 standard.

The exact methods, metrics, split contracts, and hard stops are frozen in
`benchmarks/rss_v1/experiment_v1.yaml`.
