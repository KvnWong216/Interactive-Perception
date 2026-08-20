# PIU original-scene fresh execution

## Result

The disposable seed-1399 run closes the prompt-relevant information-acquisition
loop once in the original cluttered LIBERO scenario:

```text
"Place the butter in the basket"
  -> public agentview + wrist RGB
  -> Grounding DINO / SAM / DINOv2 object packet
  -> frozen SigLIP prompt-conditioned belief and uncertainty
  -> Qwen registered action-effect predictions
  -> continuous expected-utility selection: OPEN_CONTAINER
  -> freshly sampled frozen pi05_libero OPEN_AND_OBSERVE
  -> six public RGB observations
  -> v13 complementary fusion: singleton REVEALED
  -> public belief update
  -> replan: MOVE_CLOSER
  -> INFORMATION_ACQUIRED
```

This run does not execute `MOVE_CLOSER` or the final butter placement. The
primary target-observability endpoint is 1/1; final-task success is not measured
in this trace. The run is development-only, not clean or sealed evidence.

The fresh physical option report was launched from the retained v8 initial
report. The corrected current parser was then rerun on bit-identical public RGB
and selected the same `OPEN_CONTAINER` target; both reports are retained for
provenance, and the assembled trace links the corrected report to the rollout.

## Camera contract

Stock `pi05_libero` receives its native `agentview`, wrist RGB, public robot
state, and semantic subtask. PIU also uses both public RGB streams. They are
complementary sensors: a positive reveal in either view is retained, and
visibility failure in one view does not negate positive evidence in the other.
`EMPTY` requires both target heads to be singleton-negative and the independent
coverage head to be singleton-completed.

No online component reads simulator segmentation, drawer joints, target poses,
semantic IDs, task predicates, BEV, or global cameras. The separate evaluator
replay after controller termination measured a minimum middle-drawer joint of
-0.1613 and maximum target footprint of 818 pixels. These values score the run;
they did not choose, stop, classify, or update an online action.

## Canonical artifacts

- Initial corrected PIU report:
  `results/diagnostics/piu_messy_corrected_initial_seed1399_v1.json`
- Fresh π0.5 option and public-RGB outcome:
  `results/diagnostics/piu_messy_fresh_e2e_seed1399_v1.json`
- Independently rescored v13 outcome:
  `results/diagnostics/piu_messy_fresh_e2e_seed1399_v1_public_rgb_outcome_v13.json`
- Corrected post-action belief/replan:
  `results/diagnostics/piu_messy_corrected_post_open_seed1399_v1.json`
- Machine-readable assembled trace:
  `results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition_trace.json`
- Public demo:
  `results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition.mp4`
- Contact sheet:
  `results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition_contact_sheet.png`

## Reproduction

The wrapper creates new immutable output paths and refuses to overwrite an
existing run:

```bash
EXPERIMENT_GPU_INDEX=0 EXPERIMENT_ALLOW_LOCAL_RUSTDESK=1 \
  bash scripts/run_piu_original_demo.sh my_run_id
```

Required local checkpoints are intentionally not stored in Git:

- `checkpoints/perception/grounding-dino-tiny`
- `checkpoints/perception/sam-vit-base`
- `checkpoints/perception/dinov2-small`
- `checkpoints/perception/siglip-base-patch16-224`
- `checkpoints/perception/qwen2.5-vl-3b-instruct`
- the sibling `openpi` checkout and `pi05_libero` checkpoint

The wrapper runs GPU preflight before every perception/VLM stage. The fresh
π0.5 runner performs its own preflight and launches a new policy server.

## Status

- Information acquisition: **GO for one disposable development execution**.
- Complementary v13 clean validation: **NOT-GO / not run**.
- Physical `MOVE_CLOSER` after this replan: **NOT-GO / not run**.
- Final butter placement: **NOT-GO / not run in this trace; earlier attempts failed**.
- Paper result: **NOT-GO** until clean v13 evaluation and the preregistered main
  matrix are complete.
