# PIU V0 full-pipeline asset index

Source rollout: `results/smoke/piu_v0_full_pipeline_v4_seed1399.json`.

## Primary demo

- `visualizations_v1/piu_v0_full_pipeline_seed1399.mp4`: all five cases in
  sequence. The left panel is wrist RGB; the right panel is a post-terminal,
  evaluator-only global replay. Outcome and terminal remain `pending` during
  action playback and appear only at the terminal frame.

## Per-case assets

Each directory under `visualizations_v1/` contains:

- a wrist/global replay video;
- public six-point RGB storyboard;
- structured belief before/after plots;
- V0 node uncertainty maps;
- explicit action-utility comparison;
- learned action-effect distribution;
- evaluator-only target-pixel timeline.

The five cases are `hidden_butter`, `same_rgb_visible_cream_cheese`,
`open_visible_butter`, `middle_drawer_empty`, and
`drawer_action_failed_control`.

## Interpretation limits

The maps are structured task-fact/node maps, not Grounding-DINO/SAM pixel
grounding. `DIRECT_ACT` is a semantic handoff and was not physically executed.
This disposable seed-1399 run is an engineering smoke, not clean or sealed
evidence. Exact hashes and the visualization contract are recorded in
`visualizations_v1/assets_manifest.json`.
