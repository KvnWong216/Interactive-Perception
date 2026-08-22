# Heuristic V0 experiment record (legacy)

> These results belong to the frozen baseline and historical outcome critic.
> The new experiment matrix and go/no-go gates are in
> [`research_plan.md`](research_plan.md). No learned candidate-interaction
> performance result exists yet.

The authoritative data split is
`benchmarks/piu_v1/scene_disjoint_protocol.yaml`. The outcome protocol and
seed provenance are under `benchmarks/rss_v1/`. Do not infer split membership
from a script name or an old report.

Current retained datasets are:

| Data | Purpose | Status |
|---|---|---|
| PIU prototype train v3 | fitting | development |
| PIU conformal calibration v3 | thresholds | frozen calibration |
| PIU scene-disjoint v4 | first clean diagnosis | consumed; now development |
| Outcome v12b clean | outcome clean validation | GO |
| Outcome v12b sealed audit | one-time outcome audit | GO |

The outcome critic passed its component gate, but the object-level initial
belief/action model failed its first scene-disjoint gate. Therefore the full
method is NOT-GO and no new sealed PIU audit is authorized.

Every reported success rate must specify prompt, policy inputs, semantic
action, endpoint, sample count, lower bound, oracle status, and whether the
data are development or sealed. Drawer-joint success is never called target
reveal. A visual reveal is never called final task success.

Online-allowed inputs are stock agentview RGB, wrist RGB, public robot state,
the complete prompt, and public visual/action history. Segmentation, hidden
poses, joints, target identity/position, task predicates, and global cameras
are evaluator-only and must be read after controller termination.
