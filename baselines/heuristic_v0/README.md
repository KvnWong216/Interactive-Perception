# Heuristic V0 baseline

The pre-redesign PIU system is frozen at Git commit `e7db12b7f35d9be416fc3ed57d36b12560e40cf0`
and Git tag `baseline/heuristic-v0`. It includes Grounding DINO, SAM, DINOv2,
SigLIP, Qwen structured scoring, hand-composed uncertainty/utility, the
deterministic selector, and the original pi0.5 bridge.

It remains a baseline and engineering sanity check. No new rules, category
vocabulary, weights, or selector branches may be added. To reproduce it without
mixing legacy dependencies into the learned method:

```bash
git worktree add ../Interactive-Perception-heuristic-v0 baseline/heuristic-v0
```

The learned reference path lives in `src/calibrated_interaction/`. Legacy
results must be reported as `B1 Heuristic V0`, never as the proposed method.
