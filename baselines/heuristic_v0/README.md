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
results must be reported as `B2 Heuristic V0` under the frozen B0--B8
comparison registry, never as the proposed method. The historical tag remains
unchanged.

The tag exposes one inference decision, not a full replanning loop. The formal
adapter therefore runs exactly that decision from a hash-bound paired public
capture and executes at most its one selected primitive. It never adds a second
decision on the current branch. A non-shared observation primitive is an
abstention; failure to finish in one supported option remains a failed episode.
`run_piu_heuristic_v0_inference.py` requires an external GPU runtime and records
tree hashes for all five legacy model directories before
`run_piu_heuristic_v0_once.py` creates the B2 episode.
