# Pure π0.5 debug and gate report

## Controlled path

`run_pure_pi05_scenario_sr.py` uses `OffScreenRenderEnv`, the stock 256×256
LIBERO cameras, openpi's 180° image rotation, padded 224×224 inputs, eight-state
robot vector, five-action replanning, and the frozen `pi05_libero` checkpoint.
It does not import the benchmark rollout, segmentation, uncertainty probes, or
the router. Only the BDDL scene and instruction change.

## Results

| Gate | Condition | Result | Decision |
|---|---|---:|---|
| G1 | stock LIBERO reproduction | 20/20 | GO |
| G2 | visible-target camera control | 5/5 | GO |
| G3 | T01, open middle layer | 15/30 | NOT-GO: 0.90 reliability not certified |
| G4-v1 | LIBERO ACT vs REMOVE_OCCLUDER | validation 20/20, mean set 1.0 | GO in binary scope |
| G5 | T01, explicit search then retrieval | 0/5 | composition fails |
| G5 | T01, final goal only | 0/5 | task fails |

The explicit condition opened the middle drawer in 1/5 episodes but completed
retrieval in 0/5. The final-goal condition completed retrieval in 0/5. The
capability condition opened the middle drawer in 15/30.

The exact one-sided 95% binomial lower bound for 15/30 is 0.339. It fails the
preregistered 0.90 minimum reliability requirement. This is a hard NOT-GO, not
a threshold to tune after seeing the result.

Joint traces rule out a scoring-boundary explanation. Across the 15 successes,
middle-layer displacement is 0.1420–0.1492 (median 0.1450). Across the 15
failures it is 0.0000005–0.0000071 (median 0.0000028). One failed seed moves
the bottom layer by 0.0725; the other failures do not pull a drawer. Raw
episodes and the gate report are frozen under `results/capability/`.

A replay of typical failure seed 1 is stored at
`results/demos/T01_multi_drawer_search_capability_seed001.mp4`. It again runs
530 steps without opening a layer; middle-layer displacement is 0.00000393.
The matching replay record is frozen under `results/capability/`.

## Debug conclusion

The earlier 0/5 drawer result combined prompt mismatch, cabinet geometry,
missing visual context, and a wrapper-dependent evaluation path. These were
code and experiment-design problems. After fixing them, pure π0.5 opens the
drawer in 15/30 trials, so neither a broken action client nor an over-strict
endpoint explains the remaining failures.

The remaining failure is best described as context-sensitive and
composition-sensitive transfer. Stock drawer opening is 5/5, while the custom
scene is 15/30 under a direct command and 0/5 on the full task. This is
consistent with context shortcut dependence, but does not establish broad
model overfitting.

`SegmentationRenderEnv` is not required for policy execution. It remains useful
for evaluator-only visibility measurements, but all capability conclusions
above come from the pure off-screen path.

## Main-experiment blocker

The router still contains plumbing defaults: confidence 0.5, absence mass
0.05, false-absence loss 4.0, clutter mass 1.9/0.1, and a 120-step fixed-rule
prefix. They must be learned or calibrated on held-out scenes, or covered by a
predeclared sensitivity study. Router results produced before that are smoke
tests, not paper evidence.

## G4 implementation

`semantic_conformal.py` replaces a raw confidence cutoff with a conformal set
over `ACT`, `REMOVE_OCCLUDER`, `ROTATE`, `MOVE_CLOSER`, and `NOT_FOUND`.
Calibration uses the score `1 - p(true intent)` and the finite-sample corrected
split-conformal quantile. The artifact records the policy ID, split ID, alpha,
sample count, and attainable coverage resolution. The CLI refuses fewer than
30 held-out examples by default.

G4-v1 was fit on a newly collected, frozen 100-observation LIBERO dataset: 40
samples learn black-box action prototypes, 40 separate samples calibrate at
alpha 0.1, and 20 held-out samples validate the artifact. Validation coverage
is 1.0 (20/20) with mean prediction-set size 1.0. Dataset SHA-256 is
`6873f44e46c903ebecf2b5aa10b8f91ff740dea3629e189f3ac81e1a2dd5db86`.

This is GO only for `ACT` versus `REMOVE_OCCLUDER` under the collected LIBERO
distribution. `NOT_FOUND`, `ROTATE`, and `MOVE_CLOSER` remain uncalibrated.
Conformal coverage is a guarantee about semantic intent sets under
exchangeability, not a guarantee of robot task success.

For online routing, semantic decoding must resolve only public anchors such as
drawer fronts, occluders, and placement regions. The existing evaluator traces
also contain `task_target` anchors for scoring; those reveal the hidden target
pose and must never enter the controller. A conformal layer does not make that
oracle input permissible.

The pipeline now treats these as separate gates: semantic conformal coverage
says whether the correct intent is in the set; `check_capability_gate.py` says
whether π0.5 can reliably realize the selected intent in context. Both must
pass before an autonomous information action is authorized.
