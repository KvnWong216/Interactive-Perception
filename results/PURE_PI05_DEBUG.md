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
| G3 | T01, open middle layer | 3/5 | skill observed; reliability not certified |
| G4 | calibration audit | missing frozen VLM and held-out calibration | NOT-GO |
| G5 | T01, explicit search then retrieval | 0/5 | composition fails |
| G5 | T01, final goal only | 0/5 | task fails |

The explicit condition opened the middle drawer in 1/5 episodes but completed
retrieval in 0/5. The final-goal condition completed retrieval in 0/5. The
capability condition opened the middle drawer in 3/5.

The exact one-sided 95% binomial lower bound for 3/5 is 0.189. This debug
sample therefore cannot certify a high-reliability executor requirement. For
example, a preregistered 0.8 requirement fails. Required reliability is a
paper-level risk choice and is deliberately a required CLI argument, not a
code default.

## Debug conclusion

The earlier 0/5 drawer result combined prompt mismatch, cabinet geometry,
missing visual context, and a wrapper-dependent evaluation path. These were
code and experiment-design problems. After fixing them, pure π0.5 opens the
drawer in 3/5 trials, so neither a broken action client nor an over-strict
endpoint explains the remaining failures.

The remaining failure is best described as context-sensitive and
composition-sensitive transfer. Stock drawer opening is 5/5, while the custom
scene is 3/5 under a direct command and 0/5 on the full task. This is consistent
with shortcut dependence or overfitting, but five seeds cannot establish broad
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

This is the intended solution to G4, but G4 remains NOT-GO today: existing
traces are test-set traces and cannot be reused as calibration data. The next
GPU collection must use disjoint stock/control scenes with evaluator-provided
true intent labels. Conformal coverage is a guarantee about semantic intent
sets under exchangeability, not a guarantee of robot task success.

For online routing, semantic decoding must resolve only public anchors such as
drawer fronts, occluders, and placement regions. The existing evaluator traces
also contain `task_target` anchors for scoring; those reveal the hidden target
pose and must never enter the controller. A conformal layer does not make that
oracle input permissible.

The pipeline now treats these as separate gates: semantic conformal coverage
says whether the correct intent is in the set; `check_capability_gate.py` says
whether π0.5 can reliably realize the selected intent in context. Both must
pass before an autonomous information action is authorized.
