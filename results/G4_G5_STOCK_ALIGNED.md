# G4/G5 stock-aligned decision

## Controlled setting

The source task is `libero_object[4]`: pick up ketchup and place it in the
basket. The stock task succeeds in 100/100 reproduction episodes. The paired
occlusion benchmark keeps its cameras, robot reset, controller, prompt, object
set, and goal. It moves one or two existing distractors onto evaluator-selected
camera rays. The visible/single/dual conditions score 5/5, 5/5, and 2/5.

## G4: semantic intent calibration

The frozen dataset contains 200 observations and 1600 independently sampled
π0.5 chunks. Each class uses seeds 0--19 for prototypes, 20--39 for conformal
calibration, and 40--49 for held-out validation. Marginal conformal coverage is
0.825, so G4 fails.

A class-conditional Mondrian calibrator was then frozen. A new seed 50--59
audit was collected once and not used for further model selection. Audit
coverage is 0.90 overall, but class coverage is 0.80/0.90/1.00/0.90 for
`ACT`/`MOVE_CLOSER`/`REMOVE_OCCLUDER`/`ROTATE`. The preregistered 0.90 minimum
for every class is not met. Mean set size is 2.1. G4 is NOT-GO.

The failure is structural: `ACT`, `MOVE_CLOSER`, and `ROTATE` produce similar
initial behavior. Requested prompt intent is not the same as executed action
intent. A better classifier cannot manufacture behavioral separation that the
policy does not express.

## G5: executor authorization

The required reliability is 0.90 using a one-sided 95% exact binomial lower
bound.

| Primitive | Trials | Lower bound | Decision |
|---|---:|---:|---|
| `ACT` | 100/100 | 0.970 | GO |
| `MOVE_CLOSER` | 30/30 | 0.905 | GO |
| `REMOVE_OCCLUDER` / open container | 15/30 | 0.339 | NOT-GO |
| `ROTATE` | 0/30 | 0.000 | NOT-GO |

The complete G5 gate is NOT-GO. The router may authorize only `ACT` and
`MOVE_CLOSER`; the other branches must stop or request assistance.

## Paper consequence

The defensible contribution is not a claim that action spread reveals semantic
uncertainty. The evidence supports a capability-gated, class-conditional
decision interface and exposes behavior aliasing as the failure mode of frozen
VLA action sampling. The next method must decode achieved behavior from closed-
loop outcomes or use an independently trained semantic critic.
