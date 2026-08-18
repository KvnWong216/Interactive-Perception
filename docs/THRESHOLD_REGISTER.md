# Threshold register

The deployed router has no confidence trigger. It minimizes expected task risk
over calibrated beliefs and action effects. The remaining constants define
evaluation, calibration, or low-level controller completion:

| Constant | Role | Chosen from test data? | Online routing input? |
|---|---|---:|---:|
| conformal error `0.05` | desired class-conditional set coverage | no | through the frozen v10 conformal set |
| action lower bound `0.80` | capability gate | owner-specified before audit | no |
| original action bound `0.90` | strict sensitivity report | no | no |
| target pixels `256` | one pi0.5 visual-token footprint | no; derived from 256/16 image-token geometry | no |
| drawer joint `-0.14` | evaluator opening label | no; mechanism endpoint | no |
| return pose `0.012 m`, `0.08 rad` plus registered public observation pose | motor/observation diagnostic | frozen before collection | controller only |
| information cost `0.10` | normalized task cost | frozen before paper test | yes, as a loss rather than a trigger |

The first two statistical constants decide whether evidence is strong enough
to publish or enable an action. They do not inspect an episode and decide when
to interact. The pixel and joint constants create offline ground truth and are
forbidden policy inputs. The return tolerances stop a proprioceptive controller;
the v10 information endpoint is determined by persistent visual evidence rather
than the final motor pose. The paper reports the preregistered cost
sweep, so the main claim cannot depend on one favorable cost.

The earlier five-pixel label remains in the immutable v3 diagnostic artifact;
it counted 5--7 leaked pixels as useful evidence and caused v8 to fail. The v9
contract uses one visual-token footprint instead, and reports sensitivity at
128, 256, and 300 pixels. This changes evaluator ground truth, not online
routing.

Zero target pixels do not by themselves imply `EMPTY`. The temporal evaluator
also requires same-camera-pose counterfactual target visibility at an aligned
public history point. Final drawer state and exact return pose remain separate
motor diagnostics. This is a logical evidence contract,
not a scalar threshold tuned on model output.

Conformal score cutoffs are the only learned thresholds. They are finite-sample
quantiles fit on the independent calibration split, not hand-tuned confidence
values. An ambiguous prediction set causes `SAFE_STOP`; it is never collapsed
with an additional threshold.
