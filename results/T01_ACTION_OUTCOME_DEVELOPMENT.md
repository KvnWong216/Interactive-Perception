# T01 action-outcome development

The drawer joint moved in 97/100 earlier trials. This is an opening-capability
result, not proof that the policy acquired prompt-relevant visual information.

The paired transition dataset contains 180 calibration-only episodes over
seeds 400--459. Under the stricter policy-camera endpoint:

| Intended outcome | Correct | One-sided 95% lower bound |
|---|---:|---:|
| `REVEALED` | 57/60 | 0.876 |
| `EMPTY` | 56/60 | 0.854 |

`EMPTY` is conservative: it additionally requires the seed-matched
target-present run to reach `REVEALED`. An open joint alone is not an empty-space
certificate.

Three frozen-feature critics were tested on the development diagnostic split:

| Critic | Coverage | Singleton-correct | Decision |
|---|---:|---:|---:|
| global prefix v1 | 60/60 | 21/60 | NOT-GO |
| spatial prefix v2 | 18/21 | 13/21 | NOT-GO |
| target-query + standardized spatial v3 | 20/21 | 15/21 | NOT-GO |

For v3, `REVEALED` coverage is 7/7 but singleton-correct is only 2/7; `EMPTY`
coverage is 6/7. Seeds 453--459 were inspected during head development and are
diagnostic only. Audit seeds 500--599 remain untouched.

The next valid design is a temporal outcome head: a short wrist/agentview
history plus robot proprioception and the executed information action. It must
be frozen before a new confirmation split is collected. No confidence
threshold may substitute for this missing evidence.

## Post-open failure diagnosis and executor revision

The three strict `REVEALED` failures (seeds 416, 436, and 438) are not drawer-
opening failures. Their middle layers reach approximately -0.161, but both
policy cameras contain zero target pixels at the terminal arm pose. The arm and
gripper remain at the cabinet and block or miss the newly exposed interior.

An evaluator-only intervention opened the same layer while leaving the arm at
the stock reset pose. Target visibility was then 60/60, with a one-sided 95%
lower bound of 0.951. A new proprioceptive `RETURN_TO_OBSERVE` controller was
also tested from three fixed non-home perturbations over seeds 600--629. It
returned to the stock pose and restored target visibility in 30/30 trials,
with lower bound 0.905. Neither diagnostic authorizes the executor: the former
opens the drawer through simulator state, and the latter does not start from a
real pi0.5 post-open trajectory.

The preregistered replacement is therefore:

```text
OPEN_AND_OBSERVE = frozen pi0.5 OPEN_CONTAINER
                 + proprioceptive RETURN_TO_OBSERVE
```

It keeps both policy camera extrinsics unchanged. The return stage reads only
end-effector pose and gripper state; drawer joints, target pose, segmentation,
and labels remain evaluator-only. Version-1 audit seeds 500--599 stay sealed.
The new development split is 600--659 and its future audit is 700--799. Only a
continuous physical rollout on that split can replace the current 57/60 and
56/60 effect rates.
