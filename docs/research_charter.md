# PIU research charter

## Frozen main line

The project studies one causal chain: whether a robot can recognize that the
current observation is insufficient for a prompt, acquire the missing evidence
through physical interaction, bind that evidence to the requested target, and
use it in the next manipulation. Benchmark construction, effect prediction,
calibration, and executor primitives are supporting instruments, not separate
research stories.

The first method question is deliberately restricted to the unchanged T01D
hidden-butter drawer scene:

```text
closed drawer -> OPEN -> butter evidence -> prompt-conditioned binding
              -> target contact -> PICK -> PLACE
```

No additional scenario family is activated until the acquisition-to-utilization
bottleneck has a positive causal ceiling or is routed to a qualified dedicated
PICK primitive.

## Scientific claim hierarchy

1. Retained evidence establishes a descriptive gap: OPEN succeeds in 9/10
   groups, produces a nonempty target mask in 8/10, and raw post-OPEN DIRECT has
   zero target grasp contacts.
2. The evaluator-only visual-prompt experiment asks whether target indication
   can causally change the frozen executor. It is an oracle ceiling, never the
   public-input method.
3. Only after a prospectively sized formal oracle test is positive may a learned
   public-RGB spatial binder be activated as the main-method candidate.
4. A learned binder must improve both a spatial binding metric and downstream
   target contact on disjoint groups. Target contact alone is information
   utilization, not direct proof of binding accuracy.
5. Full-task success is not claimed until PICK and PLACE are separately
   qualified and the entire loop is executed.

## Non-negotiable boundaries

- Policy inputs contain only first-person RGB, the prompt, public robot state,
  public action history, and public candidate descriptions.
- Simulator masks, instance IDs, object poses, contacts, joint truth, and task
  predicates remain in separately stored training labels/evaluator sidecars or
  an explicitly labeled oracle column.
- Retrospective seeds 1400--1409 are development evidence. They cannot be
  renamed as calibration or test data.
- The four-token global-mean PaliGemma representation remains a rejected
  baseline. The successor retains token masks, camera spans, spatial indices,
  and pre/post order.
- Belief entropy is not a success metric unless the belief variable has an
  independently validated label and calibration protocol.
- No hand-weighted information value is used in this sprint. Future value terms
  require either learned parameters or a frozen, preregistered user/physics
  contract plus ablations.
- Engineering budgets and primitive enablement gates are reported as such; they
  are not statistical evidence for a paper claim.

## Two-day sprint exit gate

The offline exit is a validated, leakage-controlled transition/evaluation
pipeline plus an external-GPU full-prefix extractor and a lightweight spatial
binder implementation. The empirical exit additionally requires the external
identified pi0.5 endpoint: nine development screen runs, five independent pilot
runs, and a prospectively frozen formal sample-size plan.

If the endpoint is absent, software completion is reported separately from the
unrun causal experiment. No replay, dry run, or retained result is relabeled as
new method evidence.
