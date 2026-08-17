# Prompt-to-action lessons from task-oriented grasping

Three task-oriented grasping systems support a modular design, but none solves
our active-perception decision.

| Work | Prompt-to-action path | Useful here | Boundary |
|---|---|---|---|
| [TaskGrasp](https://proceedings.mlr.press/v155/murali21a.html) | object + task semantics rank grasp candidates | separate executable candidates from task-conditioned utility | supervised grasp labels; target is already available |
| [LERF-TOGO](https://proceedings.mlr.press/v229/rashid23a.html) | language field finds object, DINO groups it, conditional language query ranks grasps on parts | ground the object/region before scoring an action | multi-view 3D reconstruction; no decision about acquiring missing evidence |
| [ShapeGrasp](https://arxiv.org/abs/2403.18062) | RGB-D mask becomes a part graph; an LLM first names parts, then scores task utility | separate grounded state identification from task reasoning | assumes a visible segmented target and user-supplied mask points; no calibrated uncertainty |

Their shared lesson is factorization:

```text
ground what exists -> build feasible candidates -> score candidates for the prompt
```

Our missing piece comes before their grasp selector:

```text
ground what is still unknown
  -> identify the cause of missing evidence
  -> keep only physically certified information actions
  -> predict each action's observable result
  -> choose by expected task-risk reduction
```

This directly motivates the current implementation:

1. A prompt-conditioned target-state head supplies typed hypotheses instead of
   decoding semantics from raw action spread.
2. A capability registry generates only actions the frozen VLA can execute in
   the declared context.
3. Prompt-attended spatial history grounds the action's visible effect.
4. A hierarchical critic first recognizes completion, then distinguishes
   `REVEALED` from local `EMPTY` evidence.
5. Expected-risk planning ranks actions; language similarity never bypasses
   physical reliability or temporal preconditions.

We should not copy their masks, point clouds, object identities, or part names
from the simulator into the online policy. Those would answer the uncertainty
question before the method acts. Their strongest paper-facing use is therefore
the modularity precedent and the following ablations: prompt-blind candidate
ranking, no spatial grounding, flat rather than hierarchical outcome
classification, and no capability/effect model.
