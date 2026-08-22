# Heuristic V0 method (legacy baseline)

> This document describes the frozen pre-redesign baseline. The proposed
> candidate-conditioned calibrated method is specified in
> [`research_plan.md`](research_plan.md) and
> [`ADR-0001`](adr/0001_candidate_conditioned_calibrated_interaction.md).
> Do not add weights, rules, or selectors to this legacy pipeline.

PIU maintains a prompt-conditioned structured belief over task facts and an
object/region uncertainty field. Grounding DINO proposes open-vocabulary
regions, SAM supplies masks, DINOv2 supplies region features, and frozen
SigLIP aligns those observations with the full task prompt. Qwen2.5-VL reasons
only over registered candidates and predicts their possible public visual
effects.

For candidate action `a`, the planner compares expected task uncertainty after
the action with the current stop risk:

```text
Q(b,a) = cost(a) + sum_y P(y | b,a,h) V(b[a,y], m[a,y])
U_res(a | b,m) = V_stop(b,m) - Q(b,a)
```

The selected output is a typed semantic option such as `MOVE_CLOSER`,
`NEXT_BEST_VIEW`, `REMOVE_OCCLUDER`, `OPEN_CONTAINER`, or `ACT`. The frozen
pi0.5 policy converts its semantic prompt to low-level actions. After every
option, a public-proprioception controller returns to the initial end-effector
pose while retaining all intermediate RGB evidence.

Information outcomes use exactly six public history points: before, quarter,
half, three-quarters, option end, and returned home. The hierarchical critic
first separates failed versus completed observation coverage, then target
revealed versus local empty. A non-singleton set always yields `SAFE_STOP`.

Scenario identities, poses, prompts, seeds, targets, joints, and action priors
live in YAML or CLI arguments. Runtime modules contain no T01-specific value.
