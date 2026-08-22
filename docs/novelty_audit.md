# Novelty gate

Audit cutoff: **2026-08-22**. `Y` means explicitly supported by the audited
paper; `N` means absent or outside its task; `P` means partial; `?` means the
public paper/abstract did not establish it. A question mark is not treated as a
positive novelty comparison.

| method | 1 prompt defines information | 2 calibrated uncertainty | 3 compares physical candidates | 4 action-conditioned information effect | 5 continuous robot action | 6 pretrained VLA | 7 articulated containers | 8 open-vocabulary target | 9 privileged semantic API online | 10 manipulation, not active viewpoint | 11 task execution, not only QA | 12 counterfactual action supervision | 13 coverage/reliability guarantee | 14 OOD object/scene/prompt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PROBE | Y | N | P | N | Y | N | P | Y | N | Y | N | N | N | Y |
| LIBERO-Occ | Y | N | N | N | Y | Y | Y | P | N | N | Y | N | N | Y |
| CoMe / Act-Sense-Act | Y | N | N | N | Y | Y | P | Y | N | P | Y | N | N | Y |
| ZS-IP | Y | N | P | N | Y | N | N | Y | N | Y | P | N | N | Y |
| CNABU | N | P | Y | Y | Y | N | N | N | N | P | N | Y | N | Y |
| Dengler et al. | N | P | Y | P | Y | N | N | N | N | P | N | P | N | Y |
| KnowNo | Y | Y | Y | N | P | N | P | Y | N | Y | Y | N | Y | Y |
| UAOR | Y | N | N | N | Y | Y | P | Y | N | N | Y | N | N | Y |
| SCALE | Y | N | N | N | Y | Y | P | Y | N | N | Y | N | N | Y |
| SaPaVe | Y | N | N | N | Y | Y | P | Y | N | N | Y | N | N | Y |
| ActiveVLA | Y | N | P | N | Y | Y | Y | Y | N | N | Y | N | N | Y |
| VLMPC | Y | N | Y | Y | Y | N | P | Y | N | Y | Y | P | N | Y |
| proposed reference method | Y | Y | Y | Y | Y | Y | required in benchmark | Y | **N by contract** | Y | Y | Y | Y | required group splits |

Sources and publication status are recorded in
[`literature_lineage.md`](literature_lineage.md). Particularly important
threats are the 2026 preprints [PROBE](https://arxiv.org/abs/2608.17129),
[ZS-IP](https://arxiv.org/abs/2602.18374), and
[LIBERO-Occ](https://arxiv.org/abs/2606.10862), the 2025 CNABU
[paper](https://arxiv.org/abs/2502.20606), and the peer-reviewed 2026
[SaPaVe](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SaPaVe_Towards_Active_Perception_and_Manipulation_in_Vision-Language_Action_Models_CVPR_2026_paper.pdf)
and [ActiveVLA](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.pdf).

## Gate decision

**GO for a falsification-oriented reference implementation; NO-GO for broad
novelty language.** These claims are already occupied and must not appear as
our contribution:

- a benchmark for occluded manipulation (LIBERO-Occ, PROBE, ActiveManip-Bench);
- manipulation to reveal hidden evidence (Mechanical Search, SMS, PROBE,
  ZS-IP, CNABU/Dengler);
- a VLM choosing an exploration action (ZS-IP, PROBE, CoMe);
- active-perception behavior inside a VLA (CoMe, SaPaVe, ActiveVLA);
- action-conditioned future/belief prediction by itself (VLMPC, CNABU);
- conformal robot planning by itself (KnowNo and later conformal planners).

The remaining conjunction is narrower:

> Given a user's task prompt, temporal first-person RGB, public action-observation
> history, and a registry of executable manipulation primitives, learn each
> candidate's task-relevant physical effect; calibrate the route decision; then
> dispatch a short structured subtask to a frozen continuous-action VLA and
> replan from the post-action observation.

No single audited work establishes this entire conjunction. This is an
inference from the matrix, not a claim made by any cited paper.

## Required head-to-head tests

The novelty gate stays open only if all four comparisons are run on identical
candidate sets, executor, scene initializations, and public observations:

1. prompted VLM router versus learned route-only versus learned route+effect;
2. KnowNo-only abstention versus calibrated effect-aware physical resolution;
3. counterfactual effect supervision versus route labels alone;
4. frozen π0.5 direct versus the complete post-action replan loop.

The primary endpoint is scene/composition-disjoint full task success. Component
metrics cannot substitute for it. Counterfactual ranking accuracy means: among
candidate actions forked from one identical simulator state, the model ranks an
action with a more useful observed prompt-relevant effect ahead of one with a
less useful effect.

## Immediate no-go triggers

- If route+effect does not beat route-only on unseen primitive-target
  compositions, remove the effect head from the claimed method.
- If calibration does not lower false-direct and invalid-information-action
  risk at matched coverage, do not claim reliable decision uncertainty.
- If a prompted Qwen-VL router matches the learned model, diagnose data
  shortcuts and visually inaccessible labels before adding capacity.
- If the text subtask bridge matches projected tokens, retain text.
- If gains are supported only by `OPEN` in one drawer, position the repository
  as a benchmark/calibrated baseline, not an ICRA/RSS method paper.
