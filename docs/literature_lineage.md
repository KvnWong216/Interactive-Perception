# Literature lineage and status audit

Audit cutoff: **2026-08-22**. Sources are papers, proceedings, OpenReview,
official project pages, or official repositories. A 2026 arXiv upload is marked
as a preprint unless a proceedings page exists; submission or withdrawal does
not count as peer review. “Calibration” below means calibration against held-out
outcomes, not a model's verbal confidence.

## A. Language/VLM planning and capability grounding

| paper | venue/status/year | official URL | problem | observation | uncertainty representation | action space | uncertainty-to-action mechanism | VLM/VLA integration | supervision | evaluation | code availability | borrowed component | limitation relative to our problem | novelty threat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SayCan | CoRL, peer-reviewed, 2022 | [project/paper](https://say-can.github.io/) | language planning grounded in robot affordances | language + skill values | implicit LM and affordance scores | fixed learned skill bank | product/ranking of semantic and feasibility scores | LLM planner over named skills | pretrained LM + skill RL/data | mobile manipulation | official simulated release | capability-constrained candidates | no calibrated prompt-relevant physical information effects | medium |
| Code as Policies | ICRA, peer-reviewed, 2023 | [project/code](https://code-as-policies.github.io/) | synthesize embodied programs from language | text + perception APIs | none explicit | exposed code/API functions | generated program calls grounded APIs | LLM composes robot APIs | few-shot prompting | several real robots | official code/Colab | schema/API constrained generation | no calibrated route/effect model; generated code expands attack surface | low |
| Inner Monologue | CoRL 2022, peer-reviewed (proceedings 2023) | [PMLR paper](https://proceedings.mlr.press/v205/huang23c.html) | close LLM planning loop with feedback | language scene/success/human feedback | qualitative feedback | pretrained manipulation skills | feedback appended to next planning prompt | frozen LLM planner + skill library | none beyond component pretraining | simulated/real tabletop and mobile manipulation | project materials | typed action-observation history | no decision calibration or candidate counterfactual effects | medium |
| ReAct | ICLR, peer-reviewed, 2023 | [paper](https://arxiv.org/abs/2210.03629) | interleave reasoning and external actions | text observations | none explicit | environment/tool actions | observations update later reasoning | prompted LLM agent | in-context examples | QA, ALFWorld, WebShop | official repository linked by paper | ReAct-style baseline/history format | not continuous robot manipulation and not calibrated | low |
| VoxPoser | CoRL oral, peer-reviewed, 2023 | [project/code](https://voxposer.github.io/) | open-set language-conditioned 3-D manipulation | RGB-D + language | no calibrated decision uncertainty | dense 6-DoF trajectories through value maps | model-based replanning after visual feedback | LLM code + VLM grounding + motion planner | zero-shot; optional learned dynamics | sim and real manipulation | official code | open-vocabulary grounding separated from control | needs RGB-D/value-map stack; no candidate effect calibration | medium |
| Semantic Mechanical Search | ICRA, peer-reviewed, 2023 | [paper/project](https://arxiv.org/abs/2302.12915) | find fully occluded objects using semantic priors | shelf/bin image + target | semantic occupancy distribution | downstream search/manipulation policy | semantic distribution biases search | LLM/VLM semantic plug-in to geometric planner | prompted foundation models | sim and physical pharmacy/kitchen/office | official code/data linked | prompt-relevant search prior | target search, not general task routing or calibrated VLA bridge | high |

Capability-set constraint is therefore not an implementation detail: SayCan
grounds language by executable skill values, CaP is bounded by exposed APIs,
and VoxPoser grounds language into geometry. The VLM may propose open-vocabulary
targets and relations, but legality, reachability, and execution success must be
grounded by the capability registry, executor data, or counterfactual simulation.
The closed loop should record typed `(candidate, public outcome)` events, as in
Inner Monologue/ReAct, without placing free-form hidden reasoning in the control
contract.

## B. VLA semantic/action interfaces

| paper | venue/status/year | official URL | problem | observation | uncertainty representation | action space | uncertainty-to-action mechanism | VLM/VLA integration | supervision | evaluation | code availability | borrowed component | limitation relative to our problem | novelty threat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| π0 | arXiv technical report, 2024 | [paper](https://www.physicalintelligence.company/download/pi0.pdf) | generalist continuous robot control | image, language, state | none as decision contract | continuous action chunks via flow matching | not applicable | PaliGemma prefix and separate action expert share transformer attention; suffix queries prefix KV | large robot corpus | multiple real embodiments | [openpi](https://github.com/Physical-Intelligence/openpi) | frozen low-level executor | no calibrated information-action router | low |
| π0.5 | CoRL, peer-reviewed, 2025 | [PMLR](https://proceedings.mlr.press/v305/black25a.html) | open-world generalization and semantic subtasks | images, language, detections, state | none calibrated for route selection | high-level semantic outputs + continuous actions | learned high-level inference, not conformal arbitration | π0 backbone; heterogeneous co-training and knowledge insulation | robot + web + semantic data | long-horizon real homes | [openpi](https://github.com/Physical-Intelligence/openpi) | short text subtask to frozen executor | does not isolate calibrated prompt-relevant physical information selection | medium |
| CogACT | arXiv preprint/project release, 2024 | [paper/project](https://arxiv.org/abs/2411.19650) | improve VLA action modeling | vision + language + state | none explicit | diffusion action transformer | not applicable | componentized VLM plus specialized action module | robot demonstrations | five embodiments, sim + real | project code/models | separate cognition/action roles | no information-effect prediction/calibration | low |
| CoMe-VLA / Act, Sense, Act | arXiv preprint, 2026 | [paper](https://arxiv.org/abs/2602.04600) | learn non-Markovian active perception | temporal egocentric RGB + proprioception | information gain/decision branching, not reported conformal sets | exploration and manipulation actions | cognition head transitions subtasks using temporal memory | VLA with cognitive auxiliary head and dual-track memory | large human egocentric data then robot alignment | wheel-based humanoid long-horizon tasks | not established in audited source | temporal history/cognition baseline | broad active perception and camera behavior; no calibrated candidate effect comparison | high |
| UAOR | **withdrawn ICLR 2026 submission; arXiv preprint**, 2026 | [OpenReview](https://openreview.net/forum?id=3azIn8ImwP) | reduce observation forgetting in VLA layers | current visual observation | intermediate action entropy | original VLA actions | high entropy triggers observation reinjection | training-free FFN attention retrieval | none | simulation + real tasks claimed | project page in paper | distinguish action entropy from route uncertainty | thresholded internal entropy, no physical information action or coverage | medium |
| SCALE | arXiv preprint, 2026 | [paper](https://arxiv.org/abs/2602.04208) | single-pass uncertainty-conditioned perception/action scaling | current VLA inputs | self-uncertainty/action entropy | original VLA action | uncertainty modulates visual and action exploration | inference-time VLA intervention | none | simulation + real claimed | not established | uncertainty baseline | no candidate effects, conformal sets, or physical evidence task | medium |
| Coarse-to-Control | arXiv preprint, 2026 | [paper](https://arxiv.org/abs/2606.07107) | long-horizon action-token planning | VLA observation | none explicit | coarse and executable discrete action tokens | plan tokens condition control tokens | shared action-token vocabulary | robot trajectories | LIBERO, SimplerEnv, real robot | not established | alternative token bridge | planning tokens are not calibrated decision uncertainty | low |
| ZR-0 | arXiv preprint with code, 2026 | [paper](https://arxiv.org/abs/2606.30552) | cross-embodiment dense embodied reasoning | image, language, state/history | none explicit | DiT continuous chunks | not applicable | Qwen3-VL system-2 and action expert coupled by cross-attention | 60M-frame dense ECoT corpus | LIBERO, RoboTwin, RoboCasa, real xArm | [official GitHub](https://github.com/RUCKBReasoning/ZR-0) | evidence for cross-attention alternative | far beyond one-GPU stage; no calibrated information routing | low |
| SaPaVe | CVPR, peer-reviewed, 2026 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SaPaVe_Towards_Active_Perception_and_Manipulation_in_Vision-Language_Action_Models_CVPR_2026_paper.pdf) | jointly learn camera control and manipulation | active head RGB-D/view pose | no conformal decision set | decoupled continuous camera + manipulation actions | semantic camera control selects active views | end-to-end VLA with geometry-aware active-view module | ActiveViewPose-200K + robot data | ActiveManip-Bench + real robot | [project](https://lmzpai.github.io/SaPaVe/) | strong active-perception baseline | primarily active viewpoint change, explicitly outside our scope | high but orthogonalized |
| ActiveVLA | CVPR, peer-reviewed, 2026 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_ActiveVLA_Injecting_Active_Perception_into_Vision-Language-Action_Models_for_Precise_3D_CVPR_2026_paper.pdf) | active views/zoom for precise 3-D manipulation | multi-view RGB-D/3-D | amodal relevance, diversity, occlusion objectives | viewpoint/zoom + manipulation | optimize viewpoint around critical 3-D region | BridgeVLA/PaliGemma backbone with 3-D active perception | robot demonstrations | RLBench, COLOSSEUM, GemBench, real robot | not established | scope-separation baseline | camera/NBV method, eight H100s reported; excluded from our method | high but out of scope |

In official openpi, image and language tokens form a bidirectional prefix;
π0.5 encodes state in discrete prefix tokens, while noisy actions and flow time
form an action-expert suffix whose queries attend to the cached prefix. Thus a
short text subtask preserves the released inference contract. A projected soft
token or new cross-attention path changes that distribution and belongs only in
a later ablation. Query/cognition/planning tokens are internal computation;
action tokens or flow suffixes parameterize control. Naming a token
“uncertainty” does not make it calibrated or decision-relevant.

## C. Uncertainty and calibration

| paper | venue/status/year | official URL | problem | observation | uncertainty representation | action space | uncertainty-to-action mechanism | VLM/VLA integration | supervision | evaluation | code availability | borrowed component | limitation relative to our problem | novelty threat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KnowNo | CoRL, peer-reviewed, 2023 | [PMLR paper](https://proceedings.mlr.press/v229/ren23a.html) | align LLM planner uncertainty and help seeking | instruction + generated action options | conformal action prediction set | discrete language plans/skills | singleton execute; multi-set ask for help | wrapper around pretrained LLM planner | held-out calibration examples | multiple simulated/real embodiments | official project/code linked | split-conformal route set | asks humans; no physical action-effect prediction | high |
| Semantic Entropy | Nature, peer-reviewed, 2024 | [article](https://www.nature.com/articles/s41586-024-07421-0) | detect meaning-level LLM confabulation | multiple text generations | entropy after semantic-equivalence clustering | none | diagnostics/selective prediction | black/white-box LLM sampling | no task training | QA/hallucination datasets | code/data linked | semantic-entropy baseline | expensive text sampling; not a robot decision distribution | low |
| Evidential Deep Learning | NeurIPS, peer-reviewed, 2018 | [proceedings](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) | classification uncertainty | task features | Dirichlet evidence/subjective opinion | none | downstream abstention/OOD use | neural classification head | supervised classes + evidential loss | image classification/OOD | paper supplement | optional EDL ablation | fixed mutually exclusive ontology and loss assumptions required; not coverage guarantee | medium |
| Ask Before You Act | RSS OOD workshop, **workshop paper**, 2025 | [workshop listing](https://sites.google.com/stanford.edu/ood-workshop-rss-25/accepted-contributions) | trigger human VLA intervention | π0-FAST token stream | token entropy/perplexity | continuous shared control or one-step human action | threshold triggers pause/help/blending | introspection on fine-tuned autoregressive VLA | robot demonstrations | manipulation tasks/data regimes | not established | token-entropy baseline | workshop evidence; unreliable in low-data regime; not calibrated action routing | medium |
| CoFineLLM | L4DC, peer-reviewed, 2026 | [PMLR](https://proceedings.mlr.press/v331/wang26c.html) | train LLM planners for smaller conformal sets | language plans | conformal action set | discrete plans | singleton execute, otherwise help | uncertainty-aware LLM fine-tuning | plan labels + conformal objective | language robot planning | not established | set-size/coverage metrics | no physical effect or continuous VLA loop | medium |

Verbal confidence is generated text, not a proper scoring-rule estimate.
Raw free-text entropy changes with synonyms, tokenization, and answer length;
semantic entropy partly removes that invariance but still does not measure the
robot's finite candidate decision. Decision uncertainty is the calibrated
distribution/set over executable candidates. A conformal set offers marginal
coverage under exchangeability; it is not a per-example probability of success.
Use EDL only if the effect ontology is fixed and mutually exclusive and it wins
held-out NLL, Brier, ECE, coverage, and OOD tests. Here the effect facts can
co-occur, so calibrated Bernoulli factors are better justified.

Metric meanings: coverage measures how often the truth lies in the set; set
size measures residual ambiguity/efficiency; ECE bins confidence versus empirical
accuracy; Brier is squared probabilistic error; NLL is a proper log score and
strongly penalizes confident errors.

## D. Physical information acquisition

| paper | venue/status/year | official URL | problem | observation | uncertainty representation | action space | uncertainty-to-action mechanism | VLM/VLA integration | supervision | evaluation | code availability | borrowed component | limitation relative to our problem | novelty threat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mechanical Search | ICRA, peer-reviewed, 2019 | [project/paper](https://ai.stanford.edu/mech-search/multistep/) | retrieve known target under clutter | RGB-D | detector/action confidence | push, suction, grasp | iterative hand-designed/learned search policies | no VLM/VLA | simulation/robot trials | 15k sim + 300 physical | project assets | mechanical-search baseline | fixed known target/bin; no prompt calibration | high |
| VLMPC | RSS, peer-reviewed, 2024 | [RSS paper](https://www.roboticsproceedings.org/rss20/p106.pdf) | forward-looking manipulation control | current/goal image or language | hierarchical pixel/knowledge cost | sampled action sequences | action-conditioned video predicts futures, VLM ranks | VLM sampler/cost + lightweight video predictor | action-video data | public benchmarks + real tasks | [official GitHub](https://github.com/PPjmchen/VLMPC) | counterfactual future baseline | predicts full pixels, not calibrated task-relevant evidence; heavier | high |
| CNABU / Map Space Belief Prediction | arXiv preprint, 2025 | [paper](https://arxiv.org/abs/2502.20606) | manipulation-enhanced semantic mapping | RGB-D metric-semantic grid | calibrated evidential map belief | view then push | candidate-conditioned belief prediction + information gain planning | no VLM/VLA | simulated map transitions | sim + real cluttered shelves | not established in audited source | direct prior for counterfactual effect learning | global map completion, privileged geometry, view actions, no prompt task routing | very high |
| Dengler et al. | Humanoids, peer-reviewed, 2025 | [official publication](https://www.hrl.uni-bonn.de/publications/2025/dengler25humanoids) | efficient uncertainty-aware mapping/manipulation | monocular/RGB-D semantic map | Dirichlet/Beta map uncertainty | NBV + targeted push | RL NBV and uncertainty-informed push reduce map uncertainty | no VLM/VLA | learned mapping/RL | occlusion-heavy shelves, real applicability | official links page | physical uncertainty-action comparator | optimizes global map; includes NBV; not prompt-conditioned task completion | very high |
| ZS-IP | arXiv preprint, 2026 | [paper](https://arxiv.org/abs/2602.18374) | zero-shot semantic-query interactive perception | RGB with keypoints/pushlines + memory | VLM reasoning, no calibrated distribution | push, pull, grasp | memory-guided VLM directly chooses controller action | VLM + visual prompts + robot controller | zero-shot/prompted | real Franka occlusion scenes | not established | strongest VLM-router baseline | no held-out calibration or learned candidate effects | very high |
| LIBERO-Occ | arXiv preprint with code, 2026 | [paper](https://arxiv.org/abs/2606.10862) | evaluate VLA scene-induced occlusion | primary RGB + imagined complementary view | occlusion severity, no decision calibration | original VLA action | viewpoint imagination completes perception without physical interaction | generator conditions VLA | occlusion benchmark + generation | LIBERO task suites | [official GitHub](https://github.com/litsh/Libero-Occ) | occlusion benchmark/passive completion baseline | imagined view; no physical information action | high |
| PROBE | arXiv preprint, 2026-08-17 | [paper](https://arxiv.org/abs/2608.17129) | manipulation-grounded visual QA | cluttered scene + question + tool observations | no conformal decision set | pick/push then answer | VLM agent manipulates until it can answer | frontier VLM tools; distilled smaller agent | successful teacher trajectories/mixed recipe | 150 tasks, six question types, sim-to-real | release not established at cutoff | closest benchmark/data threat | ends in an answer, not continuous VLA task completion; no calibrated candidate effects | very high |
| CoMe-VLA | arXiv preprint, 2026 | [paper](https://arxiv.org/abs/2602.04600) | learned active sensing/manipulation transitions | temporal egocentric history | learned cognition transition | broad sense/act behaviors | auxiliary cognition head switches subtasks | integrated VLA | human egocentric + robot data | humanoid long horizon | not established | temporal active-perception baseline | not candidate-effect calibrated; camera actions included | high |

## Audited conclusion

Action-conditioned belief/effect is explicit in CNABU and VLMPC; Dengler et al.
predict/use uncertainty change for mapping. ZS-IP and PROBE let a VLM agent choose
physical exploration without calibrated candidate-effect distributions.
LIBERO-Occ imagines evidence rather than acting. PROBE terminates in VQA, not a
continuous VLA task. The smallest defensible claim is therefore:

> Prompt-conditioned, calibrated selection of physical information actions
> through candidate-conditioned task-relevant effect prediction, integrated
> with a frozen continuous-action VLA executor.

This remains a hypothesis, not a novelty fact, until experiments beat a prompted
VLM router, route-only, KnowNo-only, and counterfactual/oracle comparisons on
scene-disjoint splits.
