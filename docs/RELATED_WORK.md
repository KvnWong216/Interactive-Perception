# Related work, organised by what it contributes to this pipeline

Entries are grouped by the pipeline module they serve, and each says what we
take and what does not transfer. Everything listed is a work the author could
identify with confidence; a short list of claims that could **not** be verified
is kept at the end, deliberately, because an earlier draft of this project's
notes cited three of them with wrong venues and two that may not exist.

---

## 1. Uncertainty representation — the Dirichlet belief

**Jøsang, *Subjective Logic: A Formalism for Reasoning Under Uncertainty*,
Springer 2016.** The opinion/Dirichlet correspondence, and the definitions of
vacuity and dissonance we report. *Take:* the whole representation. *Note:*
vacuity is `W/S` with `W` the non-informative prior weight; writing `K/S` is
correct only under one-pseudo-count-per-category, and we shipped that bug.

**Sensoy, Kaplan and Kandemir, "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018.** Evidence-as-Dirichlet-parameters,
`alpha = e + 1`, `S = sum(alpha)`. *Take:* the evidence accumulation form.
*Does not transfer:* their evidence comes from a trained classifier head; ours
comes from observation and from action attribution, so their calibration
results do not carry over.

**Malinin and Gales, "Predictive Uncertainty Estimation via Prior Networks",
NeurIPS 2018.** Separates distributional from data uncertainty. *Take:* the
argument for why a Dirichlet, rather than a softmax entropy, is the right
object when "I have not looked" must be distinguishable from "it is ambiguous".

**Kendall and Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for
Computer Vision?", NeurIPS 2017.** Aleatoric/epistemic split. *Take:* framing.
*Caution:* our hidden-target uncertainty is neither in their sense — it is
*resolvable by action*, a third category their taxonomy does not name.

## 2. Interactive perception — the reason action enters at all

**Bohg et al., "Interactive Perception: Leveraging Action in Perception and
Perception in Action", IEEE T-RO 2017.** The survey that defines the field.
*Take:* the framing and the vocabulary; cite as the field's anchor.

**Novkovic et al., "Object Finding in Cluttered Scenes Using Interactive
Perception", ICRA 2020.** Volumetric occupancy map, learned push policy to
reveal a hidden target. *Take:* occluded volume computed **from depth**, which
is how we can ground `VIEWPOINT_BLOCKED` without reading simulator state — the
fix for our oracle-leak risk. *Does not transfer:* pushing cannot reveal the
inside of a shut drawer, and their visibility model has no representation for
kinematic occlusion. It covers T06 and none of T01/T02/T03.

**Danielczuk et al., "Mechanical Search: Multi-Step Retrieval of a Target
Object Occluded by Clutter", ICRA 2019.** The closest task formulation to ours.
*Take:* the multi-step retrieval framing and their action set. *Does not
transfer:* the target is specified by image/model, not by language, so "have I
found it" is a matching problem rather than a semantic one.

**Kurenkov et al., "Visuomotor Mechanical Search: Learning to Retrieve Target
Objects in Clutter", IROS 2020.** *Take:* comparison point for learned
search policies.

## 3. Active vision — why viewpoint alone is not enough

**Bajcsy, "Active Perception", Proceedings of the IEEE, 1988** and **Aloimonos,
Weiss and Bandyopadhyay, "Active Vision", IJCV 1988.** The founding statements.
**Connolly, "The Determination of Next Best Views", ICRA 1985.** NBV.
**Bajcsy, Aloimonos and Tsotsos, "Revisiting Active Perception", Autonomous
Robots 2018.** The modern restatement.

*Take:* these are what our `NBV_INSUFFICIENT` certificate is arguing against.
Our benchmark's contribution here is negative and measurable: a hemisphere
sweep showing zero recoverable pixels from 360 poses proves the task is outside
what viewpoint selection can solve. Cite them to establish that the obvious
alternative was tested rather than assumed away.

## 4. Decision making — choosing the primitive

**Kaelbling, Littman and Cassandra, "Planning and acting in partially
observable stochastic domains", *Artificial Intelligence* 101, 1998.** *Take:*
the POMDP formulation and the Bayes-risk comparison our router uses.

**Howard, "Information Value Theory", IEEE Trans. SSC, 1966.** *Take:* the
value-of-information argument for paying a cost to observe. This is the
principled version of "when is it worth opening the drawer".

**Silver and Veness, "Monte-Carlo Planning in Large POMDPs" (POMCP), NeurIPS
2010** and **Somani et al., "DESPOT: Online POMDP Planning with
Regularization", NeurIPS 2013.** *Take:* if one-step Bayes risk proves too
myopic — specifically on multi-container ordering — these are the standard
upgrades. *Note:* we deliberately start myopic; adopt only if measurement
shows the myopic router losing.

## 5. Language-grounded priors — where `omega_k` and `a_k` come from

**Radford et al., "Learning Transferable Visual Models From Natural Language
Supervision" (CLIP), ICML 2021.** *Take:* the text-image similarity that
supplies prompt-to-node relevance. *Caution:* CLIP similarity is a
**co-occurrence** prior, not a containment prior, and it is state-blind — it
cannot know this drawer is shut or already searched. It may only initialise
`omega_k`; the belief must carry the state.

**Zhou et al., "ESC: Exploration with Soft Commonsense Constraints for
Zero-shot Object Navigation", ICML 2023.** *Take:* obtaining commonsense
placement priors from an LLM and using them to direct search. *Does not
transfer:* navigation only — the agent moves, it never changes the world.

**Rana et al., "SayPlan: Grounding Large Language Models using 3D Scene Graphs
for Scalable Robot Task Planning", CoRL 2023.** *Take:* the 3D scene graph with
container nodes; this is the data structure our `LocationHypothesis` set
abstracts. *Does not transfer:* symbolic planning with no uncertainty
quantification.

**Gu et al., "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and
Planning", ICRA 2024.** *Take:* how to build the graph from perception rather
than declare it — the other half of the oracle-leak fix.

**Kant et al., "Housekeep: Tidying Virtual Households using Commonsense
Reasoning", ECCV 2022.** *Take:* evidence that LLM object-placement priors are
accurate enough to be worth conditioning on.

**Ahn et al., "Do As I Can, Not As I Say" (SayCan), CoRL 2022.** *Take:*
affordance-weighted language grounding; the closest precedent for multiplying a
language score by a feasibility term, which is structurally what
`omega_k * [k is OBSERVED]` does.

## 6. Abstention — making `NOT_FOUND` principled

**Ren et al., "Robots That Ask For Help: Uncertainty Alignment for Large
Language Model Planners" (KnowNo), CoRL 2023.** *Take:* conformal prediction to
set the router's threshold with a coverage guarantee instead of by hand. This
is the single most useful import available to us: it removes the "your result
is threshold tuning" objection.

**Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification", 2021**, and **Vovk, Gammerman
and Shafer, *Algorithmic Learning in a Random World*, Springer 2005.** *Take:*
the machinery and the calibration protocol.

### Prompt-induced and semantic uncertainty

**Xiao and Wang, "Quantifying Uncertainties in Natural Language Processing
Tasks", AAAI 2019.** Token/distribution uncertainty is a useful baseline, but
continuous trajectory spread has the same weakness as token entropy: multiple
numerically different outputs may implement one intent.

**Kuhn, Gal and Farquhar, "Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation", 2023.** *Take:* group
samples by meaning before measuring uncertainty. Our corresponding object is a
coarse action intent, not an action coordinate: several chunks that all mean
`REMOVE_OCCLUDER` should contribute one semantic mode.

**Kadavath et al., "Language Models (Mostly) Know What They Know", 2022.**
Self-evaluation can be informative, but its calibration degrades on new tasks.
*Take:* VLM self-probes may be an evidence channel. *Do not transfer:* a stated
confidence is not a calibrated probability and cannot set our stopping rule.

**Schuster et al., "Confident Adaptive Language Modeling", NeurIPS 2022.**
Confidence can control computation. *Does not transfer:* CALM calibrates early
exit in token generation; it does not decide whether a physical action will
reveal missing state.

**Quach et al., "Conformal Language Modeling", 2023**, and **Kumar et al.,
"Conformal Prediction with Large Language Models for Multi-Choice Question
Answering", 2023.** *Take:* set-valued output and held-out calibration rather
than a hand-set threshold. *Difference:* our output space is a fixed semantic
primitive set and our label is the information action required by the scene.

### Closest embodied conformal work

**Ren et al. (KnowNo), CoRL 2023** is the closest language-planning precedent:
execute a singleton conformal action set and ask for help otherwise. Our gap is
different: ambiguity can sometimes be resolved autonomously by changing the
world, so a non-singleton set should route to `REMOVE_OCCLUDER` or viewpoint
acquisition before asking a human.

**Dixit et al., "Adaptive Conformal Prediction for Motion Planning among
Dynamic Agents", L4DC 2023.** Shows how delayed observations can update
conformal uncertainty online under shift. This is relevant if repeated search
steps violate a static exchangeability assumption, but its trajectory safety
sets are not semantic action sets.

**PlanCP, "Conformal Prediction for Uncertainty-Aware Planning", 2023.**
Conformalizes diffusion-planning uncertainty. It is relevant to π0.5's sampled
action distribution, but it operates on plan/dynamics uncertainty rather than
the distinction between `ACT`, information acquisition, and `NOT_FOUND`.

**CoFineLLM, "Conformal Finetuning of LLMs for Language-Instructed Robot
Planning", L4DC 2026.** Reduces conformal set size through finetuning. We keep
π0.5 frozen, so it is a future upper bound rather than a transferable method.

**Lee and Kuo, "Diff-DAgger: Uncertainty Estimation with Diffusion Policy for
Robotic Manipulation", 2024.** This is the verified diffusion-policy work most
directly relevant to our action sampling. It shows why ensemble/action variance
confuses legitimate multimodality with OOD failure, and instead scores a
generated action using the diffusion training loss. *Take:* semantic modes and
executor OOD must be separated. *Current limitation:* the π0.5 websocket API
returns actions but not its flow-matching training loss or intermediate vector
field, so this score cannot be reproduced without a versioned server change.

**Li et al., "Spatial Memory for Out-of-Vision Manipulation in
Vision-Language-Action" (SOMA), ICML 2026.** SOMA constructs persistent memory
from multi-view head-camera scans and retrieves instruction-relevant spatial
cues. It is a strong baseline for `VIEWPOINT_BLOCKED`. It does not, from the
paper's stated method, establish the circulated Bayesian containment model, and
a view scan cannot reveal the interior of a closed drawer. Our
`NBV_INSUFFICIENT` condition is the boundary where SOMA-style viewing stops and
physical information acquisition begins.

**Karli, Kurumisawa and Fitzgerald, "Ask Before You Act: Token-Level
Uncertainty for Intervention in Vision-Language-Action Models", RSS OOD
Robotics Workshop 2025.** Uses entropy/perplexity from a fine-tuned π0-FAST to
request human correction. It is a direct VLA uncertainty baseline, but depends
on discrete action-token probabilities and responds with human intervention;
π0.5 exposes neither those logits nor that fallback.

**Yang et al., "UAOR: Uncertainty-aware Observation Reinjection for
Vision-Language-Action Models", 2026.** Uses layer-wise action entropy to
reinject visual features. *Take:* uncertainty can change perception rather than
only stop execution. *Difference:* it requires white-box hidden layers, retains
task/architecture-calibrated thresholds, and re-reads the same image rather
than physically revealing an occluded state.

**Huang et al., "VLAConf: Calibrated Task-Success Confidence for
Vision-Language-Action Models", 2026.** Learns a one-class confidence head on
frozen VLA internal representations and post-hoc calibrates task-success
confidence. This is the closest executor-competence baseline to our capability
gate. It is more sample-efficient than repeated action draws but requires
white-box representations and training; our exact binomial capability gate is
black-box and primitive-specific.

**Valle et al., "Evaluating Uncertainty and Quality of
Vision-Language-Action-enabled Robots", 2025.** Evaluates action and trajectory
quality metrics against human labels. *Take:* report execution quality beside
binary success. *Caution:* correlation with human ratings is not calibrated
semantic intent coverage.

The paper's defensible novelty is therefore not "the first conformal robot" or
"the first uncertain VLA". Subject to a final systematic literature review, it
is the combination of (i) semantic uncertainty over sampled VLA action chunks,
(ii) a distinction between viewpoint-resolvable and manipulation-resolvable
uncertainty, (iii) a separate executor-reliability gate, and (iv) autonomous
physical information acquisition as the response to a non-singleton action
set.

## 7. Base policy and benchmark

**Black et al., "π0: A Vision-Language-Action Flow Model for General Robot
Control", 2024**, and **π0.5 (Physical Intelligence, 2025)**. The frozen
executor. *Take:* flow-matching sampling gives independent action draws per
`infer` call, which is what makes the action-spread measurement possible at
all.

**Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", CoRL
2024**, **Octo Model Team, "Octo: An Open-Source Generalist Robot Policy", RSS
2024**, **Brohan et al., RT-1 (RSS 2023) and RT-2 (CoRL 2023).** *Take:*
alternative frozen executors for a generality claim — the argument that the
finding is about monolithic VLAs rather than about π0.5 needs at least one
second policy.

**Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
Learning", NeurIPS 2023 Datasets & Benchmarks.** The simulator and scene
assets. *Note:* stock LIBERO contains **no fridge task at all**, so T02 is
off-distribution by construction.

**Anderson et al., "On Evaluation of Embodied Navigation Agents", 2018.**
*Take:* SPL, the template for success weighted by efficiency, which is the
model for the time-normalised success metric this benchmark still lacks.

---

## Claims that could not be verified

An earlier synthesis circulated in this project cited the following. Three had
demonstrably wrong venues or titles; two could not be located at all. None
should enter a bibliography without a direct check.

| Claim as circulated | Status |
|---|---|
| ESC, "Exploration with **Spatial** Commonsense", **CVPR 2023** | Wrong on both: it is **Soft** Commonsense Constraints, **ICML 2023** |
| KnowNo, "CoRL/**NeurIPS**" | **CoRL 2023** |
| HOV-SG, "...for **Active Object Search**" | Real work, but subtitled for **language-grounded navigation** |
| "Map Space Belief Prediction for Manipulation-Enhanced Mapping", RSS 2025 | Plausible, **unverified** |
| "Language-Conditioned Conformal Prediction for Embodied Search", 2024/25 | **Could not be located** |
| "SOMA (Spatial Memory for VLA)" | Now verified as Li et al., ICML 2026; earlier 2024/25 dating was wrong |
| "KnowNo-3D Series" | No distinct paper/series located; cite KnowNo only unless a primary source is supplied |
| "Conformalized VLA Trajectory Sets for Safe Manipulation", 2024 | Could not be located under this title |
| "Quantifying Uncertainty in Diffusion Policies for Robotic Manipulation", RA-L 2024 | Could not be located under this title; Diff-DAgger is the verified nearby work |
