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
| "SOMA (Spatial Memory for VLA)" | **Could not be located** |
