# Calibrated interaction package

| responsibility | input | output / shape | train/freeze | lineage | privileged state | tests | ablation |
|---|---|---|---|---|---|---|---|
| candidate contracts | VLM JSON candidate | typed `CandidateAction` | frozen schema | SayCan, CaP | never | `test_calibrated_interaction.py` | fixed vs VLM-grounded candidates |
| capability adapter | candidate + registry + task prompt | short π0.5 instruction | frozen | SayCan, π0.5 | never | registry/serializer test | text vs projected token |
| VLM context encoding | prompt + RGB history + public history | `H_t [B,T,D_vlm]`, candidate embeddings `[B,A,D_vlm]` | frozen Stage 1 | π0.5, CogACT | never | feature-manifest test required before training | no prompt; last frame only; RGB-D |
| interaction decoder | `H_t`, candidate embeddings, masks | `c_tj [B,A,D]` | train | candidate queries; componentized VLA | never | `test_candidate_interaction_model.py` | route-only; no effect supervision |
| effect head | `c_tj` | Bernoulli logits `[B,A,6]` | train | CNABU/VLMPC-inspired, modified to task facts | never online | model/data tests | softmax ontology; EDL; no effect head |
| route head | `c_tj` + effect logits | route logits `[B,A]` | train | KnowNo action options, learned here | never | model/selector tests | route-only; verbal confidence |
| calibration | held-out logits/labels | temperature + route/effect sets | frozen after fit | KnowNo/split conformal | labels only offline | calibration/selector tests | no calibration; raw entropy |
| controller | candidates, calibrated sets, public history | action/abstain + trace | frozen rule | Inner Monologue/ReAct | never | replay integration | KnowNo-only; ReAct router |
| data firewall | counterfactual JSON | policy input and separate supervision | frozen | reproducibility requirement | only evaluator partition | leakage/split tests | not applicable |

The decoder has one cross-attention block and two output heads. It has no
global confidence, task confidence, resolution uncertainty, sufficiency head,
belief token, manual effect library, or second selector.

Install the small learned head only when needed:

```bash
uv sync --extra learned --extra dev
```

VLM dependencies are separate because Stage 0 control and calibration tests do
not require loading a multi-billion-parameter encoder:

```bash
uv sync --extra vlm
```
