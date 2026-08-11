#### Prompt ladder, pooled over scenes

| Prompt rung | Episodes | Task success | Information endpoint | Correct terminal | Premature commit | False NOT_FOUND |
|---|---|---|---|---|---|---|
| `implicit` | 15 | 20% [0, 40] | 33% [13, 60] | 20% | 60% | 0% |
| `hinted` | 15 | 0% [0, 0] | 33% [13, 60] | 0% | 53% | 0% |
| `explicit` | 15 | 13% [0, 33] | 47% [20, 73] | 13% | 40% | 0% |
| `capability` | 10 | n/a | 50% [20, 80] | n/a | 40% | n/a |

#### Decision failure versus skill gap

| Contrast | Paired episodes | Mean difference in endpoint rate | 95% CI |
|---|---|---|---|
| `capability` - `implicit` | 10 | +0.000 | [+0.000, +0.000] |
| `explicit` - `implicit` | 15 | +0.133 | [+0.000, +0.333] |

**Information endpoint rate, per scene**

| Task | `implicit` | `hinted` | `explicit` | `capability` | n |
|---|---|---|---|---|---|
| T01_drawer_retrieval | 0% | 0% | 0% | 0% | 5 |
| T04_visible_direct | 100% | 100% | 100% | 100% | 5 |
| T06_dense_clutter_partial_occlusion | 0% | 0% | 40% | n/a | 5 |

**Task success rate, per scene**

| Task | `implicit` | `hinted` | `explicit` | `capability` | n |
|---|---|---|---|---|---|
| T01_drawer_retrieval | 0% | 0% | 0% | 0% | 5 |
| T04_visible_direct | 60% | 0% | 40% | 40% | 5 |
| T06_dense_clutter_partial_occlusion | 0% | 0% | 0% | n/a | 5 |

#### Are the uncertainty readings interpretable?

| Prompt rung | Mean vacuity | Mean dissonance | Saturated fraction | Uninformative episodes | Errors |
|---|---|---|---|---|---|
| `implicit` | 0.459 | 0.249 | 0.24 | 0/15 | 0 |
| `hinted` | 0.467 | 0.250 | 0.20 | 0/15 | 0 |
| `explicit` | 0.481 | 0.238 | 0.19 | 0/15 | 0 |
| `capability` | 0.350 | 0.333 | 0.22 | 0/10 | 0 |

