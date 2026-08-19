# T01 mechanism results

| Test | Input | Prompt | Action | Endpoint | Online | Oracle | Result | LB95 | Split | Sealed |
|---|---|---|---|---|---|---|---|---|---|---|
| Stock pi05 reproduction | stock RGB/state | ten stock LIBERO object prompts | ACT | final task success | yes | no | 100/100 | 0.970 | reproduction | no |
| T01 drawer motor primitive | stock RGB/state | Open the middle layer of the drawer | OPEN_CONTAINER | drawer joint opened | yes | no | 97/100 | 0.924 | capability development | no |
| Expected-risk initial routing | prompt-conditioned frozen-prefix belief | butter/cream-cheese counterfactuals | ACT vs OPEN_TO_INSPECT | target-observability route | no | no | 296/300 | 0.970 | offline development | no |
| v12b clean RGB outcome singleton — FAILED | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 40/40 | 0.928 | fresh clean development 1900-1939 | no |
| v12b clean RGB outcome singleton — REVEALED | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 40/40 | 0.928 | fresh clean development 1900-1939 | no |
| v12b clean RGB outcome singleton — EMPTY | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 39/40 | 0.887 | fresh clean development 1900-1939 | no |
| Clean physical information endpoint — REVEALED | stock RGB/state | Open the middle layer of the drawer | OPEN_AND_OBSERVE | prompt-resolvable target at any public history point | yes | no | 40/40 | 0.928 | fresh clean development 1900-1939 | no |
| Clean physical information endpoint — EMPTY | stock RGB/state | Open the middle layer of the drawer | OPEN_AND_OBSERVE | local middle-layer searched-region coverage with no target evidence | yes | no | 40/40 | 0.928 | fresh clean development 1900-1939 | no |
| v12b sealed RGB outcome singleton — FAILED | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 100/100 | 0.970 | one-time sealed audit 900-999 | yes |
| v12b sealed RGB outcome singleton — REVEALED | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 100/100 | 0.970 | one-time sealed audit 900-999 | yes |
| v12b sealed RGB outcome singleton — EMPTY | six agentview+wrist RGB frames | Find the butter | OPEN_AND_OBSERVE | public-RGB observable outcome | no | no | 94/100 | 0.885 | one-time sealed audit 900-999 | yes |
| Sealed physical information endpoint — REVEALED | stock RGB/state | Open the middle layer of the drawer | OPEN_AND_OBSERVE | prompt-resolvable target at any public history point | yes | no | 100/100 | 0.970 | one-time sealed audit 900-999 | yes |
| Sealed physical information endpoint — EMPTY | stock RGB/state | Open the middle layer of the drawer | OPEN_AND_OBSERVE | local middle-layer searched-region coverage with no target evidence | yes | no | 100/100 | 0.970 | one-time sealed audit 900-999 | yes |
| Oracle-free five-case information loop smoke | prompt + stock RGB/state/history | butter/cream-cheese counterfactuals | ACT or OPEN_AND_OBSERVE | correct route/outcome; DIRECT_ACT semantic handoff | yes | no | 5/5 | 0.549 | disposable seed1399 smoke | no |
| Original-prompt physical information acquisition | prompt + stock RGB/state/history | Place the butter in the basket | OPEN_AND_OBSERVE then replan | prompt-relevant information acquired | yes | no | 1/1 | 0.050 | disposable seed1399 physical diagnostic | no |
| Original-prompt physical final continuation | stock RGB/state after public-RGB REVEALED | Place the butter in the basket | fixed 400-step ACT | final task success | yes | no | 0/1 | 0.000 | disposable seed1399 physical diagnostic | no |
| Post-reveal final retrieval | stock RGB/state | Place the butter in the basket | ACT after drawer opening | final task success | yes | yes | 0/5 | 0.000 | development diagnostic | no |

Primary endpoint: target observability. Final task success is never substituted for it or inferred from it.
