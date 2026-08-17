# T01 temporal-label audit

The v7 outcome artifact is invalidated. During the first 700--799 audit pass,
seed 770 revealed the butter at the 50% and 75% opening history points, then
lost it from both final policy views after return. A final-frame-only evaluator
would incorrectly label this trajectory `EMPTY` although the robot had already
acquired the requested evidence.

Evidence:

- `outputs/t01_open_and_observe_effect_v1_audit/images/revealed_full/seed770/history_02_agentview.png`
- `outputs/t01_open_and_observe_effect_v1_audit/images/revealed_full/seed770/history_03_agentview.png`
- `outputs/t01_open_and_observe_effect_v1_audit/images/revealed_full/seed770/after_agentview.png`
- `outputs/t01_open_and_observe_effect_v1_audit/images/revealed_full/seed770/after_wrist.png`

The run was stopped after 77/300 rows. Those rows are debug-only; seeds
700--799 can never be reused as a sealed audit. No v7 audit result was produced.

Version 3 labels the complete six-point public history. Any target visibility
produces `REVEALED`. `EMPTY` additionally requires the region to remain open,
return-to-observe to complete, and same-camera-pose counterfactual visibility
of the seed-matched target trajectory. The replacement model is v8 and its
untouched audit seeds are 900--999.
