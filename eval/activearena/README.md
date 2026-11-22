# ActiveArena Evaluation Integration (`eval/activearena`)

This directory integrates **ActiveArena** (simulator & 35-task ID/OOD benchmark) and **ActiveArena-VLA** (frozen 18-DoF `oft_subtask_action_12_ws` / `oft_instruction_action_12_ws` executors) directly into IP-VLA's evaluation suite with **zero upstream simulator patches**.

## How Zero-Oracle Integration Works

ActiveArena's `script/eval_policy.py` dynamically imports `--policy_name` via `importlib.import_module`. When `--mode ipvla_planner` is selected, `eval/activearena/run_eval.py` launches:

1. **Frozen Action Server** (`ActiveArena-VLA/deployment/model_server/server_policy.py` on port `19000`) serving `oft_subtask_action_12_ws`.
2. **IP-VLA Planner Server** (`ipvla/stage2/planner_server.py` on port `19100`) serving `Qwen3-VL-2B-Instruct` + merged Stage-1/Stage-2 LoRA + calibrated `RidgeTempConfidenceHead`.
3. **Unmodified ActiveArena Simulator** (`ActiveArena/script/eval_policy.py`) importing `eval.activearena.autonomous_policy`, which subclasses `StarVLAOFTClient` and replaces ONLY `_subtask_instruction(...)` (privileged simulator oracle) with `TwoStageSwitchingController`.

Every episode writes a structured per-step switching trace (`ipvla_switching_trace.jsonl`) recording `phi`, `phi_kin`, `phi_grip`, `phi_dw`, `novelty_r`, `wake`, `commit`, `pinned_steps`, `total_vlm_calls`, and `total_commits`.
