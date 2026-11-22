# IP-VLA: Evidence-Gated Active Perception & Continuous Cerebellum Switching

A clean, plug-and-play research codebase for **evidence-gated active perception subtask planning** (Stage-1 open-video training & calibrated confidence readout) and **closed-loop embodied evaluation on ActiveArena** (Stage-2 continuous cerebellum inertia, non-oracle `inspect`-keyframe visual memory, and two-stage confidence-gated switching).

---

## Repository Layout

```text
ip-vla/
├── configs/
│   ├── paths.example.yaml            # Portable path template (copy to paths.local.yaml or override via env)
│   ├── stage1_v2_rebalanced.json     # Stage-1 v2 4:2:4:4:2 open-video training configuration
│   └── stage2_activearena.yaml       # Stage-2 cerebellum inertia, memory & switching hyperparameters
├── ipvla/
│   ├── config.py                     # Unified env-var + YAML path loader
│   ├── stage1/
│   │   ├── data_builder.py           # 100% English open-video causal suite builder (0 ActiveArena rows)
│   │   ├── train_vlm.py              # Qwen3-VL-2B LoRA r=32 + Softplus Ridge-Temp confidence training
│   │   └── confidence.py             # Standalone calibrated confidence head inference module
│   ├── stage2/
│   │   ├── cerebellum.py             # Continuous cerebellum inertia: phi_t = max(phi_kin, phi_grip, phi_dw)
│   │   ├── memory.py                 # Non-oracle inspect-keyframe memory buffer (pins reveal frames)
│   │   ├── switching.py              # Two-stage wake (novelty r_t + T_lock) & relative confidence commit gate
│   │   └── planner_server.py         # Localhost HTTP server for Qwen3-VL-2B (+ LoRA + confidence head)
│   └── data_pipeline/                # Open-video Teacher A/B causal annotation queue (zero-leak cleanup)
├── eval/
│   ├── open_video/                   # Stage-1 2x2 Jump-Cut vs. Evidence-Removal & Experiment E probes
│   └── activearena/                  # ActiveArena integrated as first-class closed-loop eval suite
│       ├── setup_activearena.sh      # One-command install of ActiveArena + ActiveArena-VLA + assets + weights
│       ├── autonomous_policy.py      # Drop-in zero-oracle policy adapter for unmodified ActiveArena eval
│       └── run_eval.py               # Unified CLI for baselines, IP-VLA planner, and ablations
├── scripts/
│   └── eval_stage2_activearena.sh    # One-command Stage-2 ActiveArena evaluation launcher
├── docs/
│   └── METHOD_SPEC.md                # Unified Stage-1 & Stage-2 mathematical specification
└── tests/
    └── test_cerebellum_and_switching.py
```

---

## Quickstart

### 1. Install `ipvla`

```bash
pip install -e ".[dev]"
pytest -q
```

Configure local checkpoint/environment paths by copying `configs/paths.example.yaml` to `configs/paths.local.yaml` (git-ignored) or setting environment variables (`IPVLA_BASE_VLM`, `IPVLA_STAGE1_LORA`, `IPVLA_STAGE1_CONF_HEAD`, `ACTIVEARENA_ROOT`, etc.).

### 2. Setup ActiveArena Evaluation Suite

ActiveArena and ActiveArena-VLA are integrated cleanly under `eval/activearena/` without patching upstream simulator code:

```bash
bash eval/activearena/setup_activearena.sh
```

Verify readiness (0 GPU usage):

```bash
bash scripts/eval_stage2_activearena.sh \
  --mode oracle_subtask_oft \
  --tasks check_block_color check_cola_date \
  --episodes 5 \
  --dry-run
```

### 3. Run Closed-Loop ActiveArena Evaluation (Paired Seeds)

All modes evaluate on the exact same official ActiveArena seed lists (`id` = `info_gathering_demo`, `ood` = `info_gathering_randomized`):

```bash
# Tier-1A: Official End-to-End Instruction Baseline (oft_instruction_action_12_ws)
bash scripts/eval_stage2_activearena.sh \
  --mode instruction_oft --setting ood \
  --tasks check_block_color check_cola_date --episodes 10

# Tier-1B: Official Subtask Executor + Privileged Env Oracle (Upper Bound)
bash scripts/eval_stage2_activearena.sh \
  --mode oracle_subtask_oft --setting ood \
  --tasks check_block_color check_cola_date --episodes 10

# Tier-2: Autonomous IP-VLA Stage-2 Planner (0 Oracle!) -> Frozen oft_subtask_action_12_ws
bash scripts/eval_stage2_activearena.sh \
  --mode ipvla_planner --setting ood \
  --tasks check_block_color check_cola_date --episodes 10 \
  --cerebellum-mode physical_cerebellum \
  --memory-mode inspect_keyframe_pin
```

### 4. Cerebellum & Memory Ablations

Swap `--cerebellum-mode` (`physical_cerebellum` | `no_inertia_k0` | `fixed_period`), `--memory-mode` (`inspect_keyframe_pin` | `sliding_window`), or add `--no-lora` (Base `Qwen3-VL-2B` zero-shot planner) on the exact same seeds.

---

## Stage-1 Open-Video Summary (`Qwen3-VL-2B`, 0 ActiveArena rows)

- **Cross-goal counterfactual (`goal_cross`)**: **`2/2` (4/4 exact)** on held-out open-video recordings.
- **Calibrated Confidence Readout (`Ridge + Softplus Temp`)**: Strictly improves both Brier ($\downarrow$) and AUROC ($\uparrow$) over constant baseline and calibrated token log-probability across all 4 slices (`all`, `os_ost_part3`, `roboturk`, `triplet`).
- **Controlled ActiveArena `check_block_color` 2×2 + Color-Swap (E1)**: **`36/36` (100%)** across `full`, `jump_keep_slot1`, `jump_keep_slot2`, `smooth_no_evidence`, `masked`, and `color_swap`.
