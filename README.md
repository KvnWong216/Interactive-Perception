# IP-VLA: Evidence-Gated Active Perception & Continuous Cerebellum Switching

A clean, plug-and-play research codebase for **IP-VLA** — featuring **Stage-1 zero-simulation causal open-video active perception & calibrated decision confidence** and **Stage-2 closed-loop embodied evaluation on ActiveArena** with unified plug-and-play reproductions of key active-perception baselines (**ASA**, **ZS-IP**, **AAWR**, and official **ActiveArena-VLA OFT**).

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
│   ├── stage1/                       # Stage-1 open-video suite builder, LoRA trainer & Ridge-Temp confidence head
│   ├── stage2/                       # Stage-2 continuous cerebellum inertia, inspect-keyframe memory & switching
│   └── data_pipeline/                # Open-video Teacher A/B causal annotation queue (zero-leak cleanup)
├── eval/
│   ├── open_video/                   # Stage-1 2x2 Jump-Cut vs. Evidence-Removal & Experiment E probes
│   └── activearena/                  # Unified zero-patch ActiveArena closed-loop evaluation suite
│       ├── run_eval.py               # Single CLI for IP-VLA + all 5 baseline modes on identical paired seeds
│       ├── autonomous_policy.py      # IP-VLA Stage-2 zero-oracle closed-loop policy adapter
│       ├── asa_{policy,server}.py    # Baseline [1]: Embodiment-aligned ASA (EgoActiveFlowModelv2)
│       ├── zsip_{policy,server}.py   # Baseline [2]: Zero-Shot Interactive Perception (frozen VLM + EO + memory)
│       ├── aawr_{network,policy,server}.py # Baseline [3]: Asymmetric AWR (privileged Q/V -> unprivileged actor)
│       └── setup_activearena.sh      # One-command bootstrap for ActiveArena, assets, and official checkpoints
├── scripts/
│   ├── eval_stage2_activearena.sh    # Unified ActiveArena evaluation wrapper
│   ├── run_asa_stage3_formal.sh      # End-to-end [Collect -> Train -> Eval] pipeline for ASA [1]
│   ├── run_zsip_formal.sh            # End-to-end [Smoke -> Eval] pipeline for ZS-IP [2]
│   └── run_aawr_formal.sh            # End-to-end [Collect -> Train -> Eval] pipeline for AAWR [3]
├── docs/
│   ├── METHOD_SPEC.md                # Unified Stage-1 & Stage-2 mathematical specification
│   └── COLLABORATOR_AAWR_GUIDE.md    # Collaborator step-by-step guide (including AAWR on 8GB 4060 / 24GB 3090)
└── tests/                            # Full pytest suite covering IP-VLA, ASA, ZS-IP, and AAWR
```

---

## 3-Step Quickstart

```bash
# 1. Install package & run unit tests
pip install -e ".[dev]"
pytest -q

# 2. Bootstrap ActiveArena simulator, assets & official checkpoints (resumable)
cp configs/paths.example.yaml configs/paths.local.yaml   # edit local venv/checkpoint paths if needed
bash eval/activearena/setup_activearena.sh

# 3. Verify readiness (0 GPU usage)
bash scripts/eval_stage2_activearena.sh --mode ipvla_planner --setting id --tasks check_block_color --dry-run
```

---

## Unified Closed-Loop Evaluation Modes (Identical Official Seeds)

All 6 modes run through the exact same `eval/activearena/run_eval.py` harness on identical official `ID` (`info_gathering_demo`) and `OOD` (`info_gathering_randomized`) seed lists:

| Mode Flag (`--mode`) | Method | Training Requirement | Privileged Info at Eval? | One-Command Pipeline |
| --- | --- | --- | :---: | --- |
| `asa_baseline` | **ASA / CoMe-VLA [1]** | Stage-3 Astribot Flow finetune | No (`instruction` mode) | `bash scripts/run_asa_stage3_formal.sh` |
| `zsip_baseline` | **ZS-IP [2]** | **Zero training** (frozen 2B VLM) | No | `bash scripts/run_zsip_formal.sh` |
| `aawr_baseline` | **AAWR [3]** | Offline privileged-$Q/V$ AWR (~1.2 GB VRAM) | **No** (critics discarded) | **`bash scripts/run_aawr_formal.sh`** |
| `instruction_oft` | **ActiveArena-OFT (Instr.)** | Official released 100k-step ckpt | No | `bash scripts/eval_stage2_activearena.sh --mode instruction_oft` |
| `oracle_subtask_oft` | **ActiveArena-OFT + Oracle** | Official released 100k-step ckpt | **Yes (Upper Bound)** | `bash scripts/eval_stage2_activearena.sh --mode oracle_subtask_oft` |
| **`ipvla_planner`** | **IP-VLA (Ours)** | Stage-1 open-video LoRA + frozen OFT | **No** | `bash scripts/eval_stage2_activearena.sh --mode ipvla_planner` |

For detailed collaborator onboarding and customizing **AAWR** on `8 GB RTX 4060` or `24 GB RTX 3090` GPUs, see **[docs/COLLABORATOR_AAWR_GUIDE.md](docs/COLLABORATOR_AAWR_GUIDE.md)**.
