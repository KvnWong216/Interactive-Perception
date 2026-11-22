# 合作者开发与基线复现指南（含 AAWR 即拉即用全流程）

本仓库采用**完全统一的“三段式 + 客户端-服务端解耦”规范**接入所有对比基线（`ASA`、`ZS-IP`、`AAWR`、官方 `OFT` 以及我们的 `IP-VLA`），保证所有方法在**完全相同的官方评测种子、零特权泄漏、零修改上游仿真器**的前提下公平对比，并且天然支持在 **8 GB RTX 4060** 或 **24 GB RTX 3090** 上直接运行。

---

## 一、 3 分钟即拉即用启动

```bash
git clone <REPO_URL> ip-vla
cd ip-vla
pip install -e ".[dev]"
pytest -q
```

1. 复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`（已被 `.gitignore` 忽略），填入本机的 `ActiveArena` 与 Python 环境路径；
2. 若新机器尚未下载 ActiveArena 仿真器、资产与权重，一键执行：
   ```bash
   bash eval/activearena/setup_activearena.sh
   ```
3. 零显存预检所有模式是否就绪：
   ```bash
   bash scripts/eval_stage2_activearena.sh --mode aawr_baseline --setting id --tasks check_block_color --dry-run
   ```

---

## 二、 统一基线复现范式（对照 `ASA` / `ZS-IP` / `AAWR`）

仓库里每一条自包含基线都严格拆成以下 **4 个标准组件**（合作者扩写或调优任何基线只需改对应文件）：

| 标准环节 | `ASA [1]` 对应文件 | `ZS-IP [2]` 对应文件 | **`AAWR [3]` 对应文件（本次新增）** |
| --- | --- | --- | --- |
| **1. 零泄漏离线数据采集** | `scripts/collect_asa_astribot_data.py` (`seed >= 500000`) | 无需训练（0 样本） | `scripts/collect_aawr_astribot_data.py` (`seed >= 600000`，含训练期特权状态 `priv_state` 64-D + 专家/次优混合轨迹) |
| **2. 离线模型训练** | `scripts/train_asa_astribot_stage3.py` | 无需训练（冻结 2B VLM） | `scripts/train_aawr_astribot.py` + `eval/activearena/aawr_network.py`（严格复现原仓库 `asym_awr.py`） |
| **3. 独立 HTTP 推理服务 + 仿真策略适配器** | `eval/activearena/asa_server.py` + `asa_policy.py` (`:19200`) | `eval/activearena/zsip_server.py` + `zsip_policy.py` (`:19300`) | `eval/activearena/aawr_server.py` + `aawr_policy.py` (`:19400`，评测期**100% 剥离特权状态**） |
| **4. 一键全流程脚本** | `bash scripts/run_asa_stage3_formal.sh` | `bash scripts/run_zsip_formal.sh` | **`bash scripts/run_aawr_formal.sh`** |

---

## 三、 AAWR 基线原理与即开即用命令

### 1. AAWR 核心机制（严格对齐上游 `aawr-main/src/algorithm/asym_awr.py`）
- **训练期非对称特权价值估计（Privileged Critics）**：双 $Q$ 网络 `_Qs` 与期望分位数价值网络 `_V`（`expectile = 0.9`）读取**仅在训练期可见的 64 维特权状态 $s_t^+$**（目标积木/可乐罐真实 7D 位姿、真实隐藏贴纸颜色/保质期哈希、真实子任务阶段 one-hot），学习优势值 $A_t = \min_{i=1,2} Q_{\bar{\phi}_i}(s_t^+, a_t) - V_\psi(s_t^+)$；
- **无特权策略加权回归（Unprivileged Actor）**：策略网络 $\pi_\theta(a_{t:t+H} \mid o_{t-3:t}, q_t, G)$ **只看 4 帧因果头相机 RGB + 18 维关节角 + 去剧透指令**，按指数优势权重 $w_t = \min(\exp(3.0 \cdot A_t), 100)$ 加权回归动作块；
- **评测期零特权闭环（`--mode aawr_baseline`）**：只加载 $\pi_\theta$，在官方留出测试种子上完全自主闭环执行。

### 2. 一键全自动跑完 `[1/3] 采数 -> [2/3] 训练 -> [3/3] 闭环评测`

```bash
# 默认跑 3 个代表性主动感知任务 (check_block_color, check_cola_date, check_cola_color)
# 单张 8GB RTX 4060 即可轻松跑完（训练显存仅 ~1.2 GB，仿真显存 ~2.2 GB）！
GPU_ID=0 EPISODES_PER_TASK=30 MAX_STEPS=4000 EVAL_EPISODES=10 \
  bash scripts/run_aawr_formal.sh
```

### 3. 分步调试与合作者扩展方向（适合本科生分工深化）
1. **快速冒烟验证（3 分钟走通全链路）**：
   ```bash
   EPISODES_PER_TASK=2 MAX_STEPS=100 EVAL_EPISODES=1 TASKS="check_block_color" \
     bash scripts/run_aawr_formal.sh
   ```
2. **升级视觉编码器或在线交互轮次（AAWR 论文 Offline-to-Online 扩展）**：
   - 在 `eval/activearena/aawr_network.py` 中可将轻量 `UnprivilegedVisualEncoder` 替换为冻结 `DINOv2`（对应原论文 `src/models/dino_adapter.py`）或冻结 `Qwen3-VL-2B` 视觉特征；
   - 可在 `scripts/collect_aawr_astribot_data.py` 中用训练出的当前策略 $\pi_\theta$ 在训练种子上继续采样 on-policy 交互轨迹并追加进 `manifest.json`（对应原论文 `iterations=6` 的 offline-to-online 迭代），切记**训练种子永远保持 `seed >= 600000`**，严禁触碰 `eval_seed_lists/`！

---

## 四、 团队 Git 协作硬规矩
1. 每位同学在独立分支开发：`git checkout -b feat/<姓名>-aawr`；
2. 提交 PR 前必须通过：`pytest -v`；
3. **三条红线**：① 严禁把 `.pt` / `.safetensors` / `.npz` / `runs/` 提交进 Git；② 严禁在评测期读取 `task_env` 的任何特权字段；③ 所有方法统一汇报 `runs/activearena/<setting>/<run_id>/summary.json` 中的同种子成功率。
