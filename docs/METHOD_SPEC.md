# IP-VLA RSS 2027 统一方法、核心实证与 Stage-2 冻结架构方案（2026-11-21 修订版）

> 本文档固化截至 2026-11-21 18:10 JST 的全部 Stage-1 实证结果、11 月 21 日审稿修正项、新推导的**连续小脑惯性 + 两级信度门控统一数学框架**，以及明日启动 ActiveArena Stage-2 动作-语义联合验证与冲刺 RSS 2027 稿件的执行合同，供 OpenResearch 框架（`gpt-6.2-astra` / `gpt-6.2-sol`）执行严格审稿与 Revise。

---

## 一、 核心研究问题与论文主张边界

在部分可观测机器人操作中，**全局总目标 $G$ 从第 0 秒起已知，真正未知的是物理世界的隐藏属性 $\theta$**（例如灰积木背面颜色贴纸、被布遮挡的指定物体位置、餐具筒内待找物体）。我们研究的问题是：
1. **Stage-1（开放视频自主主动感知高层选择）：** 让 $2\text{B}$ 级 VLM（`Qwen3-VL-2B-Instruct`）在给定全局目标 $G$ 与 $K=4$ 帧因果视觉历史 $H_t$ 下，**自由生成**下一步英文祈使子任务短句 $u_k$，并从生成完毕的最后一个有效词元隐状态 $h_{\mathrm{last}}(G, H_t, u_k) \in \mathbb{R}^{2048}$ 读出**校准后的决策信度** $c(u_k \mid G, H_t) \in [0, 1]$。
2. **Stage-2（ActiveArena 机器人连续控制与小脑惯性仲裁）：** 冻结 VLM 底座，接入原生 18-DoF 动作专家（Action Expert / “小脑”，`QwenOFTState` 契约），通过**纯小脑计算的连续物理执行惯性 $\phi_t$** 与 **Stage-1 决策信度 $c(u')$** 构成两级原子交接机制，实现高频连续控制与自主主动取证闭环。

---

## 二、 已落盘的核心实证结果（全部按严格 `/16` 完整三元组与 `class & noun_ok` 口径核验）

### 2.1 受控机制主证据：ActiveArena 2×2 跳切 vs. 证据移除 + 跨局颜色替换 `36/36 (100%)`（实验 E1）
在 6 局留出 ActiveArena（六色各 1 局，**全局目标 $G$ 与末帧 $f_4$ 字节级完全不变，末帧贴纸 0 像素可见**）上：
- 旧模型 `e49634f5` 在 `full (6/6)`、`color_swap (6/6)`、`smooth_no_evidence (6/6)`、`masked (6/6)` 均全对，但在 `jump_keep_slot2 ([p1, f3_true, p3, f4])` 上暴露了 `0/6` 的第 3 槽位过拟合缺陷；
- **实验 E1（[receipt.json](results/exp_e_slot_and_gmismatch_v1/receipt.json)，仅在训练集加入 Slot-1/2/3 揭示帧槽位随机化 + 零跳切无证据 `[f4]*4`，64 步 `336×252 LoRA r=8`）** 实现全条件 **`36/36 (100.0%)` 满贯**：
  - `full ([f1, f2, f3_true, f4])`: **`6/6`**（放置于真色垫）
  - `jump_keep_slot2 ([p1, f3_true, p3, f4])`: **`6/6`**（同款 `p3->f4` 8.5s 大跳切但保留揭示帧于第 2 槽，仍 100% 放对真色垫）
  - `jump_keep_slot1 ([f3_true, p2, p3, f4])`: **`6/6`**（揭示帧移至最早第 1 槽且后段跳切，仍 100% 放对真色垫）
  - `smooth_no_evidence ([f4, f4, f4, f4])`: **`6/6`**（`0.0s` 零跳切且无揭示帧，仍 100% 退回 `Rotate ... to inspect`）
  - `masked ([p1, p2, p3, f4])`: **`6/6`**（跳切且无揭示帧，100% 退回检查）
  - `color_swap ([f1, f2, f3_OTHER, f4])`: **`6/6`**（`f1/f2/f4` 字节级不变，仅把第 3 帧换为另一局异色贴纸，100% 精准改放新颜色垫！）
- 底座 `Qwen3-VL-2B` 与 856 条通用视频 SFT 在同末帧反转上均为 **`0/6`**。
- 受控集末 Token 隐状态信度读出（`e49634f5`）：**`AUROC = 0.9260`, `Brier = 0.1469`**（vs 校准 Token Log-Prob `0.6538 / 0.2495`、常数 `0.5000 / 0.2827`）。
- 离线 18-DoF 动作接口验证（12 测试锚点）：选对子任务 **`12/12`**，原生关节 $L_1 = \mathbf{0.0694\text{ rad}}$（翻腕检视 `4/6`，变位置色垫落点代理 `2/6`；严格限定为离线验证，明日进入闭环仿真）。

### 2.2 开放视频（OS/OST Part 3 + RoboTurk，严格 `0` ActiveArena）：从捷径诊断到 `G_mismatch` 突破
1. **数据与口径勘误（响应 `REVIEW_2026-11-21_STAGE1_EVIDENCE.md`）：**
   - 固定 144 行测试池中，54 行配对池严格拆分为 **`16` 组完整三元组（48 行，4 段录像：OS 15 组 + RoboTurk 1 组）** 与 6 条独立双行，所有反转指标统一采用严格 **`/16`** 分母与 **`accepted = class_ok and noun_ok`** 口径；
   - 独立末帧审阅（`current_frame_audit.csv`）区分出 25 组三元组（16 固定 + 9 新增 OS 录像）中 **12 组末帧含混（真需历史证据）** 与 **13 组末帧清晰可定位**；
   - 黑勺子案例精确时间戳修正为 **`329.542 s` (`s003`) / `332.542 s` (`s004`)**，明确标注具体目标由示范名词回填构造，并并列保留 `Rank-8` 成功原句（`"Grasp the black spoon"`）与高分辨率 `Rank-32` 保守搜索原句（`"Search through the utensil holder"`）。
2. **发现并手术修复“平滑近景复读目标”捷径（实验 E2，[receipt.json](results/exp_e_slot_and_gmismatch_v1/receipt.json)）：**
   - 旧 6 轮模型 `725c7958`（4,135 样本，无负目标对比）在 4 段新 OS 录像的“同视觉换目标 `goal_cross`”上得 **`0/2`**（盲目复读目标名词去抓），在末帧含混 12 组零跳切 `[f4]*4` 上仅 `1/12` 退回搜索；
   - 仅从训练集构造跨录像错配负目标样本 `G_mismatch`（平滑近景物体 $A$ + 询问另一录像物体 $B \to$ 搜索 $B$）并补训 120 步后（**E2**）：
     - **RoboTurk 未见操作者：** `31/45` $\to$ **`34/45 (75.6%)`**（搜索 `6/10` + 拿取 `9/10`，历史最高）；
     - **4 段全新 OS 录像（9 组新三元组）：** 前后反转 `6/9` $\to$ **`8/9 (88.9%)`**，同末帧命名拿取 $\to$ 搜索翻 **3 倍（`2/9` $\to$ `6/9 = 66.7%`）**；
     - **同视觉换目标 `goal_cross`：** 打破零至 **`1/2`（`3/4` 精确）**——Scene 20 同一组 4 帧问蓝笔答 `"Pick up the blue pen"`，改问黑线缆答 `"Scan the room to search for the black cable"`；
     - **2×2 零跳切清晰/含混分离：** 末帧含混 12 组退回搜索 `1/12` $\to$ **`5/12 (41.7%)`**（`masked` 达 **`9/12`**），末帧清晰 13 组仍保持 **`11/13 (84.6%)`** 拿取（仅 `2/13` 退回）。
3. **正在运行的全量 Stage-1 v2 重构配比（Run `b4af4318`，`6,601` 条全英文样本，`0` ActiveArena，`540` 训练录像 / `20` 验证 / `90` 测试，严格隔离 4 段新留出录像）：**
   - 针对 E2 中 `G_mismatch` 占比 `25%` 导致 Scene 22 略保守的问题，重构 `BS=16` 配比为 **`4 : 2 : 4 : 4 : 2`**（具体目标正例含槽位随机 `25%` / 防假负例多模板 `G_mismatch` **`12.5%`** / RoboTurk `25%` / OS `25%` / 动作描述 `12.5%`）；
   - 修复信度头温度正斜率约束（$a_+ = \mathrm{softplus}(\eta) > 0$）并将验证校准池扩容 2.5 倍至 **`148` 条**；
   - **已出炉前 2 轮（共 6 轮）：** Epoch 1 前后反转即刷新全场最高 **`12/16 (75.0%)`**（揭示前搜索 `14/16`、揭示后命名拿取 `14/16` 高度均衡）；Epoch 2 验证分由 `105.0` 跳升至 **`120.0`**，同末帧命名反转升至 **`6/16`**，预计今晚 19:25 JST 全部完成。

---

## 三、 统一数学框架：连续小脑惯性 $\phi_t$ 与两级信度门控切换（修复 v6.1 两个破绽）

### 3.1 修复目标
旧版 v6.1 的 $\phi_t^{\mathrm{old}} = (1 - d_t) p_t$（$d_t = d_{t^-} \lor [p_t \ge 0.95]$）存在两处理论/物理瑕疵：
1. **初期零阻尼 + $p_t=0.95$ 断崖跳变：** $p_t \to 0^+$ 时阻力为 $0$（抑制不住刚切换后的抖动），而 $p_t \to 0.95^-$ 阻力达极大值后在 $0.95$ 处垂直归零；对主动取证类技能，中后期看清证据后反而阻力越来越大、阻碍早停；
2. **两阶段符号撞车与机制割裂：** Stage-1 信度 $c_k$ 与 Stage-2 惯性 $c_t$ 撞名，且生成后的决策信度未接入执行提交门。

### 3.2 纯小脑连续物理惯性 $\phi_t \in [0, 1]$（不从 VLM 读进度！）
由底层 **小脑（Action Expert + 18-DoF 本体流）** 以 15–30 Hz 高频计算无量纲切换代价：
$$\phi_t^{\mathrm{kin}} = \min\!\left(1,\; \frac{\|\dot{q}_t\|_W}{v_{\mathrm{ref}}}\right) \in [0, 1], \qquad \phi_t^{\mathrm{rem}} = (1 - \tau_t)^{\gamma} \in [0, 1]\ (\gamma \ge 1,\ \text{默认 } \gamma=1),$$
$$\phi_t = \frac{1}{2}\phi_t^{\mathrm{kin}} + \frac{1}{2}\phi_t^{\mathrm{rem}} \in [0, 1].$$
- **主方案（零参数物理小脑）：** $\tau_t = \tau_t^{\mathrm{chunk}} = j / H \in [0, 1]$（当前 $H=16$ 步动作块走到第 $j$ 步的线性相位），叠加实测 18-DoF 加权关节速度 $\phi_t^{\mathrm{kin}}$（$v_{\mathrm{ref}}$ 取训练集 95 分位数冻结）。即使跨多个动作块执行长技能，块末 $j \to H$ 时机械臂若仍在半空高速移动，$\phi_t^{\mathrm{kin}}$ 仍自然接管防中断保护；一旦减速停稳，$\phi_t \to 0$ 连续平滑释放！
- **消融对照（小脑学习型进度头）：** 在 AE 隐特征 + 18-DoF 本体历史上挂 2 层 MLP 预测整段技能进度 $\hat{p}_t^{\mathrm{AE}} \in [0, 1]$（梯度严格 `detach`，不回传 Qwen），替换 $\tau_t = \hat{p}_t^{\mathrm{AE}}$ 做配对比较。

### 3.3 期望净收益唤醒门与四条数学性质保证
令三参数仲裁裕度（与旧版同样仅拟合 3 个标量 $\alpha, \beta, \lambda$）为：
$$\ell_t = \alpha\, r_t + \beta - \lambda\, \phi_t, \qquad q_t = \sigma(\ell_t), \qquad \text{WAKE\_VLM}(t) \iff \ell_t > 0,$$
凸可行域约束：
$$\alpha \ge \epsilon,\qquad \beta \le -\epsilon,\qquad \lambda \ge 0,\qquad \alpha + \beta - \lambda \ge \epsilon \qquad (\epsilon = 10^{-3}).$$
该设计严格满足 4 条性质：
1. **全局连续无断崖：** $\phi_t$ 关于 $(\|\dot{q}_t\|_W, \tau_t)$ 处处连续有界于 $[0, 1]$；
2. **无证据不误唤醒：** $r_t = 0 \implies \ell_t = \beta - \lambda \phi_t \le -\epsilon < 0$；
3. **初期抗抖滞回：** 新技能刚提交 $\tau_t = 0 \implies \phi_t \ge \tfrac{1}{2}$，唤醒阈值自动抬高至 $r_{\mathrm{th}}(0) = \frac{-\beta + \frac{1}{2}\lambda}{\alpha} > \frac{-\beta}{\alpha}$；
4. **强证据可打断无死锁：** 即使满惯性 $\phi_t = 1$，只要 $r_t = 1$ 仍有 $\ell_t = \alpha + \beta - \lambda \ge \epsilon > 0$。

### 3.4 两级原子事务（无缝融合 Stage-1 信度与 Stage-2 惯性）
1. **高频唤醒门（小脑 15–30 Hz，无自回归解码）：** $\text{WAKE\_VLM}(t) \iff \ell_t > 0$；未通过时持续平滑执行当前技能动作块；
2. **低频大脑生成 + Stage-1 信度提交核验：** 唤醒后 VLM 自由生成 $u' \sim \pi_\theta(\cdot \mid G, H_t)$ 并计算正斜率校准的末 Token 决策信度 $c(u' \mid G, H_t)$：
   $$\text{COMMIT}(u') \iff (\ell_t > 0) \;\land\; (u' \neq u_k) \;\land\; \big(c(u' \mid G, H_t) \ge \tau_{\mathrm{conf}}\big).$$
   通过则原子切换 $u_{k+1} \leftarrow u'$ 并重置 $\tau_t \leftarrow 0$；未通过则拒绝无证据盲目提交。

---

## 四、 明日启动 Stage-2 ActiveArena 动作-语义联合验证与冲刺 RSS 计划

1. **Qwen 冻结策略与双对照消融：**
   - **底座（ViT + 28 层 LLM 主权重）100% 冻结**，动作头（18-DoF `QwenOFTState`）及小脑进度头梯度严格 `detach` 不回传污染 VLM 语义空间；
   - 做一组关键对照：**（a）纯冻结 Stage-1 v2 权重（含 LoRA 全冻结，仅训动作头/唤醒头/仲裁标量）** vs **（b）冻结底座 + ActiveArena 极轻量 LoRA 领域适配（混入 25% Stage-1 开放视频防遗忘 + 槽位随机化）**。
2. **小脑惯性 3 路配对消融（同一 ActiveArena 闭环仿真与相同种子）：**
   - `(A)` 无惯性基线（$\lambda = 0$）；
   - `(B)` 零参数物理小脑主方案（$\phi_t = \frac{1}{2}\phi_t^{\mathrm{kin}} + \frac{1}{2}(1 - j/H)$）；
   - `(C)` 学习型小脑进度头（AE 2 层 MLP $\hat{p}_t^{\mathrm{AE}}$）对比旧线性锁存 $(1 - d_t)\hat{p}_t^{\mathrm{AE}}$。
3. **五段闭环分解指标：** 在 ActiveArena 仿真闭环上拆开统计 `高层选对 -> 执行取证 -> 看见新证据 -> 改口执行 -> 任务最终成功` 转化率、无效切换次数（chattering）与任务耗时，形成完整 RSS 2027 稿件。
