# LTC‑NCP‑VA 回归项目

> **使用液态时间常数（Liquid‑Time‑Constant, LTC）单元 + 神经电路策略（Neural Circuit Policy, NCP）稀疏连线 + Transformer增强，自零开始实现文本价效度（Valence‑Arousal, VA）回归。**
>
> 面向中英双语情绪分析，聚焦科研实验与模型训练。

---

## 📐 1 分钟原理速览

| 模块 | 思想 |
| --- | --- |
| **LTC 单元** | 隐状态满足一阶 ODE，时间常数 `τ` 可学习：\(h_{t+1}=h_{t}+\frac{Δt}{τ_t}(-h_t+W_xx_t+W_hσ(h_t))\)。连续时间动态提升时间分辨率。 |
| **NCP 连线** | 模仿线虫神经回路的三层稀疏图（传感 → 中间 → 运动），减少约65%参数量，增强可解释性。 |
| **Transformer增强** | 融合自注意力机制，捕捉全局文本依赖关系，增强特征表示能力，提升情感识别准确性。 |
| **融合架构** | Token 嵌入 → 多层次NCP‑LTC‑RNN → Transformer编码 → 双分支情感头，输出 \([V, A]\in[-1,1]\) 连续值，可精确捕捉细腻情感波动。 |

完整技术细节见 **`docs/00_theory.pdf`**。

---

## 🛠️ 项目结构

```
.
├── data/                  # 原始与处理后数据集
│   ├── scripts/           # 清洗 / 增强脚本
├── ltc_ncp/               # 核心模型库
│   ├── cells.py           # LTCCell 实现（支持Euler/RK4积分）
│   ├── wiring.py          # NCP 稀疏连接生成器
│   └── model.py           # LTC_NCP_RNN 主架构实现
├── configs/               # YAML 实验配置
├── train.py               # 训练入口
├── evaluate.py            # 指标与可视化
├── run_fixed.sh           # 增强版训练脚本
└── reports/               # 实验报告
```

---

## 📦 安装指南

```bash
# 1. 克隆仓库
$ git clone https://github.com/yourname/ltc-ncp-va.git && cd ltc-ncp-va

# 2. 创建环境（推荐 PyTorch ≥2.3）
$ conda env create -f environment.yml && conda activate ltcva

# 3. 下载数据集（≈ 5 万句）
$ bash data/scripts/fetch_all.sh
```

> **硬件建议**：单张 RTX 3090（24 GB）或RTX 4090（24+ GB）即可支持 batch 128。仅 CPU 训练约慢 10 倍。

---

## 🚀 快速上手

```bash
# 使用增强版训练脚本（基于RMSE优化）
$ bash run_fixed.sh

# 或手动指定参数
$ python train.py \
        --config configs/optimized_performance.yaml \
        --epochs 30 --amp
```

输出文件：
* `results/metrics.json` — RMSE、CCC、Spearman ρ
* `results/plots/` — VA 散点图与时间序列叠图

---

## 📊 增强架构与性能

### 2.0版本创新点

- **Transformer增强**：4层Mini-Transformer架构，与LTC-RNN深度融合，增强全局依赖性捕捉能力
- **维度适配机制**：新增特征适配层(feature_adapter)和专用输入适配器(input_adapter)，解决维度不匹配问题
- **双分支情感头**：专用V/A分支，通过情感交互层捕捉情感维度间的复杂关系
- **增强型鲁棒架构**：全链路异常处理与回退机制，提升数值稳定性，处理NaN值和极端情况
- **注意力增强**：引入自注意力机制，对不同情感特征进行智能权重分配
- **MSE-RMSE优化目标**：由CCC评估标准转向RMSE作为主要优化目标，精确量化绝对误差

### 主要特性

- **动态维度处理**：自适应处理批次和序列维度，支持变长文本
- **双向增强**：双向LTC-RNN提升情感理解的上下文感知能力
- **高效稀疏连接**：NCP掩码减少~65%参数量，同时保持表现力
- **元特征融合**：句长、标点密度等统计特征增强模型性能
- **参数规模提升**：总参数量超过510万，显著增强模型表达能力

### 实验结果

* **RMSE优化**：在验证集上实现约0.24的RMSE，同时保持较高CCC
* **V-A收敛性**：V维度CCC达到0.52，A维度CCC达到0.57，体现情感维度间的有效建模
* **训练稳定性**：异常处理机制确保训练过程中无NaN问题，梯度传播平稳

详见 `reports/` 目录下的实验报告。

---

## 🧩 技术细节

- **维度处理**：支持批量序列处理 `[batch_size, seq_len, hidden_size]`
- **状态追踪**：记录所有时间步隐状态，便于可视化和分析
- **梯度钩子**：使用`register_full_backward_hook`确保稀疏连接在反向传播中保持
- **混合精度训练**：支持FP16混合精度，加速训练
- **设备一致性保证**：确保所有组件位于相同计算设备上，避免不必要的数据迁移
- **调试控制**：全局DEBUG开关和分级日志系统，便于问题诊断

---

## 🧱 Roadmap

- [ ] VA + 情绪极性多任务学习
- [ ] 集成 IG 可解释性可视化
- [ ] 跨语言情感对齐优化
- [ ] 探索Transformer特征与LTC动态特征的进一步融合方案
- [ ] 开发针对跨语言情感分析的预训练策略

欢迎 PR 与 Issue！

---

## 📄 许可证

Apache 2.0 © 2025 李籽溪 / 中山大学 HCP 实验室

---

## ✨ 引用

```bibtex
@software{lee2025ltcncpva,
  author = {Li, Zixi and Lee, Tianlu},
  title = {LTC‑NCP‑VA Regression},
  year = {2025},
  url = {https://github.com/yourname/ltc-ncp-va},
  note = {Version 2.0}
}
```

---

> _"液态时间与注意力机制，共同筛出情感的更细颗粒。"_ – 项目口号
