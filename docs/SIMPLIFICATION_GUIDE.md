# LTC-NCP-VA项目简化指南

本文档详细说明了如何将LTC-NCP-VA项目简化为基础版本，移除高级功能特性。

## 移除的功能

为了简化项目，以下高级功能已被移除：

1. **FGM对抗训练**：移除了在嵌入层添加扰动的对抗训练功能
2. **软标签(Label Smoothing)**：移除了标签平滑技术
3. **多任务学习头**：移除了四象限分类辅助任务头

## 变更摘要

### 1. 配置文件变更

创建了新的简化版配置文件 `configs/simple_base.yaml`，主要变更包括：

- 禁用对抗训练相关配置
- 禁用标签平滑
- 禁用四象限分类头
- 简化模型架构参数

### 2. 模型文件变更

修改了 `src/core/model.py`，主要变更包括：

- 更改默认参数，将高级功能设为禁用状态
- 移除了四象限分类头的相关代码
- 简化了模型前向传播逻辑，直接返回VA回归输出

### 3. 训练脚本变更

创建了简化版训练脚本 `src/train_simple.py`，主要变更包括：

- 移除了对FGM对抗训练模块的引用和使用
- 移除了边界样本权重模块
- 移除了LabelSmoothingLoss和QuadrantClassificationLoss类
- 使用标准MSE损失函数替代复杂的损失函数组合
- 简化了训练和评估逻辑

### 4. 新增运行脚本

创建了用于启动简化版训练的脚本：

- `scripts/run_simple_train.ps1` (PowerShell版)
- `scripts/run_simple_train.sh` (Bash版)

### 5. 移除的文件

在简化过程中，一些与高级功能相关的文件已被移除：

- `src/core/adversarial.py`：FGM对抗训练实现
- `src/core/boundary_weights.py`：边界样本权重计算实现

更详细的文件移除信息，请参见 [REMOVED_FILES.md](./REMOVED_FILES.md)。

## 如何使用简化版

按照以下步骤使用简化版进行训练：

1. 确保环境已安装所有必要依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 使用PowerShell运行训练脚本：
   ```powershell
   .\scripts\run_simple_train.ps1
   ```
   或使用Bash：
   ```bash
   bash scripts/run_simple_train.sh
   ```

3. 可以直接运行Python脚本并自定义参数：
   ```bash
   python src/train_simple.py --config configs/simple_base.yaml --epochs 30
   ```

## 简化版本的优缺点

**优点：**
- 代码更加简洁易懂
- 训练速度更快
- 内存占用更少
- 更适合教学和入门理解

**缺点：**
- 模型鲁棒性可能降低（无对抗训练）
- 可能对边界情感区分能力下降
- 性能可能略有下降（无多任务学习增强）

## 如何恢复高级功能

如果需要恢复完整功能集，请使用原始的训练脚本和配置文件：

```bash
python src/train.py --config configs/valence_enhanced.yaml
```

或运行完整版训练脚本：

```bash
bash scripts/run_5m_training.sh
```
