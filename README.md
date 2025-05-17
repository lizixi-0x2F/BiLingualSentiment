# LTC-NCP-VA 情感分析训练项目

基于液态时间常数-神经电路策略(LTC-NCP)的情感价效度(VA)训练框架，支持中英文文本情感分析。

## 项目状态

当前处于**优化阶段**。已完成模型重训，正在针对价值(Valence)维度进行增强训练。详细的重训计划和修改请参阅 [RETRAINING_PLAN.md](RETRAINING_PLAN.md) 和 [TRAINING_MODIFICATIONS.md](TRAINING_MODIFICATIONS.md)。

## 项目结构

- `train.py` - 主要训练脚本
- `evaluate.py` - 评估与指标计算工具
- `ltc_ncp/` - 模型核心实现库
- `configs/` - 模型配置目录
- `data/` - 训练数据目录
- `reports/` - 实验报告目录
- `clean_datasets_final.py` - 数据清洗工具

## 快速开始

### 环境配置

```bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate ltc_ncp_env
```

### 训练模型

```bash
# 标准训练
python train.py --config configs/default.yaml --epochs 30

# 增强价值(Valence)预测训练
python train.py --config configs/valence_enhanced.yaml
```

### 评估模型

```bash
# 评估训练好的模型
python evaluate.py --model_path [模型路径] --config configs/valence_enhanced.yaml
```

## 情感价效度说明

系统基于情感价效度(Valence-Arousal)模型进行训练：

- **价(Valence)**: 表示情感的正负性，从-1(极负面)到+1(极正面)
- **效(Arousal)**: 表示情感的强度/激活度，从-1(平静)到+1(激烈)

四个情感象限：
- 喜悦/兴奋 (价+，效+)
- 满足/平静 (价+，效-)
- 愤怒/焦虑 (价-，效+)
- 悲伤/抑郁 (价-，效-)

## 最新模型架构

增强版LTC-NCP-VA架构的特点：
- **Transformer增强**: 4层、4头注意力机制，前馈维度512
- **双向RNN结构**: 捕捉双向文本上下文
- **多层次LTC单元**: 改进的液态时间常数处理
- **价值(Valence)增强**: 专门的价值分支网络，提高满足/平静象限识别能力
- **维度自适应机制**: 解决输入维度不匹配问题，增强稳定性
- **方向感知损失函数**: 关注情感方向正确性
- **元特征融合**: 整合句长、标点密度等特征

模型参数量：约283万参数

## 改进成果

通过架构优化解决了以下问题：
- 改善了满足/平静象限(Q2)的识别能力
- 提高了价值(Valence)维度的预测准确性
- 增强了模型数值稳定性
- 减轻了过拟合现象

## 后续优化方向

- 继续提升满足/平静象限识别能力
- 完善交互层设计，捕捉价效维度间关系
- 数据扩增与清洗，处理标注不一致问题
