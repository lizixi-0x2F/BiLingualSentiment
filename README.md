# LTC-NCP-VA: 基于液态时间常数网络的文本情感价效度分析

## 项目概述

LTC-NCP-VA是一个专注于文本情感价效度(Valence-Arousal)分析的深度学习框架，基于液态时间常数(Liquid Time-Constant)网络与神经电路策略(Neural Circuit Policy)的创新组合。本项目能够从文本中提取情感的两个核心维度：价值(Valence，情感的正负极性)和效度(Arousal，情感的激烈程度)，实现更细粒度的情感分析。

### 价效度(VA)空间解释

情感价效度空间将情感映射到二维坐标系中：

- **价值(Valence)**: 横轴，表示情感的正负程度，范围[-1, 1]
  - 正值表示积极情感(如喜悦、满足)
  - 负值表示消极情感(如悲伤、愤怒)

- **效度(Arousal)**: 纵轴，表示情感的激烈程度，范围[-1, 1]
  - 正值表示高唤起(如兴奋、愤怒)
  - 负值表示低唤起(如平静、疲倦)

四个象限代表不同的情感类别：
- 第一象限(+V, +A): 喜悦/兴奋
- 第二象限(+V, -A): 满足/平静
- 第三象限(-V, +A): 愤怒/焦虑
- 第四象限(-V, -A): 悲伤/抑郁

## 核心特性

- **双语支持**: 同时支持中文和英文文本的情感分析
- **细粒度情感**: 不仅分析情感极性，还能识别情感强度和激烈程度
- **LTC神经网络**: 基于液态时间常数的递归神经网络，适合处理时序信息
- **NCP稀疏连接**: 使用神经电路策略实现高效稀疏连接，提高泛化能力
- **先进增强技术**: 
  - **FGM对抗训练**: 通过向嵌入层添加扰动增强模型鲁棒性
  - **边界样本权重**: 对VA空间中的边界样本给予更高权重，提高区分能力
  - **VA多任务学习**: 联合优化VA回归与四象限分类，提高预测准确性

## 模型架构

LTC-NCP-RNN模型结合了多项先进技术：

1. **液态时间常数(LTC)细胞**: 一种改进的RNN单元，通过可学习的时间常数处理不同时间尺度的依赖关系
2. **神经电路策略(NCP)布线**: 基于神经科学的稀疏连接方案，降低参数量，提高泛化能力
3. **分支架构**: 专门的V分支和A分支分别处理价值和效度预测
4. **多任务学习**: 同时进行VA回归和四象限分类，互相促进性能提升
5. **注意力机制**: 引入注意力层突出重要特征
6. **情感方向感知**: 特殊设计的损失函数，对情感象限预测错误施加更高惩罚

## 安装与设置

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (GPU训练推荐)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/ltc-ncp-va.git
cd ltc-ncp-va
```

2. 创建并激活虚拟环境(可选)
```bash
conda create -n ltc-ncp-va python=3.8
conda activate ltc-ncp-va
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

使用标准配置训练模型：
```bash
python src/train.py --config configs/valence_enhanced.yaml
```

使用FGM对抗训练和VA多任务学习：
```bash
python src/train.py --config configs/valence_enhanced.yaml --epochs 50
```

使用screen后台训练：
```bash
./scripts/run_5m_training.sh
```

### 评估模型

在测试集上评估模型性能：
```bash
python src/evaluate.py --ckpt runs/models/best_model.pt --output results/evaluation
```

### 情感预测

使用训练好的模型进行情感预测：
```bash
python src/emotion_predict.py --text "这是一段测试文本，我感到非常高兴！"
```

或进入交互模式：
```bash
python src/emotion_predict.py
```

## 项目结构

```
├── configs/                # 配置文件
├── src/                    # 源代码
│   ├── core/               # 核心模型实现
│   │   ├── cells.py        # LTC细胞实现
│   │   ├── wiring.py       # NCP布线实现
│   │   ├── model.py        # LTC-NCP-RNN模型定义
│   │   ├── adversarial.py  # FGM对抗训练
│   │   └── boundary_weights.py # 边界样本权重
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── emotion_predict.py  # 情感预测脚本
├── scripts/                # 辅助脚本
├── data/                   # 数据目录
│   ├── processed/          # 处理后的数据
│   └── raw/                # 原始数据
├── checkpoints/            # 模型检查点
├── runs/                   # 训练运行目录
│   └── models/             # 训练好的模型
├── logs/                   # 日志文件
├── results/                # 评估结果
├── docs/                   # 文档
└── reports/                # 项目报告
```

## 性能指标

在双语情感数据集上的表现：

| 指标 | 价值(Valence) | 效度(Arousal) | 平均 |
|------|--------------|--------------|------|
| CCC  | 0.823        | 0.791        | 0.807|
| RMSE | 0.342        | 0.375        | 0.359|
| 象限准确率 | - | - | 78.6% |

## 贡献者

- 李子溪 (LiZiXi)

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 引用

如果你在研究中使用了LTC-NCP-VA，请引用：

```
@misc{lizx2023ltcncpva,
  author = {Li, Zixi},
  title = {LTC-NCP-VA: Text Emotion Analysis with Liquid Time-Constant Networks},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ltc-ncp-va}
}
```
