# 中英文混合情感分析模型（Pre-trained Transformer Based）

基于预训练Transformer模型的情感分析系统，用于预测中英文文本的效价(Valence)和唤醒度(Arousal)值。

**✅ 当前状态：已完成微调，模型可用于推理**
- DistilBERT模型: `outputs/pretrained_distilbert_local/best_model.pth`
- XLM-RoBERTa模型: `outputs/pretrained_xlm_roberta_local/best_model.pth`

## 项目概述

本项目实现了一个高效的双语情感分析系统，基于预训练的Transformer模型（DistilBERT和XLM-RoBERTa），可同时处理中文和英文文本，输出情感的效价(Valence，表示情感正负程度)和唤醒度(Arousal，表示情感强度)值，范围在[-1, 1]之间。

### 主要特点

- **双语支持**：同时支持中文和英文情感分析，使用混合数据集训练
- **高精度**：利用预训练语言模型的强大表示能力，R²分数优于传统方法
- **便捷微调**：支持对预训练模型进行高效微调，适应不同领域的情感分析需求
- **转移学习**：利用预训练模型的语言知识降低对标注数据的依赖
- **灵活部署**：支持多种预训练模型选择，适应不同资源限制和精度需求
- **高效训练**：支持模型层冻结、学习率预热和衰减等优化策略
- **通用接口**：提供统一的数据处理和模型接口，便于扩展支持更多模型

### 支持的模型

1. **DistilBERT多语言版**
   - 轻量级预训练模型，支持多种语言
   - 相比原始BERT模型体积减少40%，推理速度提升60%
   - 保持较高的多语言理解能力

2. **XLM-RoBERTa**
   - 在100种语言上预训练的强大模型
   - 优秀的跨语言转移能力
   - 处理多语言文本时表现更佳

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU (推荐8GB+显存)
- RAM: 16GB+
- CPU: 多核处理器

### 软件需求
- Python 3.8+
- CUDA 11.0+ (如使用GPU)
- PyTorch 1.9+

### 依赖安装
```bash
pip install -r requirements.txt
```

## 数据集说明

本项目使用两个标准化的情感数据集：

1. **中文VA数据集** (`Chinese_VA_dataset_gaussNoise.csv`)
   - 约4,100条中文文本及其效价-唤醒度标注
   - 效价范围：[-0.85, 0.86]，均值：0.03
   - 唤醒度范围：[-0.66, 0.96]，均值：0.27

2. **EmoBank英文数据集** (`emobank_va_normalized.csv`)
   - 约10,000条英文文本及其效价-唤醒度标注
   - 效价范围：[-0.9, 0.8]，均值：-0.01
   - 唤醒度范围：[-0.6, 0.7]，均值：0.02

### 数据预处理

训练过程中会自动进行以下预处理：
- 使用预训练模型的tokenizer进行分词
- 自动处理中英文文本，无需手动区分
- 标签归一化至[-1, 1]范围
- 训练/验证/测试集划分（比例可在配置中调整）

## 模型架构

本项目实现了基于预训练模型的情感分析架构，主要组件包括：

```
PretrainedSentimentModel
├── Transformer Encoder (DistilBERT/XLM-RoBERTa)
│   ├── 预训练词嵌入层
│   ├── 多层Transformer块
│   └── 上下文化表示
├── Pooling Strategy (CLS/Mean)
│   └── 序列表示提取
└── Regression Head
    ├── 多层全连接网络
    ├── GELU激活和Dropout
    └── Tanh输出层（保证[-1,1]范围）
```

### 关键设计

1. **模型选择灵活性**
   - 支持在DistilBERT和XLM-RoBERTa间切换
   - 适应不同资源限制和精度需求

2. **微调策略**
   - 冻结底层网络提高训练效率
   - 较小的学习率(2e-5)适合微调

3. **表示优化**
   - 支持CLS标记和平均池化两种策略
   - 多层回归头增强特征提取能力

4. **正则化技术**
   - 梯度裁剪防止梯度爆炸
   - Dropout和权重衰减减少过拟合
   - 学习率预热和衰减优化训练过程

## 训练指南

### 快速开始

运行预训练模型训练脚本：

```bash
./run_train.sh
```

### 自定义训练

可通过编辑脚本或传递参数自定义训练过程：

```bash
# 编辑脚本中的参数
notepad run_train.sh

# 或手动运行训练脚本指定参数
python train.py --device cuda --batch_size 32 --epochs 20 --lr 2e-5 --model_type distilbert --monitor r2
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 计算设备(cuda/cpu/mps) | cuda |
| `--batch_size` | 批处理大小 | 32 |
| `--epochs` | 训练轮次 | 20 |
| `--lr` | 学习率 | 2e-5 |
| `--weight_decay` | 权重衰减 | 0.01 |
| `--monitor` | 早停监控指标(val_loss/r2/rmse/ccc) | r2 |
| `--patience` | 早停耐心值 | 3 |
| `--model_type` | 模型类型(distilbert/xlm-roberta) | distilbert |
| `--output_dir` | 输出目录 | 自动生成 |

### 训练技巧

1. **模型选择**
   - `distilbert`: 适合资源有限情况，训练和推理更快
   - `xlm-roberta`: 适合需要更高准确度的情况，特别是在多语言场景

2. **学习率调整**
   - 对于预训练模型，推荐使用较小的学习率(1e-5 ~ 5e-5)
   - 系统内置学习率预热和衰减调度器

3. **层冻结策略**
   - 通过修改`config.py`中的`FREEZE_LAYERS`参数可以调整冻结底层数量
   - 冻结更多层可以加快训练，但可能略微降低性能

4. **池化策略**
   - `cls`: 使用[CLS]标记作为序列表示，适合分类任务
   - `mean`: 使用平均池化，有时可以提供更好的表示

5. **批量大小**
   - 根据GPU显存调整批量大小，推荐16-64之间
   - 太小的批量可能导致训练不稳定，太大的批量可能降低泛化能力

## 项目结构

详细的项目结构请查看 `PROJECT_STRUCTURE.md`。

```
.
├── shells/                     # 脚本目录
│   ├── run_train_local_models.ps1  # 本地模型训练脚本(PowerShell)
│   ├── run_train_local_models.sh   # 本地模型训练脚本(Bash)
│   ├── test_model.ps1          # 模型测试脚本(PowerShell)
│   └── test_model.sh           # 模型测试脚本(Bash)
├── train.py                    # 主训练脚本
├── train_pretrained.py         # 预训练模型微调脚本
├── README.md                   # 项目文档
├── requirements.txt            # 依赖列表
├── Chinese_VA_dataset_gaussNoise.csv  # 中文情感数据集
├── emobank_va_normalized.csv   # 英文情感数据集
├── outputs/                    # 模型输出目录
│   ├── pretrained_distilbert_local/  # DistilBERT模型输出
│   │   ├── best_model.pth      # 最佳模型权重
│   │   └── test_results.json   # 测试结果
│   └── pretrained_xlm_roberta_local/ # XLM-RoBERTa模型输出
│       ├── best_model.pth      # 最佳模型权重
│       └── test_results.json   # 测试结果
├── pretrained_models/          # 预训练模型目录
│   ├── distilbert-multilingual/# DistilBERT预训练模型
│   └── xlm-roberta-base/       # XLM-RoBERTa预训练模型
└── src/                        # 源代码
    ├── config.py               # 配置参数
    ├── inference.py            # 推理脚本
    ├── models/                 # 模型定义
    │   └── roberta_model.py    # 预训练模型实现
    └── utils/                  # 工具函数
        ├── data_utils.py       # 数据处理
        └── train_utils.py      # 训练工具
```

## 关键配置参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| MODEL_TYPE | 模型类型 | 'distilbert'或'xlm-roberta' |
| MULTILINGUAL_MODEL_NAME | 预训练模型名称 | 'distilbert-base-multilingual-cased' |
| POOLING_TYPE | 池化策略 | 'cls'或'mean' |
| FREEZE_LAYERS | 冻结层数量 | 4 |
| LEARNING_RATE | 学习率 | 2e-5 |
| BATCH_SIZE | 批量大小 | 32 |
| DROPOUT | Dropout率 | 0.3 |
| WEIGHT_DECAY | 权重衰减 | 0.01 |
| EARLY_STOPPING_METRIC | 早停指标 | 'r2' |
| WARMUP_RATIO | 预热步数比例 | 0.1 |

## 推理使用

### 使用便捷脚本

项目提供了便捷的测试脚本，可以快速测试模型并生成可视化结果。脚本会自动查找最新训练的模型（位于`outputs/pretrained_distilbert_local`或`outputs/pretrained_xlm_roberta_local`）：

**Windows (PowerShell)：**

```powershell
# 单文本推理 (使用DistilBERT模型)
.\shells\test_model.ps1 -ModelPath "outputs/pretrained_distilbert_local/best_model.pth" -ModelType "distilbert" -SingleText -Text "这是一个测试文本，我感到非常开心！"

# 单文本推理 (使用XLM-RoBERTa模型)
.\shells\test_model.ps1 -ModelPath "outputs/pretrained_xlm_roberta_local/best_model.pth" -ModelType "xlm-roberta" -SingleText -Text "这是一个测试文本，我感到非常开心！"

# 批量文本推理
.\shells\test_model.ps1 -ModelPath "outputs/pretrained_distilbert_local/best_model.pth" -InputFile "example_texts.txt" -OutputFile "predictions.csv"
```

**Linux/macOS (Bash)：**

```bash
# 单文本推理 (使用DistilBERT模型)
bash ./shells/test_model.sh --model_path "outputs/pretrained_distilbert_local/best_model.pth" --model_type "distilbert" --single_text --text "这是一个测试文本，我感到非常开心！"

# 单文本推理 (使用XLM-RoBERTa模型)
bash ./shells/test_model.sh --model_path "outputs/pretrained_xlm_roberta_local/best_model.pth" --model_type "xlm-roberta" --single_text --text "这是一个测试文本，我感到非常开心！"

# 批量文本推理
bash ./shells/test_model.sh --model_path "outputs/pretrained_distilbert_local/best_model.pth" --input_file "example_texts.txt" --output_file "predictions.csv"
```

### 直接使用推理脚本

也可以直接使用`src/inference.py`脚本进行推理，使用已经微调的模型：

```bash
# 使用DistilBERT模型
python src/inference.py `
    --model_path outputs/pretrained_distilbert_local/best_model.pth `
    --model_type distilbert `
    --device cuda `
    --text "这是一个测试文本，我感到非常开心！" `
    --visualize

# 使用XLM-RoBERTa模型
python src/inference.py `
    --model_path outputs/pretrained_xlm_roberta_local/best_model.pth `
    --model_type xlm-roberta `
    --device cuda `
    --text "这是一个测试文本，我感到非常开心！" `
    --visualize
```

或批量推理文本文件：

```bash
python src/inference.py `
    --model_path outputs/pretrained_distilbert_local/best_model.pth `
    --model_type distilbert `
    --device cuda `
    --input_file example_texts.txt `
    --output_file predictions.csv `
    --visualize
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型文件路径 | 必需 |
| `--model_type` | 模型类型(distilbert/xlm-roberta) | distilbert |
| `--device` | 计算设备(cuda/cpu/mps) | cuda |
| `--text` | 单个文本内容进行推理 | 无 |
| `--input_file` | 批量文本文件路径(.txt或.csv) | 无 |
| `--output_file` | 结果输出文件路径(.csv) | 无 |
| `--batch_size` | 批处理大小 | 16 |
| `--visualize` | 生成可视化结果 | False |
| `--no_browser` | 不自动打开浏览器查看结果 | False |
| `--debug` | 显示调试信息 | False |

## 评估与性能

训练完成后，系统会自动在测试集上进行评估并生成以下输出：

- **best_model.pth**：性能最佳的模型权重（基于R²指标）
- **metrics_epoch_*.json**：每个epoch的详细评估指标
- **test_results.json**：在测试集上的详细评估结果
- **visualizations/**：包含各种可视化结果的目录

### 已训练模型位置

微调后的模型保存在以下位置：

1. **DistilBERT多语言模型**: `outputs/pretrained_distilbert_local/best_model.pth`
   - R²: 0.66
   - RMSE: 0.21

2. **XLM-RoBERTa模型**: `outputs/pretrained_xlm_roberta_local/best_model.pth`
   - R²: 0.69
   - RMSE: 0.20

### 主要评估指标

- **R²**：确定系数，越接近1表示模型解释性越强
- **RMSE**：均方根误差，越小越好
- **MAE**：平均绝对误差，越小越好

## 可视化功能

本项目提供多种可视化方式帮助理解和分析模型的情感预测结果：

### 训练过程可视化

训练完成后，系统自动生成以下可视化：

- **训练与验证损失曲线**：显示模型在训练过程中的收敛情况
- **效价-唤醒度散点图**：比较预测值和真实值的分布
- **预测误差分布图**：可视化预测误差的统计分布
- **情感分类混淆矩阵**：评估模型在四个情感象限中的分类性能
- **特征空间可视化**：通过t-SNE降维展示模型学到的高维特征分布

### 推理结果可视化

在推理时使用`--visualize`参数可生成：

- **单个文本**：生成HTML报告，包含详细的情感分析结果和象限图
- **批量文本**：生成交互式散点图，可通过悬停查看每个文本的详细信息

### 样例可视化

![效价-唤醒度散点图](https://mermaid.ink/img/pako:eNpNj0EOgjAQRa9CM3FhIsaFiQlhQaIJCZG4oYtpGRstSFuaMCF3d6LI7v_vvEySHoyVHdAImhoZ3kgbXGZOCaunKA_qJs2jyXiYZL5oZLSgydLY83t8nCBxU0ZKnUzk--N-k0vAa0dPpQw2F6_SQrBMtDnLaj5iVcKD-iH35Nau6sFEHrtHkhVdSAGdPgfOCC5JkHYrbKBzeG5kAOoV3iCdLeo9nlAdLXTQmk6fF2lCQQ-KGLpfbtNXYQ?type=png) ![情感分类混淆矩阵](https://mermaid.ink/img/pako:eNpNj8sKwjAQRX9lmJWCVF0ILlyISwUFKW7sJiRTG9HEZJIG8eO9VaTOau45L2ZaUEa0IHsQZdfzp5a9dVw2OJbWuBsuHJ9KJRYm9MdlmZle5ZTvRVXdN5dlKHBb0zK2gbEeafiUEeyX3-TCtpvUBLY2VbBH6QduYolbHmLm5VHwCEOPGStA-2MxZZChQTfZrJWryIOM9ZdDHTiaSB5E0ujI5CxYM0i7ipfoLPaVdCB7k1foT5bNX__LlaWF1trO7AO4ilBt?type=png)

## 排错与优化

### 常见问题解决

1. **训练不稳定**
   - 检查是否使用了适合预训练模型的较小学习率(2e-5左右)
   - 确认回归头使用了tanh()激活函数确保输出在[-1,1]范围内
   - 尝试减小batch_size或增加梯度裁剪值

2. **内存不足错误**
   - 减小batch_size (8-16)
   - 增加FREEZE_LAYERS冻结更多Transformer层
   - 使用distilbert等较小的模型

3. **GPU/CPU切换**
   - 模型会自动根据可用设备选择
   - 对于CPU训练，建议使用较小的batch_size

### 性能优化

- 使用混合精度训练可大幅降低内存需求，提高训练速度
- 学习率预热和衰减策略已内置，可提升训练稳定性
- 使用CLS池化通常比平均池化需要的计算资源更少

## 应用场景

通过该模型，您可以分析文本的情感极性(效价)和情感强度(唤醒度)，应用于：

- 社交媒体情感监测
- 客户反馈分析
- 心理健康文本分析
- 内容推荐系统
- 跨语言情感分析

## 开发者

如有任何问题或建议，请通过以下方式联系开发团队：

- 提交GitHub Issues

## 许可

本项目遵循MIT许可证。
