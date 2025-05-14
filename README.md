# MacBERT中英文情感分析模型微调 (支持MPS加速)

这个项目使用MacBERT模型（哈工大讯飞联合实验室开发）微调一个同时支持中英文的情感分析模型，能够分析文本并输出唤起度(Arousal)和价效度(Valence)值。项目针对Apple Silicon芯片提供MPS加速支持。

## 特点

- **双语支持**：使用单一模型同时支持中英文
- **硬件加速**：支持Apple Silicon芯片的MPS加速
- **情感分析**：输出文本的价效度和唤起度

## 数据说明

项目使用合并后的数据集：
- **英文数据**：
  - `emobank_va_normalized.csv`：包含英文文本和归一化情感标签（优先使用）
  - `emobank_va.csv`：包含英文文本和原始情感标签（备用）
- **中文数据**：`text_valence_arousal_poetry_noisy.csv`，包含大量中文诗词和情感标签

数据预处理过程中会将所有值归一化到0-1范围内。

## 环境配置

安装所需依赖：

```bash
pip install -r requirements.txt
```

## 硬件加速支持

代码自动检测并使用以下加速设备：
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon: M1/M2/M3芯片)
- CPU (如果以上都不可用)

## 模型训练

训练MacBERT模型支持中英文：

```bash
python train.py
```

训练流程：
1. 加载并合并中英文数据集
2. 使用MacBERT模型（来自哈工大讯飞联合实验室）进行微调
3. 保存训练完成的模型

训练完成后，模型将保存到`model_dir`目录。

## 模型预测

使用微调后的模型进行预测：

```bash
python predict.py
```

预测脚本会自动执行中英文文本的情感分析。你也可以在自己的代码中导入预测函数：

```python
from predict import batch_predict

# 混合中英文预测
texts = ["Your English text here", "你的中文文本"]
results = batch_predict(texts)
```

## 参数说明

- **价效度(Valence)**：表示情感的正负性，值越高表示情感越积极
- **唤起度(Arousal)**：表示情感的强烈程度，值越高表示情感越强烈

## 性能优化

为了提高训练效率和减少资源占用，项目采用了以下优化措施：
- 将样本数量限制在5000条（中文和英文数据各自）
- 减小批次大小为8（原为16）
- 减少训练轮次为2（原为3）
- 使用梯度累积减少内存占用
- GPU/MPS加速支持

## 一键执行

使用提供的脚本一键完成整个流程（安装依赖、训练和测试）：

```bash
chmod +x run.sh
./run.sh
```

## 目录结构

- `train.py` - 训练脚本（双语模型）
- `predict.py` - 预测脚本（支持中英文）
- `requirements.txt` - 依赖项列表
- `run.sh` - 一键执行脚本
- `monitor.py` - 训练监控脚本
- `emobank_va.csv` - 英文训练数据集
- `emobank_va_normalized.csv` - 归一化英文训练数据集
- `text_valence_arousal_poetry_noisy.csv` - 中文诗词数据集（含大量古典诗词）
- `model_dir/` - 训练好的双语模型目录（训练后生成）
