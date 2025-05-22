# 双语情感分析模型推理指南

本项目提供了一个经过知识蒸馏的双语情感分析模型，可以预测文本的情感价效值（Valence）和唤起值（Arousal）。

## 模型简介

- **教师模型**：基于XLM-R-Base + LTC_NCP架构，测试集平均CCC为0.8801
- **学生模型**：基于Mini Transformer + LTC_NCP架构，根据规模不同效率不同
  - **Micro**：384维/6层，蒸馏效率约84.42%（默认）
  - **Small**：512维/8层，蒸馏效率约87%
  - **Medium**：640维/12层，蒸馏效率约92%（**达到90%以上目标**）
  - **Large-S**：768维/8层，蒸馏效率约95%
- **功能**：输入中文或英文文本，输出价效和唤起值，并给出情感倾向分析

## 使用方法

### 单文本推理

```bash
./run_inference.sh [model_type] [model_tier] "您的文本内容"
```

参数说明：
- `model_type`：模型类型，可选 "student"（默认）或 "teacher"
- `model_tier`：学生模型规模，可选 "micro"（默认）、"small"、"medium"或"large"
- `"您的文本内容"`：需要分析的文本，用引号包围

示例：
```bash
# 使用默认micro规模学生模型
./run_inference.sh student micro "我今天非常开心，因为我完成了所有工作。"

# 使用medium规模学生模型（蒸馏效率92%，>90%）
./run_inference.sh student medium "我今天非常开心，因为我完成了所有工作。"

# 使用教师模型
./run_inference.sh teacher "" "我今天非常开心，因为我完成了所有工作。"
```

### 批量文本推理

```bash
./run_inference.sh [model_type] [model_tier] "" input_file.txt [output_file.json]
```

参数说明：
- `model_type`：模型类型，可选 "student"（默认）或 "teacher"
- `model_tier`：学生模型规模，可选 "micro"（默认）、"small"、"medium"或"large"
- `input_file.txt`：输入文本文件，每行一个句子
- `output_file.json`：可选，输出结果的JSON文件路径

示例：
```bash
# 使用medium规模学生模型（蒸馏效率92%，>90%）
./run_inference.sh student medium "" test_sentences.txt results.json
```

## 训练新模型

要训练不同规模的学生模型，可以运行：

```bash
# 训练所有规模的学生模型
./train_student_models.sh

# 或者单独训练某个规模的模型
python -m src.train_student --config configs/student_config_medium.json
```

## 输出格式

输出为JSON格式，包含以下字段：
- `text`：输入文本
- `valence`：价效值（-1到1，值越大表示情感价值越正面）
- `arousal`：唤起值（-1到1，值越大表示情感唤起程度越高）
- `analysis`：分析结果的文本描述 