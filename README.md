# BiLingualSentiment - 双语情感分析系统

基于PyTorch和LTC神经电路的双语（中英文）情感分析系统，支持预测文本的效价(Valence)和唤醒度(Arousal)值。项目采用教师-学生知识蒸馏架构，实现轻量级推理模型。

## 项目架构

### 教师模型
- 基础模型：XLM-R-Base
- 神经电路：LTC_NCP (Liquid Time-Constant Neural Circuit Policies)
- 目标性能：CCC ≥ 70%

### 学生模型
- 基础模型：Mini Transformer
- 神经电路：LTC_NCP
- 目标性能：相对教师模型的CCC ≥ 95%
- 提供多种尺寸：Micro、Small、Medium、Large-S

## 功能特点

- 跨语言迁移：利用XLM-R的多语言能力，在中英文语料上均具有良好表现
- 知识蒸馏：将大型预训练模型的知识迁移到轻量级模型
- 移动端部署：支持导出为ONNX和Core ML格式，便于在移动设备上部署
- 高效推理：学生模型体积小、速度快，适合资源受限场景

## 训练技术

- R-Drop正则化
- GradNorm梯度平衡
- FGM对抗训练
- 知识蒸馏

## 目录结构

```
BiLingualSentiment/
├── src/                # 源代码
│   ├── data/           # 数据处理模块
│   ├── models/         # 模型定义
│   ├── training/       # 训练相关代码
│   └── utils/          # 工具函数
├── scripts/            # 实用脚本
├── config/             # 配置文件
├── checkpoints/        # 模型检查点（不包含在Git中）
└── models/             # 导出的模型（不包含在Git中）
```

## 安装与使用

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/BiLingualSentiment.git
cd BiLingualSentiment

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 训练教师模型
python -m src.train_teacher

# 训练学生模型
python -m src.train_student
```

### 推理

```bash
# 使用模型进行推理
python -m src.inference --text "这是一个测试文本" --model_path checkpoints/student_small/best_model.pt
```

### 导出模型

```bash
# 导出为ONNX格式
python -m src.pytorch_to_onnx --model_path checkpoints/student_small/best_model.pt

# 导出为Core ML格式 (仅macOS)
python mac_convert.py
```

## 模型尺寸选择

| 模型 | 隐藏层维度 | 层数 | 注意力头数 | 参数量 | 推荐场景 |
|------|------------|------|------------|--------|----------|
| Micro | 384 | 4 | 6 | ~5M | 低端移动设备 |
| Small | 512 | 6 | 8 | ~10M | 中端移动设备 |
| Medium | 640 | 8 | 10 | ~15M | 高端移动设备 |
| Large-S | 768 | 10 | 12 | ~25M | 服务器/桌面端 |

## 许可证

[MIT License](LICENSE)