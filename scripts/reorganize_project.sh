#!/bin/bash

# 项目结构重组脚本
# 作用：清理项目结构，提高代码组织性和可维护性

echo "开始清理和重组项目结构..."

# 1. 创建清晰的目录结构
mkdir -p src/{core,data,utils,models}  # 核心源代码目录
mkdir -p scripts                        # 脚本目录
mkdir -p experiments                    # 实验记录目录
mkdir -p notebooks                      # Jupyter笔记本目录
mkdir -p checkpoints/backup             # 模型检查点备份目录
mkdir -p logs                           # 日志统一存放
mkdir -p tests                          # 测试文件目录

# 2. 移动核心模型代码到src目录
echo "移动模型核心代码到src目录..."
cp -r ltc_ncp/* src/core/
# 在新位置创建正确的__init__.py
cat > src/core/__init__.py << EOF
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-RNN模型核心模块
"""

from .cells import LTCCell
from .wiring import NCPWiring
from .model import LTC_NCP_RNN
from .adversarial import FGM
from .boundary_weights import BoundarySampleWeighter
from .pos_features import extract_combined_pos_features, preprocess_batch_texts
EOF

# 3. 移动数据处理代码
echo "移动数据处理相关代码..."
cp clean_datasets_final.py src/data/
cp merge_datasets.py src/data/

# 4. 移动工具类代码
echo "移动工具类代码..."
cp calibrate.py src/utils/
cp mixed_precision_fix.py src/utils/

# 5. 移动训练和评估脚本
echo "移动训练和评估相关脚本..."
cp train.py src/
cp evaluate.py src/
cp emotion_predict.py src/

# 6. 移动运行脚本到scripts目录
echo "整理脚本文件..."
cp run_*.sh scripts/
cp *.sh scripts/ 2>/dev/null || :

# 7. 整理日志文件
echo "整理日志文件..."
mkdir -p logs/training
cp *.log logs/training/ 2>/dev/null || :
cp training_*.txt logs/training/ 2>/dev/null || :

# 8. 创建环境和依赖管理
echo "整理环境依赖..."
cp environment.yml ./

# 9. 创建项目README
cat > README.md << EOF
# LTC-NCP-VA 情感价效度回归模型

基于液态时间常数网络与神经电路策略的文本情感分析模型，用于预测文本的情感价效度(Valence-Arousal)值。

## 项目结构

- **src/**: 核心源代码
  - **core/**: 模型核心实现
  - **data/**: 数据处理模块
  - **utils/**: 工具函数
  - **models/**: 训练好的模型
- **scripts/**: 运行脚本
- **configs/**: 配置文件
- **data/**: 数据集
- **checkpoints/**: 模型检查点
- **logs/**: 日志文件
- **results/**: 实验结果和可视化
- **docs/**: 文档
- **tests/**: 测试文件
- **notebooks/**: Jupyter笔记本

## 功能特性

- 支持中英双语文本情感价效度预测
- 结合LTC网络与NCP连接策略的神经网络
- 支持FGM对抗训练增强模型鲁棒性
- 支持边界样本加权训练增强边界预测能力
- 支持四象限分类辅助任务与VA回归联合训练
- 支持词性特征增强Valence判断

## 快速开始

1. 安装依赖: \`conda env create -f environment.yml\`
2. 激活环境: \`conda activate ltc-ncp-env\`
3. 训练模型: \`python src/train.py --config configs/valence_enhanced.yaml\`
4. 预测情感: \`python src/emotion_predict.py --text "这是一个测试文本" --model checkpoints/model.pt\`

## 引用

如果您使用了本项目的代码或思想，请引用：

\`\`\`
@article{ltc-ncp-va-model,
  title={LTC-NCP-VA: A Liquid Time Constant Neural Circuit Policy for Text Emotion Regression},
  author={Your Name},
  year={2023}
}
\`\`\`
EOF

# 10. 创建.gitignore文件
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 环境
.env
.venv
env/
venv/
ENV/

# 日志
logs/

# 本地配置
.idea/
.vscode/
*.swp
*.swo

# 模型检查点 (太大)
checkpoints/*
!checkpoints/.gitkeep

# 实验结果数据
results/*
!results/.gitkeep

# 数据集 (根据自己情况调整)
data/processed/*
!data/processed/.gitkeep

# 系统文件
.DS_Store
Thumbs.db
EOF

# 创建需要的空文件夹占位文件
touch checkpoints/.gitkeep
touch results/.gitkeep
touch data/processed/.gitkeep
touch src/models/.gitkeep

echo "项目结构重组完成，原始文件已保留。"
echo "请检查新的目录结构，确认无误后可以删除冗余文件。" 