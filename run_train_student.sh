#!/bin/bash

# 默认使用micro档位
CONFIG=${1:-"configs/student_config.json"}

# 训练学生模型 (Mini Transformer + LTC_NCP)
# 使用知识蒸馏和对抗训练
# - FGM对抗训练增强鲁棒性
# - 来自教师模型的软目标和硬目标损失
# - 增强的LTC和神经回路机制

echo "启动学生模型训练 (使用配置: $CONFIG)..."
echo "包含知识蒸馏、对抗训练和神经回路机制"

# 方式一：以模块方式运行（从项目根目录调用）
python -m src.train_student --config $CONFIG 

# 如果上面的方式出错，可以尝试下面的方式：
# 方式二：直接切换到src目录运行（更改当前工作目录）
# cd src && python train_student.py --config ../$CONFIG && cd .. 