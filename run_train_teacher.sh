#!/bin/bash

# 训练教师模型 (XLM-R-Base + LTC_NCP)
# 使用完整的液态神经网络、神经回路机制和多种正则化技术
# - FGM对抗训练
# - R-Drop正则化
# - GradNorm动态损失平衡

echo "启动教师模型训练，包含LTC、NCP和对抗训练..."
python src/train_teacher.py --config configs/teacher_config.json 