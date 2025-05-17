#!/bin/bash

# 设置环境变量
export PYTHONPATH=.

# 创建输出目录
mkdir -p results/valence_enhanced_5m
mkdir -p runs/valence_enhanced_5m

# 训练模型名称
MODEL_NAME="valence_fgm_boundary_multitask"

# 创建screen会话并运行训练
screen -dmS ${MODEL_NAME} bash -c "python train.py --config configs/valence_enhanced.yaml --epochs 50 | tee training_${MODEL_NAME}.log"

echo "训练已在后台screen会话中启动，会话名: ${MODEL_NAME}"
echo "可以使用 'screen -r ${MODEL_NAME}' 查看训练进度"
echo "使用 Ctrl+A 然后 Ctrl+D 可以再次将会话分离到后台"
echo "训练日志也会保存到 training_${MODEL_NAME}.log 文件"
echo "特性: FGM对抗训练 + 边界样本权重 + VA多任务" 