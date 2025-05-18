#!/bin/bash
# 使用本地预训练模型运行训练脚本

# 确保脚本在错误时停止
set -e

# 检查是否有本地模型
if [ ! -d "pretrained_models/xlm-roberta-base" ] || [ ! -d "pretrained_models/distilbert-multilingual" ]; then
    echo "本地预训练模型不存在，先下载模型..."
    python download_models.py
fi

# 配置参数
MODEL_TYPE=${1:-"xlm-roberta"}  # 默认使用 XLM-RoBERTa
DEVICE=${2:-"cuda"}             # 默认使用 CUDA
BATCH_SIZE=${3:-16}             # 默认批次大小 16
EPOCHS=${4:-10}                 # 默认训练 10 轮
LR=${5:-2e-5}                   # 默认学习率 2e-5

echo "开始训练 $MODEL_TYPE 模型..."
echo "设备: $DEVICE, 批次大小: $BATCH_SIZE, 训练轮数: $EPOCHS, 学习率: $LR"

# 运行训练脚本
python train_pretrained.py \
    --model_type $MODEL_TYPE \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --monitor "r2" \
    --patience 3

echo "训练完成！"