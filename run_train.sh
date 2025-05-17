#!/bin/bash

# 训练多语言DistilBERT模型的脚本

# 默认参数
DEVICE="cuda"
BATCH_SIZE=32
EPOCHS=10
MONITOR="r2"
PATIENCE=3
LR=2e-5
WEIGHT_DECAY=0.01
MODEL_TYPE="distilbert"  # 可选: distilbert, xlm-roberta

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/pretrained_${MODEL_TYPE}_${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 输出配置信息
echo "启动多语言预训练模型微调 (DistilBERT)..."
echo "========================================"
echo "设备: $DEVICE"
echo "批次大小: $BATCH_SIZE"
echo "最大训练轮次: $EPOCHS"
echo "监控指标: $MONITOR"
echo "早停耐心值: $PATIENCE"
echo "学习率: $LR"
echo "权重衰减: $WEIGHT_DECAY"
echo "模型类型: $MODEL_TYPE"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 在screen会话中运行训练
echo "在screen会话中启动训练..."
SCREEN_NAME="pretrained_${MODEL_TYPE}_${TIMESTAMP}"

# 使用screen创建新会话并执行命令
screen -dmS "$SCREEN_NAME" bash -c "python train.py \
  --device \"$DEVICE\" \
  --batch_size \"$BATCH_SIZE\" \
  --epochs \"$EPOCHS\" \
  --monitor \"$MONITOR\" \
  --patience \"$PATIENCE\" \
  --lr \"$LR\" \
  --weight_decay \"$WEIGHT_DECAY\" \
  --model_type \"$MODEL_TYPE\" \
  --output_dir \"$OUTPUT_DIR\" 2>&1 | tee \"$OUTPUT_DIR/training.log\""

echo "训练已在screen会话中启动: $SCREEN_NAME"
echo "使用 'screen -r $SCREEN_NAME' 查看训练进度"
