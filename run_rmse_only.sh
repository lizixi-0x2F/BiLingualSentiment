#!/bin/bash

# 使用RMSE作为唯一损失函数的训练脚本
# 基于run_r2_best_model.sh修改

# 默认参数
DEVICE="cuda"
BATCH_SIZE=128
EPOCHS=20
MONITOR="r2"  # 使用R2作为选择指标
PATIENCE=5
LR=0.001
LAMBDA=0.7

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/rmse_only_${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 输出配置信息
echo "启动情感分析模型训练 (仅使用RMSE损失函数)..."
echo "========================================"
echo "设备: $DEVICE"
echo "批次大小: $BATCH_SIZE"
echo "最大训练轮次: $EPOCHS"
echo "监控指标: $MONITOR"
echo "早停耐心值: $PATIENCE"
echo "学习率: $LR"
echo "损失混合权重λ: $LAMBDA"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 在screen会话中运行训练
echo "在screen会话中启动训练..."
SCREEN_NAME="rmse_only_${TIMESTAMP}"

# 使用screen创建新会话并执行命令
screen -dmS "$SCREEN_NAME" bash -c "python train_mixed.py \
  --device \"$DEVICE\" \
  --batch_size \"$BATCH_SIZE\" \
  --epochs \"$EPOCHS\" \
  --monitor \"$MONITOR\" \
  --patience \"$PATIENCE\" \
  --lr \"$LR\" \
  --lambda \"$LAMBDA\" \
  --output_dir \"$OUTPUT_DIR\" 2>&1 | tee \"$OUTPUT_DIR/training.log\""

echo "训练已在screen会话中启动: $SCREEN_NAME"
echo "可以使用 'screen -r $SCREEN_NAME' 查看训练进度"
echo "训练结果将保存在: $OUTPUT_DIR" 