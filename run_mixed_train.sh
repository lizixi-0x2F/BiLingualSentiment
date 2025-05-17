#!/bin/bash

# 中英文混合情感分析模型训练脚本
# 默认使用大批次(128)配置

# 默认参数
DEVICE="cuda"
BATCH_SIZE=128
EPOCHS=20
MONITOR="r2"
PATIENCE=5
LR=0.0003
LAMBDA=0.7  # 调整混合权重λ从0.3到0.7，增加回归损失占比

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --monitor)
      MONITOR="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --lambda)
      LAMBDA="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 设置输出目录
if [ -z "$OUTPUT_DIR" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  OUTPUT_DIR="outputs/mixed_model_${TIMESTAMP}"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练脚本
echo "开始训练情感分析模型 (中英文混合)..."
echo "设备: $DEVICE | 批次大小: $BATCH_SIZE | 轮次: $EPOCHS | 监控指标: $MONITOR | 学习率: $LR"

python train_mixed.py \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --monitor "$MONITOR" \
  --patience "$PATIENCE" \
  --lr "$LR" \
  --lambda "$LAMBDA" \
  --output_dir "$OUTPUT_DIR"

echo "训练完成，结果保存在: $OUTPUT_DIR" 