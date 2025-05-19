#!/bin/bash

# LTC-NCP-VA 简化版训练脚本
# 移除了高级功能：FGM对抗训练、软标签、多任务学习头

echo "====================================================="
echo "  LTC-NCP-VA 简化版训练启动脚本"
echo "====================================================="
echo "模型: 基础版情感价效度分析模型"
echo "配置: configs/simple_base.yaml"
echo "特性: 基础版(无高级功能)"
echo "-----------------------------------------------------"

# 设置运行参数
EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.001
OUTPUT_DIR="results/simple_base"

# 创建目录
mkdir -p $OUTPUT_DIR

# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "开始时间: $START_TIME" | tee -a "$OUTPUT_DIR/training_log.txt"

# 运行训练脚本
echo "正在启动训练..."
python src/train_simple.py \
  --config configs/simple_base.yaml \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --save_dir "$OUTPUT_DIR" \
  | tee -a "$OUTPUT_DIR/training_log.txt"

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "结束时间: $END_TIME" | tee -a "$OUTPUT_DIR/training_log.txt"

echo "====================================================="
echo "  训练完成！结果保存在: $OUTPUT_DIR"
echo "====================================================="
