#!/bin/bash

# LTC-NCP-VA 清理版训练脚本
# 使用优化配置训练模型，已移除冗余调试信息

echo "===== LTC-NCP-VA 简洁训练 ====="
echo "日期: $(date '+%Y-%m-%d %H:%M:%S')"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo -n "GPU: "
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "警告: 未检测到GPU，将使用CPU训练"
fi

# 创建必要目录
mkdir -p results/plots runs

# 显示命令
echo "执行命令: python train.py --config configs/optimized_performance.yaml --amp --epochs 1 $@"

# 启动训练 - 使用优化配置，禁用Python缓冲以实时显示日志
PYTHONUNBUFFERED=1 python train.py \
    --config configs/optimized_performance.yaml \
    --amp \
    --epochs 1 \
    "$@" 2>&1 | grep -v "shape\|dimension\|WARNING" # 过滤掉形状和维度相关的信息

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "训练完成！模型已保存"
    latest_run=$(ls -td runs/*/ 2>/dev/null | head -1)
    
    if [ -n "$latest_run" ]; then
        echo "最新运行: $latest_run"
        echo "评估命令: python evaluate.py --ckpt $latest_run/model_best.pt"
    fi
else
    echo "训练过程中发生错误"
fi

echo "==============================" 