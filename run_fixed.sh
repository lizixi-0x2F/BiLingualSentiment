#!/bin/bash

echo "===== LTC-NCP-VA 基于RMSE的优化训练 ====="
echo "日期: $(date +%Y-%m-%d\ %H:%M:%S)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 设置环境变量以精简输出
# 设置为WARNING级别，只显示警告和错误
export LOG_LEVEL=WARNING
# 禁用Python繁冗警告
export PYTHONWARNINGS=ignore
# 确保正确编码
export PYTHONIOENCODING=utf-8
# 禁用PyTorch警告
export TORCH_WARNINGS=0 
# 禁用CUDA启动提示信息
export CUDA_LAUNCH_BLOCKING=0
# 禁用TensorFlow日志
export TF_CPP_MIN_LOG_LEVEL=3
# 保留进度条
export TQDM_DISABLE=0

# 运行训练脚本，使用优化配置
echo "优化目标: 最小化RMSE，使用纯MSE损失函数"
echo "日志级别: $LOG_LEVEL (1=详细日志 / 0=精简日志)"

# 使用静默模式运行训练，仅显示关键信息
python -W ignore train.py --config configs/optimized_performance.yaml --amp 2>&1 | grep -v "DEBUG\|tensor("

echo "训练完成！模型已保存"
echo "最新运行: $(ls -td runs/*/ | head -1)"
echo "评估命令: python evaluate.py --ckpt $(ls -td runs/*/ | head -1)/model_best.pt"
echo "=============================="
