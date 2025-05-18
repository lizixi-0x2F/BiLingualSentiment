#!/bin/bash

# 情感分析模型测试脚本
# 此脚本用于测试情感分析模型并生成可视化结果

# 默认参数
MODEL_PATH=""
MODEL_TYPE="distilbert"
INPUT_FILE="example_texts.txt"
OUTPUT_FILE="predictions.csv"
DEVICE="cuda"
SINGLE_TEXT=0
TEXT="这是一个测试文本，我感到非常开心！"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --input_file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --single_text)
      SINGLE_TEXT=1
      shift
      ;;
    --text)
      TEXT="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查是否提供了模型路径
if [ -z "$MODEL_PATH" ]; then
  # 寻找最新的模型目录
  LATEST_MODEL_DIR=$(ls -td outputs/pretrained_* | head -1)
  
  if [ -n "$LATEST_MODEL_DIR" ]; then
    MODEL_PATH="${LATEST_MODEL_DIR}/best_model.pth"
    echo "找到最新模型: $MODEL_PATH"
  else
    echo "错误: 未找到模型路径。请使用 --model_path 参数指定模型路径。"
    exit 1
  fi
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
  echo "错误: 模型文件不存在: $MODEL_PATH"
  exit 1
fi

# 检查设备
if [ "$DEVICE" = "cuda" ]; then
  CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
  if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "警告: CUDA 不可用，切换到 CPU"
    DEVICE="cpu"
  fi
fi

echo "======================================"
echo "情感分析模型测试"
echo "======================================"
echo "模型路径: $MODEL_PATH"
echo "模型类型: $MODEL_TYPE"
echo "设备: $DEVICE"

if [ $SINGLE_TEXT -eq 1 ]; then
  echo "运行单文本推理..."
  
  COMMAND="python src/inference.py \
    --model_path \"$MODEL_PATH\" \
    --model_type $MODEL_TYPE \
    --device $DEVICE \
    --text \"$TEXT\" \
    --visualize"
  
  echo "执行命令: $COMMAND"
  eval $COMMAND
else
  echo "运行批量文本推理..."
  echo "输入文件: $INPUT_FILE"
  echo "输出文件: $OUTPUT_FILE"
  
  COMMAND="python src/inference.py \
    --model_path \"$MODEL_PATH\" \
    --model_type $MODEL_TYPE \
    --device $DEVICE \
    --input_file \"$INPUT_FILE\" \
    --output_file \"$OUTPUT_FILE\" \
    --visualize"
  
  echo "执行命令: $COMMAND"
  eval $COMMAND
fi

echo "======================================"
echo "完成!"
