#!/bin/bash

# 默认参数
MODEL_PATH=${1:-"checkpoints/student_large_s/best_model.pt"}
OUTPUT_PATH=${2:-""}
MAX_SEQ_LENGTH=${3:-128}

echo "开始将模型转换为ONNX格式..."
echo "模型路径: $MODEL_PATH"
echo "最大序列长度: $MAX_SEQ_LENGTH"

# 构建命令
CMD="python -m src.convert_to_onnx --model_path \"$MODEL_PATH\" --max_seq_length $MAX_SEQ_LENGTH"

# 如果提供了输出路径
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
    echo "输出路径: $OUTPUT_PATH"
else
    echo "输出路径: 自动生成"
fi

echo "执行命令: $CMD"
echo "-------------------------------------------"

# 执行转换
eval $CMD

echo "-------------------------------------------"
echo "如果转换成功，ONNX模型可以在iOS/macOS/Android等应用中使用。"
echo "要在iOS应用中使用ONNX模型，你需要:"
echo "1. 使用ONNX Runtime Mobile SDK (https://onnxruntime.ai/)"
echo "2. 将.onnx文件添加到你的项目中"
echo "3. 使用ORT API加载和运行模型"
echo "4. 处理输入/输出格式以匹配你的应用需求" 