#!/bin/bash

# 默认参数
ONNX_PATH=${1:-"models/student_large_s.onnx"}
OUTPUT_PATH=${2:-""}
IOS_VERSION=${3:-"15"}

echo "开始将ONNX模型转换为Core ML格式..."
echo "ONNX模型路径: $ONNX_PATH"
echo "iOS最低版本: $IOS_VERSION"

# 检查ONNX文件是否存在
if [ ! -f "$ONNX_PATH" ]; then
    echo "错误: ONNX模型文件不存在 ($ONNX_PATH)"
    echo "提示: 请先使用convert_to_onnx.sh转换模型为ONNX格式"
    exit 1
fi

# 构建命令
CMD="python -m src.onnx_to_coreml --onnx_path \"$ONNX_PATH\" --ios_version $IOS_VERSION"

# 如果提供了输出路径
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
    echo "输出路径: $OUTPUT_PATH"
else
    # 自动生成输出路径
    MODEL_DIR=$(dirname "$ONNX_PATH")
    MODEL_NAME=$(basename "$ONNX_PATH" .onnx)
    OUTPUT_PATH="$MODEL_DIR/$MODEL_NAME.mlmodel"
    echo "输出路径: $OUTPUT_PATH (自动生成)"
fi

echo "执行命令: $CMD"
echo "-------------------------------------------"

# 执行转换
eval $CMD

# 检查转换是否成功
if [ -f "$OUTPUT_PATH" ]; then
    echo "-------------------------------------------"
    echo "转换成功! Core ML模型已保存到: $OUTPUT_PATH"
    echo "Core ML模型大小: $(du -h "$OUTPUT_PATH" | cut -f1)"
    echo ""
    echo "要在iOS应用中使用模型，你需要:"
    echo "1. 将.mlmodel文件添加到Xcode项目中"
    echo "2. 使用CoreML框架加载和运行模型"
    echo "3. 使用src/ios_helper/SentimentPredictor.swift辅助类处理模型输入/输出"
else
    echo "-------------------------------------------"
    echo "转换失败，未找到输出文件: $OUTPUT_PATH"
    echo "请检查上面的错误信息"
fi 