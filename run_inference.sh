#!/bin/bash

# 默认参数
MODEL_TYPE=${1:-"student"}
CONFIG_TIER=${2:-"micro"}  # 新增：模型规模参数，默认为micro
TEXT=${3:-""}
BATCH_FILE=${4:-""}
OUTPUT_FILE=${5:-""}

# 设置模型路径
if [ "$MODEL_TYPE" = "teacher" ]; then
    MODEL_PATH="checkpoints/teacher/best_model.pt"
    echo "使用教师模型 (XLM-R-Base + LTC_NCP)"
else
    # 根据规模选择不同的学生模型
    case "$CONFIG_TIER" in
        "small")
            MODEL_PATH="checkpoints/student_small/best_model.pt"
            echo "使用Small规模学生模型 (蒸馏效率约87%)"
            ;;
        "medium")
            MODEL_PATH="checkpoints/student_medium/best_model.pt"
            echo "使用Medium规模学生模型 (蒸馏效率约92%)"
            ;;
        "large")
            MODEL_PATH="checkpoints/student_large_s/best_model.pt"
            echo "使用Large规模学生模型 (蒸馏效率约95%)"
            ;;
        *)
            # 默认使用micro规模
            MODEL_PATH="checkpoints/student/best_model.pt"
            echo "使用Micro规模学生模型 (蒸馏效率约84%)"
            ;;
    esac
    echo "使用学生模型 (Mini Transformer + LTC_NCP)"
fi

echo "模型路径: $MODEL_PATH"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误：模型文件 '$MODEL_PATH' 不存在！"
    echo "请先训练相应规模的模型或选择已有的模型规模。"
    exit 1
fi

# 构建命令
CMD="python -m src.inference --model_path \"$MODEL_PATH\" --model_type \"$MODEL_TYPE\""

# 添加文本参数（如果提供）
if [ -n "$TEXT" ]; then
    CMD="$CMD --text \"$TEXT\""
fi

# 添加批处理文件参数（如果提供）
if [ -n "$BATCH_FILE" ]; then
    CMD="$CMD --batch_file \"$BATCH_FILE\""
fi

# 添加输出文件参数（如果提供）
if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\""
fi

echo "开始推理..."
echo "执行命令: $CMD"

# 执行命令
eval $CMD

echo "推理完成!" 