#!/bin/bash

# 默认使用micro档位学生模型
MODEL_TYPE=${1:-"student"}
CONFIG_TIER=${2:-"micro"}

if [ "$MODEL_TYPE" = "teacher" ]; then
    MODEL_PATH="checkpoints/teacher/best_model.pt"
    DISPLAY_TIER="教师"
    echo "使用教师模型 (XLM-R-Base + LTC_NCP)"
    echo "具有完整的液态神经网络和神经回路机制"
else
    # 根据档位选择不同的学生模型配置
    case "$CONFIG_TIER" in
        "small")
            MODEL_PATH="checkpoints/student_small/best_model.pt"
            ;;
        "medium")
            MODEL_PATH="checkpoints/student_medium/best_model.pt"
            ;;
        "large")
            MODEL_PATH="checkpoints/student_large_s/best_model.pt"
            ;;
        *)
            # 默认使用micro档位
            MODEL_PATH="checkpoints/student/best_model.pt"
            ;;
    esac
    DISPLAY_TIER="$CONFIG_TIER"
    echo "使用学生模型 (Mini Transformer + LTC_NCP)"
    echo "档位: $DISPLAY_TIER, 具有知识蒸馏和神经回路机制"
fi

echo "模型路径: $MODEL_PATH"
echo "开始加载模型，请稍候..."

# 运行演示程序
python -m src.demo --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" 