#!/bin/bash

# 双语情感分析项目完整流程脚本

echo "=========================================="
echo "      双语情感分析项目自动运行脚本"
echo "      (带液态神经网络与对抗训练)"
echo "=========================================="

# 设置学生模型档位 (micro, small, medium, large)
CONFIG_TIER=${1:-"micro"}

echo "选择的学生模型档位: $CONFIG_TIER"
echo "使用特性:"
echo "- 液态时间常数网络 (LTC)"
echo "- 神经回路机制 (NCP)"
echo "- FGM对抗训练"
echo "- R-Drop正则化"
echo "- GradNorm多任务平衡"
echo ""

# 步骤1: 训练教师模型
echo "步骤1: 训练教师模型..."
./run_train_teacher.sh
if [ $? -ne 0 ]; then
    echo "教师模型训练失败，请检查错误信息"
    exit 1
fi
echo "教师模型训练完成！"
echo ""

# 步骤2: 训练学生模型
echo "步骤2: 训练对应档位的学生模型..."
case "$CONFIG_TIER" in
    "small")
        CONFIG_PATH="configs/student_config_small.json"
        ;;
    "medium")
        CONFIG_PATH="configs/student_config_medium.json"
        ;;
    "large")
        CONFIG_PATH="configs/student_config_large_s.json"
        ;;
    *)
        # 默认使用micro档位
        CONFIG_PATH="configs/student_config.json"
        ;;
esac

./run_train_student.sh "$CONFIG_PATH"
if [ $? -ne 0 ]; then
    echo "学生模型训练失败，请检查错误信息"
    exit 1
fi
echo "学生模型训练完成！"
echo ""

# 步骤3: 运行演示程序
echo "步骤3: 启动演示界面..."
./run_demo.sh "student" "$CONFIG_TIER"

echo "=========================================="
echo "          所有步骤完成！"
echo "==========================================" 