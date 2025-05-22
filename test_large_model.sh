#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}===== 双语情感分析模型转换与测试 =====${NC}"

# 检查large模型是否存在
LARGE_MODEL_PATH="checkpoints/student_large_s/best_model.pt"
if [ ! -f "$LARGE_MODEL_PATH" ]; then
    echo -e "${RED}错误: Large模型文件不存在 ($LARGE_MODEL_PATH)${NC}"
    echo -e "${YELLOW}提示: 请先确保模型训练已完成${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p models

# 步骤1: 转换为ONNX格式
echo -e "\n${GREEN}步骤1: 将Large模型转换为ONNX格式${NC}"
./convert_to_onnx.sh "$LARGE_MODEL_PATH" "models/student_large_s.onnx"

# 检查转换是否成功
if [ ! -f "models/student_large_s.onnx" ]; then
    echo -e "${RED}错误: ONNX模型转换失败${NC}"
    exit 1
fi

# 步骤2: 对比模型大小
echo -e "\n${GREEN}步骤2: 对比模型大小${NC}"
PYTORCH_SIZE=$(du -h "$LARGE_MODEL_PATH" | cut -f1)
ONNX_SIZE=$(du -h "models/student_large_s.onnx" | cut -f1)

echo -e "PyTorch模型大小: ${BLUE}$PYTORCH_SIZE${NC}"
echo -e "ONNX模型大小: ${BLUE}$ONNX_SIZE${NC}"

# 步骤3: 测试推理
echo -e "\n${GREEN}步骤3: 测试模型推理${NC}"
echo -e "${YELLOW}使用原始模型进行推理:${NC}"
python -c "
from src.inference import run_inference
result = run_inference('$LARGE_MODEL_PATH', '这是一个非常感人的故事，让我深受感动')
print(f'价效值: {result[\"valence\"]:.4f}, 唤起值: {result[\"arousal\"]:.4f}')

result = run_inference('$LARGE_MODEL_PATH', 'This is a very touching story that deeply moved me')
print(f'价效值: {result[\"valence\"]:.4f}, 唤起值: {result[\"arousal\"]:.4f}')
"

echo -e "\n${GREEN}模型转换完成!${NC}"
echo -e "ONNX模型保存在: ${BLUE}models/student_large_s.onnx${NC}"
echo -e "iOS集成指南: ${BLUE}src/ios_helper/README.md${NC}"
echo -e "Swift实现: ${BLUE}src/ios_helper/SentimentPredictorONNX.swift${NC}" 