#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}===== 将Large-S模型转换为Core ML格式 =====${NC}"

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

# 步骤2: 将ONNX转换为Core ML格式
echo -e "\n${GREEN}步骤2: 将ONNX模型转换为Core ML格式${NC}"
chmod +x onnx_to_coreml.sh
./onnx_to_coreml.sh "models/student_large_s.onnx" "models/student_large_s.mlmodel"

# 检查转换是否成功
if [ ! -f "models/student_large_s.mlmodel" ]; then
    echo -e "${RED}错误: Core ML模型转换失败${NC}"
    exit 1
fi

# 步骤3: 对比模型大小
echo -e "\n${GREEN}步骤3: 对比模型大小${NC}"
PYTORCH_SIZE=$(du -h "$LARGE_MODEL_PATH" | cut -f1)
ONNX_SIZE=$(du -h "models/student_large_s.onnx" | cut -f1)
COREML_SIZE=$(du -h "models/student_large_s.mlmodel" | cut -f1)

echo -e "PyTorch模型大小: ${BLUE}$PYTORCH_SIZE${NC}"
echo -e "ONNX模型大小: ${BLUE}$ONNX_SIZE${NC}"
echo -e "Core ML模型大小: ${BLUE}$COREML_SIZE${NC}"

echo -e "\n${GREEN}转换完成!${NC}"
echo -e "Core ML模型保存在: ${BLUE}models/student_large_s.mlmodel${NC}"
echo -e "Swift实现: ${BLUE}src/ios_helper/SentimentPredictor.swift${NC}" 