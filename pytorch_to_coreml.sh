#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 默认参数
MODEL_PATH=${1:-"checkpoints/student_large_s/best_model.pt"}
OUTPUT_PATH=${2:-""}
MAX_SEQ_LENGTH=${3:-128}
IOS_VERSION=${4:-"15"}

echo -e "${BLUE}开始将PyTorch模型直接转换为Core ML格式...${NC}"
echo -e "模型路径: ${YELLOW}$MODEL_PATH${NC}"
echo -e "最大序列长度: ${YELLOW}$MAX_SEQ_LENGTH${NC}"
echo -e "iOS最低版本: ${YELLOW}$IOS_VERSION${NC}"

# 检查PyTorch文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}错误: PyTorch模型文件不存在 ($MODEL_PATH)${NC}"
    exit 1
fi

# 构建命令
CMD="python -m src.pytorch_to_coreml --model_path \"$MODEL_PATH\" --max_seq_length $MAX_SEQ_LENGTH --ios_version $IOS_VERSION"

# 如果提供了输出路径
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
    echo -e "输出路径: ${YELLOW}$OUTPUT_PATH${NC}"
else
    # 自动生成输出路径
    MODEL_DIR=$(dirname "$MODEL_PATH")
    MODEL_NAME=$(basename "$MODEL_PATH" .pt)
    OUTPUT_PATH="$MODEL_DIR/$MODEL_NAME.mlmodel"
    echo -e "输出路径: ${YELLOW}$OUTPUT_PATH${NC} (自动生成)"
fi

echo -e "\n${GREEN}执行命令: $CMD${NC}"
echo -e "${BLUE}-------------------------------------------${NC}"

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_PATH")

# 执行转换
eval $CMD

# 检查转换是否成功
if [ -f "$OUTPUT_PATH" ]; then
    echo -e "${BLUE}-------------------------------------------${NC}"
    echo -e "${GREEN}转换成功!${NC} Core ML模型已保存到: ${YELLOW}$OUTPUT_PATH${NC}"
    echo -e "Core ML模型大小: ${BLUE}$(du -h "$OUTPUT_PATH" | cut -f1)${NC}"
    echo -e ""
    echo -e "${GREEN}在iOS应用中使用模型:${NC}"
    echo -e "1. 将.mlmodel文件添加到Xcode项目中"
    echo -e "2. 使用CoreML框架加载和运行模型"
    echo -e "3. 使用src/ios_helper/SentimentPredictor.swift辅助类处理模型输入/输出"
else
    echo -e "${BLUE}-------------------------------------------${NC}"
    echo -e "${RED}转换失败，未找到输出文件: $OUTPUT_PATH${NC}"
    echo -e "请检查上面的错误信息"
fi 