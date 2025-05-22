#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}===== 设置BiLingualSentiment项目环境 =====${NC}"

# 检查Python版本
python_version=$(python --version 2>&1 | cut -d " " -f 2)
echo -e "检测到Python版本: ${YELLOW}${python_version}${NC}"

# 检查CUDA是否可用
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}CUDA可用${NC}"
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    echo -e "CUDA版本: ${YELLOW}${cuda_version}${NC}"
else
    echo -e "${YELLOW}警告: CUDA不可用，将使用CPU模式${NC}"
fi

# 创建虚拟环境
echo -e "\n${BLUE}创建虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    python -m venv venv
    echo -e "${GREEN}虚拟环境已创建${NC}"
fi

# 激活虚拟环境
echo -e "\n${BLUE}激活虚拟环境...${NC}"
source venv/bin/activate
echo -e "${GREEN}虚拟环境已激活${NC}"

# 安装依赖
echo -e "\n${BLUE}安装依赖...${NC}"
pip install -r requirements.txt

# 创建必要的目录
echo -e "\n${BLUE}创建必要的目录...${NC}"
mkdir -p data
mkdir -p checkpoints/teacher
mkdir -p checkpoints/student_micro
mkdir -p checkpoints/student_small
mkdir -p checkpoints/student_medium
mkdir -p checkpoints/student_large_s
mkdir -p models
mkdir -p logs

echo -e "\n${GREEN}环境设置完成!${NC}"
echo -e "使用以下命令激活环境:"
echo -e "${YELLOW}source venv/bin/activate${NC}" 