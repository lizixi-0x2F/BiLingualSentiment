#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}===== 准备macOS转换包 =====${NC}"

# 检查large模型是否存在
LARGE_MODEL_PATH="checkpoints/student_large_s/best_model.pt"
if [ ! -f "$LARGE_MODEL_PATH" ]; then
    echo -e "${RED}错误: Large模型文件不存在 ($LARGE_MODEL_PATH)${NC}"
    echo -e "${YELLOW}提示: 请先确保模型训练已完成${NC}"
    exit 1
fi

# 创建传输包目录
TRANSFER_DIR="mac_transfer_package"
mkdir -p "$TRANSFER_DIR"
mkdir -p "$TRANSFER_DIR/src/models"
mkdir -p "$TRANSFER_DIR/checkpoints/student_large_s"

echo -e "\n${GREEN}正在准备文件...${NC}"

# 复制必要的文件
cp mac_convert.py "$TRANSFER_DIR/"
cp MACOS_CONVERSION.md "$TRANSFER_DIR/README.md"
cp src/models/student_model.py "$TRANSFER_DIR/src/models/"
cp "$LARGE_MODEL_PATH" "$TRANSFER_DIR/checkpoints/student_large_s/"

# 创建目录
mkdir -p "$TRANSFER_DIR/models"

# 创建一个便捷脚本
cat > "$TRANSFER_DIR/convert.sh" << 'EOF'
#!/bin/bash
echo "安装依赖..."
pip install torch torchvision coremltools transformers

echo "开始转换模型..."
python mac_convert.py

echo "转换完成! 模型保存在models/目录中"
EOF

# 使脚本可执行
chmod +x "$TRANSFER_DIR/convert.sh"

# 创建压缩包
echo -e "\n${GREEN}创建压缩包...${NC}"
tar -czf "mac_transfer_package.tar.gz" "$TRANSFER_DIR"

# 计算文件大小
PACKAGE_SIZE=$(du -h "mac_transfer_package.tar.gz" | cut -f1)

echo -e "\n${GREEN}传输包准备完成!${NC}"
echo -e "压缩包: ${YELLOW}mac_transfer_package.tar.gz${NC} (${BLUE}$PACKAGE_SIZE${NC})"
echo -e "使用以下命令将压缩包传输到Mac:"
echo -e "${YELLOW}scp mac_transfer_package.tar.gz username@mac_ip:~/${NC}"
echo -e "然后在Mac上解压并运行转换脚本:"
echo -e "${YELLOW}tar -xzf mac_transfer_package.tar.gz${NC}"
echo -e "${YELLOW}cd mac_transfer_package${NC}"
echo -e "${YELLOW}./convert.sh${NC}"
echo -e "转换完成后，使用以下命令将转换后的模型传回Linux:"
echo -e "${YELLOW}scp models/student_large_s.mlmodel username@linux_ip:~/BiLingualSentiment/models/${NC}" 