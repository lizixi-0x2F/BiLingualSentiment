#!/bin/bash
# 运行基于本地预训练模型的微调训练

# 确保当前工作目录是项目根目录
cd "$(dirname "$0")/.." || exit

# ANSI颜色代码
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始基于本地预训练模型的微调训练${NC}"

# 检查本地预训练模型是否存在
if [ ! -d "pretrained_models/distilbert-multilingual" ] || [ ! -d "pretrained_models/xlm-roberta-base" ]; then
  echo -e "${YELLOW}警告: 本地预训练模型不存在${NC}"
  echo -e "请先运行 python download_models.py 下载模型"
  exit 1
fi

# 确保输出目录存在
mkdir -p outputs

# 运行训练脚本 - DistilBERT
echo -e "${BLUE}开始训练 DistilBERT 模型...${NC}"
python train_pretrained.py \
  --model_type distilbert \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-5 \
  --monitor r2 \
  --patience 3 \
  --output_dir outputs/pretrained_distilbert_local

# 运行训练脚本 - XLM-RoBERTa
echo -e "${BLUE}开始训练 XLM-RoBERTa 模型...${NC}"
python train_pretrained.py \
  --model_type xlm-roberta \
  --batch_size 32 \
  --epochs 10 \
  --lr 2e-5 \
  --monitor r2 \
  --patience 3 \
  --output_dir outputs/pretrained_xlm_roberta_local

echo -e "${GREEN}训练完成!${NC}"
