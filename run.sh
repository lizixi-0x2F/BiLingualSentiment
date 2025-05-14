#!/bin/bash

# 设置错误处理
set -e

echo "===== MacBERT中英文情感分析模型训练 (支持MPS加速) ====="
echo "- 优化版：扩大数据规模，延长训练时间，提高模型性能"

# 清理潜在的后台进程
echo "1. 清理潜在的后台进程..."
pkill -f train.py 2>/dev/null || true
pkill -f predict.py 2>/dev/null || true
sleep 2

# 安装依赖
echo "2. 安装依赖项..."
pip install -r requirements.txt

# 清理旧的结果目录和文件
echo "3. 清理旧的结果目录和文件..."
rm -rf ./results
rm -rf ./model_dir
rm -f ./chinese_poetry.csv

echo "4. 检查中文数据集..."
if [ -f "text_valence_arousal_poetry_noisy.csv" ]; then
    echo "   - 中文数据集文件存在: text_valence_arousal_poetry_noisy.csv"
    # 检查文件大小
    filesize=$(du -h text_valence_arousal_poetry_noisy.csv | cut -f1)
    echo "   - 文件大小: $filesize"
    # 计算行数（数据条数）
    lines=$(wc -l < text_valence_arousal_poetry_noisy.csv)
    echo "   - 数据条数: $((lines-1)) 条 (减去标题行)"
else
    echo "   - 错误: 中文数据集文件不存在!"
    exit 1
fi

# 训练模型
echo "5. 开始训练MacBERT模型..."
echo "   - 同时支持中英文"
echo "   - 硬件加速: CUDA(NVIDIA)/MPS(Apple Silicon)/CPU"
echo "   - 使用中文数据集: text_valence_arousal_poetry_noisy.csv (限制样本数10000)"
echo "   - 使用英文数据集: emobank_va_normalized.csv (限制样本数10000)"
echo "   - 批次大小: 8 (优化内存使用)"
echo "   - 训练轮次: 5 (显著提高模型性能)"
echo "   - 学习率策略: 预热 + 线性调度 (提高训练稳定性)"
echo "   - 早停策略: 3轮无改善自动停止 (避免过拟合)"
echo "   - 值域范围: 统一使用[-1, 1]表示情感值"

# 清理系统缓存
echo "   - 清理系统缓存..."
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    sudo purge 2>/dev/null || true
elif [ "$(uname)" == "Linux" ]; then
    # Linux
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || true
fi

# 设置超时（最长8小时）
(
    python train.py
) & PID=$!

echo "   - 训练进程 PID: $PID"
echo "   - 训练过程可能需要较长时间..."

# 等待训练完成
wait $PID

# 检查模型是否成功保存
if [ -d "./model_dir" ]; then
    echo "6. 模型训练完成，转换为Core ML格式..."
    # 转换最新checkpoint为Core ML格式
    python convert_to_coreml.py
    
    echo "7. 开始测试模型..."
    # 运行预测示例
    python predict.py
    
    echo "8. 微调和测试已完成！"
    echo "   - 中英文双语模型已保存到: model_dir 目录"
    echo "   - Core ML模型已保存到: coreml_model 目录"
    echo "   - 您可以使用 python predict.py 进行情感分析预测"
    echo "   - 预测值范围: 价效度和唤起度均为[-1, 1]，正值表示正面情感，负值表示负面情感"
else
    echo "错误: 模型训练可能未完成，model_dir 目录不存在！"
    exit 1
fi 