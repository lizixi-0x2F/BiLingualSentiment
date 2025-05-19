#!/usr/bin/env pwsh

# LTC-NCP-VA 简化版本评估脚本 (PowerShell版)

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  LTC-NCP-VA 简化版评估脚本" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 设置参数
$MODEL_PATH = "results/simple_base/best_model.pt"
$TEST_DATA = "data/processed_clean/merged_val_clean.csv"
$OUTPUT_DIR = "results/simple_base/evaluation"
$BATCH_SIZE = 64

# 创建目录
New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null

Write-Host "模型路径: $MODEL_PATH" -ForegroundColor Yellow
Write-Host "测试数据: $TEST_DATA" -ForegroundColor Yellow
Write-Host "输出目录: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan

# 记录开始时间
$START_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "开始时间: $START_TIME" -ForegroundColor Green

# 运行评估脚本
Write-Host "正在评估模型..." -ForegroundColor Green
python src/evaluate_simple.py `
  --ckpt $MODEL_PATH `
  --data $TEST_DATA `
  --output $OUTPUT_DIR `
  --batch_size $BATCH_SIZE

# 记录结束时间
$END_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "结束时间: $END_TIME" -ForegroundColor Green

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  评估完成！结果保存在: $OUTPUT_DIR" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan
