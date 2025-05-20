#!/usr/bin/env pwsh

# LTC-NCP-VA 简化版训练脚本 (PowerShell版)
# 移除了高级功能：FGM对抗训练、软标签、多任务学习头

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  LTC-NCP-VA 简化版训练启动脚本" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "模型: 基础版情感价效度分析模型" -ForegroundColor Yellow
Write-Host "配置: configs/simple_base.yaml" -ForegroundColor Yellow
Write-Host "特性: 基础版(无高级功能)" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan

# 设置运行参数
$EPOCHS = 30
$BATCH_SIZE = 32
$LEARNING_RATE = 0.001
$OUTPUT_DIR = "results/simple_base"

# 创建目录
New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null

# 记录开始时间
$START_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "开始时间: $START_TIME" -ForegroundColor Green
"开始时间: $START_TIME" | Out-File -FilePath "$OUTPUT_DIR/training_log.txt" -Append

# 运行训练脚本
Write-Host "正在启动训练..." -ForegroundColor Green
python src/train_simple.py `
  --config configs/simple_base.yaml `
  --epochs $EPOCHS `
  --batch_size $BATCH_SIZE `
  --lr $LEARNING_RATE `
  --save_dir $OUTPUT_DIR `
  | Tee-Object -FilePath "$OUTPUT_DIR/training_log.txt" -Append

# 记录结束时间
$END_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "结束时间: $END_TIME" -ForegroundColor Green
"结束时间: $END_TIME" | Out-File -FilePath "$OUTPUT_DIR/training_log.txt" -Append

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  训练完成！结果保存在: $OUTPUT_DIR" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan
