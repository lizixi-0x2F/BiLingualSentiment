#!/usr/bin/env pwsh

# 融合门控测试脚本

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  测试融合门控实现" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 记录开始时间
$START_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "开始时间: $START_TIME" -ForegroundColor Green

# 运行测试脚本
Write-Host "运行融合门控测试..." -ForegroundColor Green
python tests/test_fusion_gate.py

# 记录结束时间
$END_TIME = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "结束时间: $END_TIME" -ForegroundColor Green

Write-Host "=====================================================" -ForegroundColor Cyan
