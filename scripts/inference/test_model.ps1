# 情感分析模型测试脚本
# 此脚本用于测试情感分析模型并生成可视化结果

# 参数设置
param(
    [string]$ModelPath,
    [string]$ModelType = "distilbert",
    [string]$InputFile = "example_texts.txt",
    [string]$OutputFile = "predictions.csv",
    [string]$Device = "cuda",
    [switch]$SingleText,
    [string]$Text = "这是一个测试文本，我感到非常开心！"
)

# 检查是否提供了模型路径
if (-not $ModelPath) {
    # 寻找最新的模型目录
    $LatestModelDir = Get-ChildItem -Path "outputs" -Directory | Where-Object { $_.Name -like "pretrained_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($LatestModelDir) {
        $ModelPath = Join-Path -Path $LatestModelDir.FullName -ChildPath "best_model.pth"
        Write-Host "找到最新模型: $ModelPath"
    } else {
        Write-Host "错误: 未找到模型路径。请使用 -ModelPath 参数指定模型路径。" -ForegroundColor Red
        exit 1
    }
}

# 检查模型文件是否存在
if (-not (Test-Path $ModelPath)) {
    Write-Host "错误: 模型文件不存在: $ModelPath" -ForegroundColor Red
    exit 1
}

# 检查设备
if ($Device -eq "cuda" -and -not $(python -c "import torch; print(torch.cuda.is_available())").Trim() -eq "True") {
    Write-Host "警告: CUDA 不可用，切换到 CPU" -ForegroundColor Yellow
    $Device = "cpu"
}

Write-Host "======================================"
Write-Host "情感分析模型测试" -ForegroundColor Cyan
Write-Host "======================================"
Write-Host "模型路径: $ModelPath"
Write-Host "模型类型: $ModelType"
Write-Host "设备: $Device"

if ($SingleText) {
    Write-Host "运行单文本推理..." -ForegroundColor Green
    
    $Command = "python src/inference.py " + `
              "--model_path `"$ModelPath`" " + `
              "--model_type $ModelType " + `
              "--device $Device " + `
              "--text `"$Text`" " + `
              "--visualize"
    
    Write-Host "执行命令: $Command"
    Invoke-Expression $Command
} else {
    Write-Host "运行批量文本推理..." -ForegroundColor Green
    Write-Host "输入文件: $InputFile"
    Write-Host "输出文件: $OutputFile"
    
    $Command = "python src/inference.py " + `
              "--model_path `"$ModelPath`" " + `
              "--model_type $ModelType " + `
              "--device $Device " + `
              "--input_file `"$InputFile`" " + `
              "--output_file `"$OutputFile`" " + `
              "--visualize"
    
    Write-Host "执行命令: $Command"
    Invoke-Expression $Command
}

Write-Host "======================================"
Write-Host "完成!" -ForegroundColor Green
