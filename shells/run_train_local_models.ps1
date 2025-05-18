# 运行基于本地预训练模型的微调训练

# 确保当前工作目录是项目根目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path (Join-Path $scriptPath "..")

# 显示消息函数
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "开始基于本地预训练模型的微调训练"

# 检查本地预训练模型是否存在
if (-not (Test-Path "pretrained_models\distilbert-multilingual") -or -not (Test-Path "pretrained_models\xlm-roberta-base")) {
    Write-ColorOutput Yellow "警告: 本地预训练模型不存在"
    Write-Output "请先运行 python download_models.py 下载模型"
    exit 1
}

# 确保输出目录存在
if (-not (Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

# 运行训练脚本 - DistilBERT
Write-ColorOutput Cyan "开始训练 DistilBERT 模型..."
python train_pretrained.py `
  --model_type distilbert `
  --batch_size 32 `
  --epochs 10 `
  --lr 3e-5 `
  --monitor r2 `
  --patience 3 `
  --output_dir outputs\pretrained_distilbert_local

# 运行训练脚本 - XLM-RoBERTa
Write-ColorOutput Cyan "开始训练 XLM-RoBERTa 模型..."
python train_pretrained.py `
  --model_type xlm-roberta `
  --batch_size 32 `
  --epochs 10 `
  --lr 2e-5 `
  --monitor r2 `
  --patience 3 `
  --output_dir outputs\pretrained_xlm_roberta_local

Write-ColorOutput Green "训练完成!"
