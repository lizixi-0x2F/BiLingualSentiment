# 使用本地预训练模型运行训练脚本

# 检查是否有本地模型
if (-not (Test-Path "pretrained_models\xlm-roberta-base") -or -not (Test-Path "pretrained_models\distilbert-multilingual")) {
    Write-Host "本地预训练模型不存在，先下载模型..."
    python download_models.py
}

# 配置参数
$MODEL_TYPE = $args[0]
if (-not $MODEL_TYPE) { $MODEL_TYPE = "xlm-roberta" }  # 默认使用 XLM-RoBERTa

$DEVICE = $args[1]
if (-not $DEVICE) { $DEVICE = "cuda" }  # 默认使用 CUDA

$BATCH_SIZE = $args[2]
if (-not $BATCH_SIZE) { $BATCH_SIZE = 16 }  # 默认批次大小 16

$EPOCHS = $args[3]
if (-not $EPOCHS) { $EPOCHS = 10 }  # 默认训练 10 轮

$LR = $args[4]
if (-not $LR) { $LR = "2e-5" }  # 默认学习率 2e-5

Write-Host "开始训练 $MODEL_TYPE 模型..."
Write-Host "设备: $DEVICE, 批次大小: $BATCH_SIZE, 训练轮数: $EPOCHS, 学习率: $LR"

# 运行训练脚本
python train_pretrained.py `
    --model_type $MODEL_TYPE `
    --device $DEVICE `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --monitor "r2" `
    --patience 3

Write-Host "训练完成！"
