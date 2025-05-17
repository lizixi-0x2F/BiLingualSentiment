"""
使用混合精度训练的示例脚本
通过启用PyTorch的自动混合精度(AMP)功能，可以显著减少GPU内存使用并加速训练

主要优势:
- 减少约50%的GPU内存使用
- 提高训练速度（取决于GPU架构，Volta/Turing/Ampere架构有明显提升）
- 保持模型精度

适用于:
- 使用NVIDIA GPU进行训练
- CUDA版本 >= 10.0
- PyTorch版本 >= 1.6.0
"""

import torch
import argparse
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
from src.utils.data_utils import prepare_pretrained_dataloaders
from src.utils.train_utils import RMSELoss, evaluate, plot_learning_curves, save_metrics, EarlyStopping

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_with_amp():
    """使用自动混合精度(AMP)进行训练的示例函数"""
    parser = argparse.ArgumentParser(description='使用混合精度训练预训练模型')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备(cuda/cpu/mps)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--model_type', type=str, default='distilbert', choices=['distilbert', 'xlm-roberta'], help='模型类型')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 确保CUDA可用
    if not torch.cuda.is_available():
        logger.error("混合精度训练需要CUDA支持，但CUDA不可用")
        return
        
    device = torch.device('cuda')
    
    # 创建配置对象
    config = Config()
    config.DEVICE = 'cuda'
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.MODEL_TYPE = args.model_type
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    else:
        config.OUTPUT_DIR = f"outputs/amp_{args.model_type}_{timestamp}"
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 根据指定的模型类型选择模型
    if args.model_type == 'distilbert':
        config.MULTILINGUAL_MODEL_NAME = 'distilbert-base-multilingual-cased'
        model = MultilingualDistilBERTModel(config).to(device)
    elif args.model_type == 'xlm-roberta':
        config.MULTILINGUAL_MODEL_NAME = 'xlm-roberta-base'
        model = XLMRobertaDistilledModel(config).to(device)
    
    # 准备数据加载器
    tokenizer = model.get_tokenizer()
    train_dataloader, val_dataloader, test_dataloader = prepare_pretrained_dataloaders(config, tokenizer, args.batch_size)
    
    # 创建损失函数
    criterion = RMSELoss()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 创建学习率调度器
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    # ======== 混合精度训练设置 ========
    # 创建GradScaler用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info(f"开始使用自动混合精度训练 {args.model_type} 模型...")
    
    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用自动混合精度进行前向传播
            with torch.cuda.amp.autocast():
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(device)
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = model(input_ids, attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
            
            # 使用scaler进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            # 累积损失
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Average Loss: {avg_loss:.4f}")
        
        # 在验证集上评估
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # 验证时也使用混合精度
                val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device, debug_info=False)
                
        logger.info(f"Validation Loss: {val_loss:.4f} | R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
    }, os.path.join(config.OUTPUT_DIR, "final_model.pth"))
    
    logger.info(f"模型已保存到 {os.path.join(config.OUTPUT_DIR, 'final_model.pth')}")
    
    # 在测试集上进行最终评估
    logger.info("在测试集上进行评估...")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, device, debug_info=False)
            
    logger.info(f"Test Loss: {test_loss:.4f} | R²: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f}")
    
    # 比较混合精度训练和标准精度训练的内存使用
    logger.info("=== 内存使用比较（理论值） ===")
    logger.info("标准FP32训练: 约 4 bytes/参数")
    logger.info("混合精度训练: 约 2-3 bytes/参数 (平均)")
    logger.info("内存使用减少: 约 30-50%")
    
    # 获取实际内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        
        logger.info(f"当前实际GPU内存分配: {memory_allocated:.2f} GB")
        logger.info(f"当前实际GPU内存预留: {memory_reserved:.2f} GB")
    
if __name__ == "__main__":
    train_with_amp()
