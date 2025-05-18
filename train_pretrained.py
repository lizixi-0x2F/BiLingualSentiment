#!/usr/bin/env python
"""
基于本地预训练模型的微调训练脚本

该脚本使用之前通过download_models.py下载的本地预训练模型，进行情感分析任务的微调训练，
而不是每次训练都重新从Hugging Face下载模型。

支持的模型:
- XLM-RoBERTa Base: 用于跨语言迁移的优秀模型
- DistilBERT Multilingual: 更轻量、更快速的替代方案

使用示例:
python train_pretrained.py --model_type xlm-roberta --device cuda --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import logging
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
from src.utils.data_utils import prepare_pretrained_dataloaders
from src.utils.train_utils import RMSELoss, evaluate, plot_learning_curves, save_metrics, EarlyStopping

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预热学习率调度器
class LinearWarmupLRScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # 记录初始学习率
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])
    
    def step(self):
        self.current_step += 1
        
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            # 预热阶段 - 线性增加学习率
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step / self.warmup_steps)
            # 衰减阶段 - 线性减少学习率
            else:
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = base_lr * max(0, 1 - progress) + self.min_lr
                
            group['lr'] = lr

def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=None, debug_info=False):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        grad_clip: 梯度裁剪值
        debug_info: 是否显示调试信息
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 调试信息
            if debug_info and batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx} - Labels stats: min={labels.min().item():.4f}, max={labels.max().item():.4f}")
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
            
            # 检查输出是否包含NaN
            if torch.isnan(outputs).any():
                logger.warning(f"NaN detected in outputs at batch {batch_idx}")
                continue
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss at batch {batch_idx}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # 更新参数
            optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    
    return avg_loss

def main():
    """主函数，处理训练流程"""
    parser = argparse.ArgumentParser(description='Pretrained Model Fine-tuning for Bilingual Sentiment Analysis (Local Models)')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备(cuda/cpu/mps)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--monitor', type=str, default='r2', help='早停监控指标(val_loss/r2/rmse/ccc)')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--model_type', type=str, default='distilbert', choices=['distilbert', 'xlm-roberta'], 
                        help='模型类型: distilbert 或 xlm-roberta')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 确保设备可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA不可用，切换到CPU')
        args.device = 'cpu'
    elif args.device == 'mps' and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning('MPS不可用，切换到CPU')
        args.device = 'cpu'
        
    device = torch.device(args.device)
    
    # 创建配置对象
    config = Config()
    config.DEVICE = args.device
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.EARLY_STOPPING_METRIC = args.monitor
    config.PATIENCE = args.patience
    config.MODEL_TYPE = args.model_type
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    else:
        config.OUTPUT_DIR = f"outputs/pretrained_{args.model_type}_{timestamp}"
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 设置本地模型路径
    pretrained_models_dir = Path("pretrained_models")
    
    # 根据指定的模型类型选择模型和本地模型路径
    if args.model_type == 'distilbert':
        config.MULTILINGUAL_MODEL_NAME = 'distilbert-base-multilingual-cased'
        local_model_path = pretrained_models_dir / "distilbert-multilingual"
        
        if not local_model_path.exists():
            logger.error(f"本地模型路径不存在: {local_model_path}")
            logger.info("请先运行 download_models.py 下载模型。")
            return
            
        # 创建模型并从本地加载
        model = MultilingualDistilBERTModel(config)
        model.load_from_pretrained(str(local_model_path))
        model = model.to(device)
        logger.info(f"已从本地目录 {local_model_path} 加载 DistilBERT 模型")
        
    elif args.model_type == 'xlm-roberta':
        config.MULTILINGUAL_MODEL_NAME = 'xlm-roberta-base'
        local_model_path = pretrained_models_dir / "xlm-roberta-base"
        
        if not local_model_path.exists():
            logger.error(f"本地模型路径不存在: {local_model_path}")
            logger.info("请先运行 download_models.py 下载模型。")
            return
            
        # 创建模型并从本地加载
        model = XLMRobertaDistilledModel(config)
        model.load_from_pretrained(str(local_model_path))
        model = model.to(device)
        logger.info(f"已从本地目录 {local_model_path} 加载 XLM-RoBERTa 模型")
        
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 准备数据加载器
    tokenizer = model.get_tokenizer()
    train_dataloader, val_dataloader, test_dataloader = prepare_pretrained_dataloaders(config, tokenizer, args.batch_size)
    
    # 创建损失函数
    criterion = RMSELoss()
    
    # 创建优化器 - 针对预训练模型微调的特殊设置
    # 区分预训练权重和新增权重，使用不同的学习率和权重衰减
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # 计算总训练步数用于学习率调度
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)  # 10% 的步数用于预热
    
    # 创建学习率调度器
    lr_scheduler = LinearWarmupLRScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # 创建早停对象
    early_stopping = EarlyStopping(
        patience=args.patience, 
        min_delta=0.001,  # 早停最小增量 
        monitor=args.monitor
    )
    
    # 记录本次训练的配置
    config_info = {
        "model_type": args.model_type,
        "model_name": config.MULTILINGUAL_MODEL_NAME,
        "local_model_path": str(local_model_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "monitor": args.monitor,
        "patience": args.patience,
        "timestamp": timestamp
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    
    # 训练循环
    logger.info("开始训练...")
    
    train_losses = []
    val_losses = []
    best_val_metric = float('-inf') if args.monitor != 'val_loss' else float('inf')
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, device,
            grad_clip=1.0, debug_info=args.debug
        )
        train_losses.append(train_loss)
        
        # 在验证集上评估
        with torch.no_grad():
            val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device, args.debug)
            val_losses.append(val_loss)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 输出当前epoch的指标
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch}/{args.epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}, MAE: {val_metrics['mae']:.4f}")
        
        # 保存每个epoch的指标
        save_metrics(val_metrics, epoch, config.OUTPUT_DIR)
        
        # 检查是否保存最佳模型
        if args.monitor == 'val_loss' and val_loss < best_val_metric:
            best_val_metric = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"已保存最佳模型 (val_loss: {val_loss:.4f})")
        elif args.monitor == 'r2' and val_metrics['r2'] > best_val_metric:
            best_val_metric = val_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"已保存最佳模型 (R²: {val_metrics['r2']:.4f})")
        elif args.monitor == 'rmse' and val_metrics['rmse'] < best_val_metric:
            best_val_metric = val_metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"已保存最佳模型 (RMSE: {val_metrics['rmse']:.4f})")
            
        # 检查早停条件
        if args.monitor == 'val_loss':
            should_stop = early_stopping(val_loss)
        elif args.monitor == 'r2':
            should_stop = early_stopping(-val_metrics['r2'])  # 取负值使其为最小化问题
        elif args.monitor == 'rmse':
            should_stop = early_stopping(val_metrics['rmse'])
        else:
            should_stop = False
            
        if should_stop:
            logger.info(f"触发早停条件，在epoch {epoch}停止训练。")
            break
    
    # 训练结束，加载最佳模型进行测试
    logger.info("训练完成，在测试集上评估最佳模型...")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', -1)
        logger.info(f"已加载最佳模型 (epoch {best_epoch})")
        
        # 在测试集上评估
        with torch.no_grad():
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, device, debug_info=args.debug)
            
        logger.info(f"测试集评估结果:")
        logger.info(f"Loss: {test_loss:.4f}")
        logger.info(f"RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"R²: {test_metrics['r2']:.4f}")
        logger.info(f"MAE: {test_metrics['mae']:.4f}")
        
        # 保存测试结果
        with open(os.path.join(config.OUTPUT_DIR, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "best_epoch": best_epoch
            }, f, indent=2)
            
        # 绘制学习曲线
        plot_learning_curves(train_losses, val_losses, os.path.join(config.OUTPUT_DIR, "learning_curves.png"))
    else:
        logger.warning(f"最佳模型文件不存在: {best_model_path}")

if __name__ == "__main__":
    main()