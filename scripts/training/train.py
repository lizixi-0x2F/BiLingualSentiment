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

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
from src.utils.data_utils import prepare_pretrained_dataloaders
from src.utils.train_utils import RMSELoss, evaluate, plot_learning_curves, save_metrics, EarlyStopping
from src.utils.visualization import (plot_va_scatter, plot_prediction_errors, 
                                    plot_training_progress, plot_feature_embeddings,
                                    plot_confusion_matrix)

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
    parser = argparse.ArgumentParser(description='Pretrained Model Fine-tuning for Bilingual Sentiment Analysis')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备(cuda/cpu/mps)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--monitor', type=str, default='r2', help='早停监控指标(val_loss/r2/rmse/ccc)')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--model_type', type=str, default='distilbert', choices=['distilbert', 'xlm-roberta'], help='模型类型')
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
    
    # 根据指定的模型类型选择模型
    if args.model_type == 'distilbert':
        config.MULTILINGUAL_MODEL_NAME = 'distilbert-base-multilingual-cased'
        model = MultilingualDistilBERTModel(config).to(device)
    elif args.model_type == 'xlm-roberta':
        config.MULTILINGUAL_MODEL_NAME = 'xlm-roberta-base'
        model = XLMRobertaDistilledModel(config).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    logger.info(f"模型: {config.MULTILINGUAL_MODEL_NAME}")
    
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
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    # 创建学习率调度器
    lr_scheduler = LinearWarmupLRScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # 创建早停对象
    early_stopping = EarlyStopping(
        patience=args.patience, 
        min_delta=config.EARLY_STOPPING_MIN_DELTA, 
        monitor=args.monitor
    )
    
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
            grad_clip=config.GRAD_CLIP, debug_info=args.debug
        )
        train_losses.append(train_loss)
        
        # 在验证集上评估
        with torch.no_grad():
            val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device, args.debug)
            val_losses.append(val_loss)
        
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
            logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
        elif args.monitor == 'r2' and val_metrics['r2'] > best_val_metric:
            best_val_metric = val_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"Saved best model (R²: {val_metrics['r2']:.4f})")
        elif args.monitor == 'rmse' and val_metrics['rmse'] < best_val_metric:
            best_val_metric = val_metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"Saved best model (RMSE: {val_metrics['rmse']:.4f})")
            
        # 检查早停条件
        if args.monitor == 'val_loss':
            stop = early_stopping(val_loss)
        elif args.monitor == 'r2':
            stop = early_stopping(val_metrics['r2'])
        elif args.monitor == 'rmse':
            stop = early_stopping(val_metrics['rmse'])
        else:
            stop = early_stopping(val_metrics.get(args.monitor, val_loss))
            
        if stop:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
            
        # 学习率调度器步进
        lr_scheduler.step()
          # 创建可视化目录
    viz_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, config.OUTPUT_DIR)
    
    # 绘制训练进度指标
    train_metrics = {
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    }
    plot_training_progress(train_metrics, 
                          title="训练与验证损失", 
                          output_path=os.path.join(viz_dir, "training_progress.png"))
    
    # 加载最佳模型进行测试
    logger.info("加载最佳模型进行测试评估...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
      # 在测试集上评估和收集预测结果
    all_predictions = []
    all_labels = []
    all_embeddings = []
    all_texts = []
    
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in test_dataloader:
            # 获取输入和标签
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch.get('text', [''] * len(input_ids))
            
            # 前向传播
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask, token_type_ids)
                
                # 如果模型有额外输出，可以获取嵌入
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    predictions, embeddings = outputs
                else:
                    predictions = outputs
                    # 使用模型的CLS token表示作为嵌入
                    with torch.no_grad():
                        if hasattr(model, 'model'):
                            embeddings = model.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids
                            ).last_hidden_state[:, 0]  # 使用[CLS]标记
            else:
                outputs = model(input_ids, attention_mask)
                
                # 同上处理嵌入
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    predictions, embeddings = outputs
                else:
                    predictions = outputs
                    # 使用模型的CLS token表示作为嵌入
                    with torch.no_grad():
                        if hasattr(model, 'model'):
                            embeddings = model.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            ).last_hidden_state[:, 0]  # 使用[CLS]标记
            
            # 计算损失
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            
            # 收集预测、标签、嵌入和文本
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if 'embeddings' in locals():
                all_embeddings.append(embeddings.cpu().numpy())
            all_texts.extend(texts)
    
    # 计算平均测试损失
    test_loss /= len(test_dataloader)
    
    # 合并所有批次的数据
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    
    # 计算评估指标
    test_metrics = {}
    
    # 计算整体指标
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    test_metrics['rmse'] = np.sqrt(mean_squared_error(all_labels, all_predictions))
    test_metrics['r2'] = r2_score(all_labels, all_predictions)
    test_metrics['mae'] = mean_absolute_error(all_labels, all_predictions)
    
    # 计算效价指标
    test_metrics['valence_rmse'] = np.sqrt(mean_squared_error(all_labels[:, 0], all_predictions[:, 0]))
    test_metrics['valence_r2'] = r2_score(all_labels[:, 0], all_predictions[:, 0])
    
    # 计算唤醒度指标
    test_metrics['arousal_rmse'] = np.sqrt(mean_squared_error(all_labels[:, 1], all_predictions[:, 1]))
    test_metrics['arousal_r2'] = r2_score(all_labels[:, 1], all_predictions[:, 1])
    
    # 输出测试结果
    logger.info("测试集评估结果:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Valence - RMSE: {test_metrics['valence_rmse']:.4f}, R²: {test_metrics['valence_r2']:.4f}")
    logger.info(f"Arousal - RMSE: {test_metrics['arousal_rmse']:.4f}, R²: {test_metrics['arousal_r2']:.4f}")
    
    # 生成可视化
    logger.info("生成预测可视化...")
    
    # 1. 效价-唤醒度散点图 (静态版本)
    plot_va_scatter(
        all_predictions, all_labels,
        title=f"{args.model_type}模型预测分布",
        output_path=os.path.join(viz_dir, "va_scatter_static.png")
    )
    
    # 2. 效价-唤醒度散点图 (交互式版本，带有文本标签)
    # 如果有文本信息，则使用文本样本，否则不使用
    if all_texts and not all(text == '' for text in all_texts):
        # 只使用一部分样本来避免交互式图表过大
        sample_size = min(500, len(all_predictions))
        indices = np.random.choice(len(all_predictions), sample_size, replace=False)
        
        plot_va_scatter(
            all_predictions[indices], all_labels[indices],
            texts=[all_texts[i] for i in indices],
            title=f"{args.model_type}模型预测分布 (交互式)",
            interactive=True,
            output_path=os.path.join(viz_dir, "va_scatter_interactive.html")
        )
    
    # 3. 预测误差分布图
    plot_prediction_errors(
        all_predictions, all_labels,
        title="预测误差分布",
        output_path=os.path.join(viz_dir, "prediction_errors.png")
    )
    
    # 4. 情感分类混淆矩阵
    plot_confusion_matrix(
        all_predictions, all_labels,
        title="情感象限预测混淆矩阵",
        output_path=os.path.join(viz_dir, "confusion_matrix.png")
    )
    
    # 5. 如果有嵌入向量，则可视化特征空间
    if all_embeddings:
        # 同样只使用一部分样本来避免t-SNE计算过于耗时
        sample_size = min(1000, len(all_embeddings))
        indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
        
        plot_feature_embeddings(
            all_embeddings[indices], all_labels[indices],
            texts=[all_texts[i] for i in indices] if all_texts else None,
            title=f"{args.model_type}模型特征空间",
            interactive=True,
            output_path=os.path.join(viz_dir, "feature_embeddings.html")
        )
    
    # 保存测试结果
    test_results = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_epoch": checkpoint['epoch'],
        "model_type": args.model_type,
        "model_name": config.MULTILINGUAL_MODEL_NAME
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4, default=str)
        
    logger.info(f"测试结果已保存到 {os.path.join(config.OUTPUT_DIR, 'test_results.json')}")
    logger.info("训练完成!")

if __name__ == "__main__":
    main()
