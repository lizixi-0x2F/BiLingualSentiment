import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import logging
import numpy as np
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.emotion_model import EmotionAnalysisModel
from src.utils.data_utils import prepare_dataloaders
from src.utils.train_utils import train_epoch, evaluate, plot_learning_curves, save_metrics, RMSELoss, CCLoss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线性预热学习率调度器
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

# 早停机制类
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.monitor = monitor
        
        # 根据监控指标确定最优值和比较方式
        if monitor in ['val_loss', 'mse', 'rmse', 'mae']:
            # 这些指标越小越好
            self.best_score = float('inf')
            self.is_better = lambda score: score < self.best_score - self.min_delta
        else:
            # 默认其他指标(如r2, ccc)越大越好
            self.best_score = float('-inf')
            self.is_better = lambda score: score > self.best_score + self.min_delta
            
        self.early_stop = False
        
    def __call__(self, score):
        if self.is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

def train(config, output_dir=None, iterations_per_batch=50):
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() or hasattr(torch.backends, "mps") 
                         and config.DEVICE == "mps" and torch.backends.mps.is_available() else "cpu")
    
    # 配置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.OUTPUT_DIR, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出将保存到: {output_dir}")
    
    # 数据准备
    train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders(config)
    print(f"数据集: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")
    
    # 创建模型
    model = EmotionAnalysisModel(config).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 定义损失函数和优化器
    criterion = RMSELoss(reduction='mean')
    
    # 调整优化器参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 设置学习率调度器
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = LinearWarmupLRScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=total_steps,
        min_lr=config.LEARNING_RATE * 0.1
    )
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        monitor=config.EARLY_STOPPING_METRIC
    )
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    metrics_history = []
    best_val_loss = float('inf')
    best_metric_value = float('-inf') if config.EARLY_STOPPING_METRIC in ['r2', 'ccc'] else float('inf')
    
    # 训练循环
    print(f"\n开始训练中英文混合情感分析模型 | 设备: {device} | 批大小: {config.BATCH_SIZE} | 学习率: {config.LEARNING_RATE}")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                grad_clip=config.GRAD_CLIP, debug_info=False, iterations_per_batch=iterations_per_batch)
        train_losses.append(train_loss)
        
        # 在验证集上评估
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, debug_info=False)
        val_losses.append(val_loss)
        metrics_history.append(val_metrics)
        
        # 更新学习率
        scheduler.step()
        
        # 保存指标
        save_metrics(val_metrics, epoch, output_dir)
        
        # 输出训练信息
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} | Loss: {train_loss:.4f}/{val_loss:.4f} | R²: {val_metrics['r2']:.4f} | V-R²: {val_metrics['valence_r2']:.4f} | A-R²: {val_metrics['arousal_r2']:.4f} | 用时: {elapsed_time:.1f}s")
        
        # 检查是否有任何NaN或Inf值
        if np.isnan(train_loss) or np.isnan(val_loss) or np.isinf(train_loss) or np.isinf(val_loss):
            logger.warning(f"发现NaN或Inf损失值，正在调整学习率")
            
            # 如果之前有保存的最佳模型，加载该模型并降低学习率
            if epoch > 1 and os.path.exists(os.path.join(output_dir, "best_model.pth")):
                logger.info("加载上一个最佳模型并降低学习率")
                checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 降低学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                logger.info(f"学习率降至 {optimizer.param_groups[0]['lr']}")
                continue
        
        # 获取当前监控指标的值
        if config.EARLY_STOPPING_METRIC == 'val_loss':
            current_metric = val_loss
            is_better = current_metric < best_metric_value
        elif config.EARLY_STOPPING_METRIC == 'r2':
            current_metric = val_metrics['r2']
            is_better = current_metric > best_metric_value
        elif config.EARLY_STOPPING_METRIC == 'rmse':
            current_metric = val_metrics['rmse']
            is_better = current_metric < best_metric_value
        elif config.EARLY_STOPPING_METRIC == 'ccc':
            # 修复：直接使用r2值，不再取相反数
            current_metric = val_metrics.get('r2', 0) if 'ccc' not in val_metrics else val_metrics['ccc']
            is_better = current_metric > best_metric_value
        else:
            # 默认使用验证损失
            current_metric = val_loss
            is_better = current_metric < best_metric_value
            
        # 保存最佳模型
        if is_better:
            best_metric_value = current_metric
            if config.EARLY_STOPPING_METRIC == 'val_loss':
                best_val_loss = val_loss
                
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, model_path)
            print(f"✓ 保存最佳模型 (指标: {config.EARLY_STOPPING_METRIC}={current_metric:.4f})")
        
        # 检查是否应该早停
        if early_stopping(current_metric):
            metric_name = config.EARLY_STOPPING_METRIC
            print(f"! 触发早停机制 ({metric_name}: {current_metric:.4f}, {early_stopping.counter}/{early_stopping.patience}轮无改善)")
            break
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, output_dir)
    
    # 保存训练配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    print(f"✓ 配置已保存至: {config_path}")
    
    # 加载最佳模型进行测试
    try:
        print("\n加载最佳模型进行测试评估...")
        best_model_path = os.path.join(output_dir, "best_model.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在测试集上评估
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        
        # 保存测试指标
        test_results_path = os.path.join(output_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        print(f"测试结果: MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
        print(f"✓ 测试结果已保存至: {test_results_path}")
        
    except Exception as e:
        logger.error(f"加载最佳模型时出错: {e}")
        print(f"! 加载最佳模型时出错: {e}")
    
    print(f"\n✓ 训练完成! 输出保存在: {output_dir}")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 'metrics': metrics_history}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感分析模型训练 (中英混合)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], help="计算设备")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    parser.add_argument("--lr", type=float, help="学习率")
    parser.add_argument("--epochs", type=int, help="训练轮次")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--patience", type=int, help="早停耐心值")
    parser.add_argument("--monitor", type=str, choices=["val_loss", "r2", "rmse", "ccc"], help="早停监控指标")
    parser.add_argument("--lambda", type=float, default=0.7, help="混合权重λ，控制回归损失与生成损失的比例，默认0.7")
    parser.add_argument("--iters_per_batch", type=int, default=50, help="每批次迭代次数，默认50")
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 更新配置
    if args.device:
        config.DEVICE = args.device
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.patience:
        config.PATIENCE = args.patience
    if args.monitor:
        config.EARLY_STOPPING_METRIC = args.monitor
    if hasattr(args, 'lambda'):
        # getattr用于处理命令行参数中的 lambda (Python关键词)
        config.LAMBDA_WEIGHT = getattr(args, 'lambda')
    
    # 开始训练
    train(config, iterations_per_batch=args.iters_per_batch) 