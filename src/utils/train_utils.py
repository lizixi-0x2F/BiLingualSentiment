import torch
import torch.nn as nn
import os
import numpy as np
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RMSE损失函数
class RMSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 计算每个样本的每个维度的均方误差
        mse = self.mse(pred, target)
        
        # 对每个样本的所有维度求平均，得到每个样本的MSE
        if len(mse.shape) > 1:
            mse = torch.mean(mse, dim=1)
        
        # 对所有样本的MSE求平方根
        rmse = torch.sqrt(mse + 1e-8)  # 添加小量防止数值不稳定
        
        # 根据reduction方式返回损失
        if self.reduction == 'mean':
            return torch.mean(rmse)
        elif self.reduction == 'sum':
            return torch.sum(rmse)
        else:  # 'none'
            return rmse

# 一致性相关系数损失函数
class CCLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CCLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 确保输入形状一致
        batch_size = pred.size(0)
        dim = pred.size(1)
        
        # 初始化损失
        loss = torch.zeros(dim, device=pred.device)
        
        # 对每个维度分别计算CCC
        for i in range(dim):
            pred_i = pred[:, i]
            target_i = target[:, i]
            
            # 计算均值和标准差
            mean_pred = torch.mean(pred_i)
            mean_target = torch.mean(target_i)
            
            var_pred = torch.var(pred_i, unbiased=False)
            var_target = torch.var(target_i, unbiased=False)
            
            # 计算协方差
            covar = torch.mean((pred_i - mean_pred) * (target_i - mean_target))
            
            # 计算CCC
            numerator = 2 * covar
            denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + 1e-10
            
            ccc = numerator / denominator
            
            # CCC值范围为[-1,1]，转换为损失
            loss[i] = 1 - ccc
        
        # 根据reduction方式返回损失
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=None, debug_info=False, iterations_per_batch=50):
    model.train()
    total_loss = 0
    total_iterations = 0
    
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
            
            # 对每个批次进行多次迭代
            batch_loss = 0
            for iter_idx in range(iterations_per_batch):
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
                    logger.warning(f"NaN detected in outputs at batch {batch_idx}, iteration {iter_idx}")
                    continue
                
                # 计算主损失 - 回归损失
                regression_loss = criterion(outputs, labels)
                
                # 获取混合权重lambda - 默认为0.7，增加回归损失占比
                lambda_weight = 0.7
                if hasattr(model, 'config') and hasattr(model.config, 'LAMBDA_WEIGHT'):
                    lambda_weight = model.config.LAMBDA_WEIGHT
                
                # 应用混合权重 - 这里我们只是设置了权重，实际当前模型没有生成loss
                loss = regression_loss
                
                # 添加时间常数正则化损失
                if hasattr(model, 'get_tau_regularization'):
                    tau_reg_loss = model.get_tau_regularization()
                    loss = loss + tau_reg_loss
                    
                    if debug_info and batch_idx % 50 == 0 and iter_idx == 0:
                        logger.info(f"Batch {batch_idx} - Reg loss: {regression_loss.item():.4f} (λ={lambda_weight:.2f}), Tau reg: {tau_reg_loss.item():.4f}")
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at batch {batch_idx}, iteration {iter_idx}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # 更新参数
                optimizer.step()
                
                # 累积损失
                batch_loss += loss.item()
                total_iterations += 1
            
            # 计算批次平均损失
            if iterations_per_batch > 0:
                batch_loss = batch_loss / iterations_per_batch
                total_loss += batch_loss
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "iters": f"{total_iterations}"})
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 记录总迭代次数
    logger.info(f"Total iterations in this epoch: {total_iterations}")
    
    return avg_loss


def evaluate(model, dataloader, criterion, device, debug_info=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            try:
                # 前向传播
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(device)
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = model(input_ids, attention_mask)
                
                # 检查输出是否包含NaN
                if torch.isnan(outputs).any():
                    logger.warning(f"NaN in outputs during evaluation, batch {batch_idx}")
                    continue
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss during evaluation, batch {batch_idx}")
                    continue
                
                # 累积损失
                total_loss += loss.item()
                
                # 收集预测值和真实值
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # 如果没有收集到任何预测值或标签
    if not all_preds or not all_labels:
        logger.error("No valid predictions collected during evaluation")
        return float('inf'), {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': float('-inf'),
            'valence_mse': float('inf'),
            'valence_rmse': float('inf'),
            'valence_mae': float('inf'),
            'valence_r2': float('-inf'),
            'arousal_mse': float('inf'),
            'arousal_rmse': float('inf'),
            'arousal_mae': float('inf'),
            'arousal_r2': float('-inf')
        }
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    
    # 合并预测值和真实值
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 调试信息
    if debug_info:
        logger.info(f"预测/真实值统计 - 预测范围: [{all_preds.min():.4f}, {all_preds.max():.4f}], 真实范围: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
    
    # 计算评估指标
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    # 分别计算valence和arousal的指标
    valence_mse = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
    valence_rmse = np.sqrt(valence_mse)
    valence_mae = mean_absolute_error(all_labels[:, 0], all_preds[:, 0])
    valence_r2 = r2_score(all_labels[:, 0], all_preds[:, 0])
    
    arousal_mse = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
    arousal_rmse = np.sqrt(arousal_mse)
    arousal_mae = mean_absolute_error(all_labels[:, 1], all_preds[:, 1])
    arousal_r2 = r2_score(all_labels[:, 1], all_preds[:, 1])
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'valence_mse': valence_mse,
        'valence_rmse': valence_rmse,
        'valence_mae': valence_mae,
        'valence_r2': valence_r2,
        'arousal_mse': arousal_mse,
        'arousal_rmse': arousal_rmse,
        'arousal_mae': arousal_mae,
        'arousal_r2': arousal_r2
    }
    
    return avg_loss, metrics


def plot_learning_curves(train_losses, val_losses, save_dir):
    try:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('情感分析模型学习曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.close()
        
        print(f"✓ 学习曲线已保存至: {os.path.join(save_dir, 'loss_curve.png')}")
    except ImportError:
        print("! 未安装matplotlib，学习曲线未绘制")
    except Exception as e:
        print(f"! 绘制学习曲线出错: {e}")


def save_metrics(metrics, epoch, save_dir):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将numpy类型转换为Python原生类型
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v
    
    # 将指标保存为JSON文件
    metrics_file = os.path.join(save_dir, f'metrics_epoch_{epoch}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
        
    # 输出主要指标
    print(f"Epoch {epoch} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f} | V-R²: {metrics['valence_r2']:.4f} | A-R²: {metrics['arousal_r2']:.4f}")


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