import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import logging

class GradNorm:
    """
    GradNorm动态损失平衡算法
    允许自动调整多任务学习中不同损失的权重
    """
    def __init__(self, model, initial_weights=None, alpha=1.0):
        """
        初始化GradNorm
        
        Args:
            model: 模型实例
            initial_weights: 初始任务权重 [valence_weight, arousal_weight]
            alpha: 控制平衡速度的超参数
        """
        self.model = model
        device = next(model.parameters()).device
        
        if initial_weights is None:
            self.weights = nn.Parameter(torch.ones(2, device=device))  # [valence_weight, arousal_weight]
        else:
            self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float, device=device))
        
        self.alpha = alpha
        self.init_losses = None
        self.optimizer = None
        
        # 保存最近一次计算的梯度范数，避免重复计算
        self.last_norms = None
        self.last_loss_ratios = None
    
    def set_optimizer(self, optimizer):
        """设置权重优化器"""
        self.optimizer = optimizer
    
    def get_weights(self, losses):
        """
        获取当前任务的权重
        
        Args:
            losses: 各个任务的损失列表 [valence_loss, arousal_loss]
            
        Returns:
            weights: 当前任务权重
        """
        # 第一次调用时记录初始损失
        if self.init_losses is None:
            self.init_losses = [loss.detach() for loss in losses]
        
        # 计算损失比率
        with torch.no_grad():
            loss_ratios = torch.stack([losses[i].detach() / self.init_losses[i] for i in range(len(losses))])
            
            # 计算平均损失比率
            mean_loss_ratio = torch.mean(loss_ratios)
            
            # 计算目标比率
            target_ratios = mean_loss_ratio ** self.alpha / loss_ratios
            
            # 如果已有上次计算的梯度范数，直接使用
            if self.last_norms is not None and self.last_loss_ratios is not None:
                # 检查损失比率是否显著变化
                if torch.allclose(loss_ratios, self.last_loss_ratios, rtol=0.01):
                    norms = self.last_norms
                else:
                    self.last_norms = None  # 强制重新计算
            
        # 如果没有最近计算的梯度范数，则假设所有任务梯度相同
        if self.last_norms is None:
            # 使用平均值作为近似
            device = self.weights.device
            norms = torch.ones(len(losses), device=device)
            self.last_norms = norms
            self.last_loss_ratios = loss_ratios
        
        # 根据目标比率计算新的权重
        with torch.no_grad():
            new_weights = target_ratios * self.last_norms
            new_weights = new_weights / torch.sum(new_weights) * len(losses)
        
        # 计算GradNorm损失
        gradnorm_loss = torch.sum(torch.abs(self.weights - new_weights.detach()))
        
        # 使用优化器更新权重
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            gradnorm_loss.backward()
            self.optimizer.step()
        
        # 确保权重始终是正数，并归一化
        with torch.no_grad():
            self.weights.copy_(torch.clamp(self.weights, min=0.01))
            self.weights.copy_(self.weights / torch.sum(self.weights) * len(losses))
        
        return self.weights.detach()
    
    def update_norms(self, losses, shared_parameters):
        """
        在主反向传播完成后更新任务梯度范数
        
        Args:
            losses: 各个任务的损失列表
            shared_parameters: 共享参数列表
        """
        # 获取当前设备
        device = next(shared_parameters).device
        
        # 计算共享参数的梯度范数
        grad_norm = torch.zeros(1, device=device)
        param_count = 0
        for param in shared_parameters:
            if param.grad is not None:
                grad_norm += torch.sum(param.grad ** 2)
                param_count += 1
        
        if param_count > 0:
            grad_norm = torch.sqrt(grad_norm / param_count)
        
        # 更新所有任务的梯度范数估计
        # 在多任务情况下，我们无法区分每个任务的贡献，所以使用相同的范数
        self.last_norms = torch.ones(len(losses), device=device) * grad_norm
        
        # 更新损失比率
        with torch.no_grad():
            if self.init_losses is not None:
                self.last_loss_ratios = torch.stack([losses[i].detach() / self.init_losses[i] for i in range(len(losses))])


class FGM:
    """
    Fast Gradient Method (FGM) 对抗训练
    通过在embedding层添加扰动来进行对抗训练
    """
    def __init__(self, model, epsilon=1e-3, emb_name='embeddings'):
        """
        初始化FGM
        
        Args:
            model: 模型实例
            epsilon: 扰动大小
            emb_name: 嵌入层参数的名称
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
    
    def attack(self):
        """
        对抗攻击：在embedding上添加扰动
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self):
        """
        恢复embedding的原始值
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """
    预热+余弦衰减的学习率调度器
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器实例
            warmup_steps: 预热步数
            total_steps: 总步数
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        # 记录初始学习率
        self.initial_lrs = []
        for group in optimizer.param_groups:
            self.initial_lrs.append(group['lr'])
    
    def step(self, current_step):
        """
        更新学习率
        
        Args:
            current_step: 当前训练步数
        """
        # 预热阶段
        if current_step < self.warmup_steps:
            progress = float(current_step) / float(max(1, self.warmup_steps))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.initial_lrs[i] * progress
        # 余弦衰减阶段
        else:
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = max(self.min_lr, cosine_value * self.initial_lrs[i])
        
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """
        返回调度器的状态字典，用于保存检查点
        
        Returns:
            state_dict: 状态字典
        """
        return {
            'initial_lrs': self.initial_lrs,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """
        从状态字典中加载调度器状态
        
        Args:
            state_dict: 状态字典
        """
        self.initial_lrs = state_dict['initial_lrs']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']


def compute_kl_loss(p, q, pad_mask=None):
    """
    计算KL散度损失
    用于R-Drop正则化
    
    Args:
        p: 第一次前向传播的logits
        q: 第二次前向传播的logits
        pad_mask: 填充掩码
        
    Returns:
        kl_loss: KL散度损失
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # 对称KL散度
    kl_loss = (p_loss + q_loss) / 2
    
    # 应用掩码（如果有）
    if pad_mask is not None:
        kl_loss = kl_loss * pad_mask.unsqueeze(-1)
    
    # 归约损失
    kl_loss = kl_loss.sum(-1).mean()
    
    return kl_loss 