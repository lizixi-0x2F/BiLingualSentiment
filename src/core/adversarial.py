#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对抗训练模块 - 快速梯度法(FGM)
为LTC-NCP-RNN模型实现对抗训练功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union


class FGM:
    """
    快速梯度法(Fast Gradient Method)对抗训练实现
    
    通过在嵌入层添加扰动来提高模型鲁棒性
    参考: https://arxiv.org/abs/1412.6572
    """
    
    def __init__(
        self, 
        model: nn.Module,
        epsilon: float = 0.5,
        emb_name: str = 'embedding',
        use_sign: bool = True
    ):
        """
        初始化FGM对抗训练器
        
        参数:
            model: 需要进行对抗训练的模型
            epsilon: 扰动大小
            emb_name: 嵌入层参数名
            use_sign: 是否使用梯度符号(FGSM)而非梯度值(FGM)
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.use_sign = use_sign
        
        # 备份参数梯度
        self.backup = {}
        # 保存嵌入层参数引用
        self.emb_backup = {}
        
    def attack(self):
        """
        在嵌入表示上添加对抗扰动
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is None:
                    continue
                    
                self.backup[name] = param.data.clone()
                
                # 计算扰动：梯度方向上的epsilon大小扰动
                if self.use_sign:
                    # FGSM模式: 使用梯度符号
                    norm = torch.sign(param.grad)
                else:
                    # FGM模式: 使用梯度L2范数归一化方向
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        norm = param.grad / norm
                
                # 添加扰动
                param.data.add_(self.epsilon * norm)
    
    def restore(self):
        """
        恢复嵌入表示
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        
        # 清空备份
        self.backup = {}
        
    def save_emb_grad(self):
        """
        保存嵌入层梯度用于分析
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is not None:
                    self.emb_backup[name] = param.grad.clone()
    
    def get_emb_grad_norm(self) -> Dict[str, float]:
        """
        获取嵌入层梯度范数，用于监控
        
        返回:
            包含各嵌入层梯度范数的字典
        """
        norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    norms[name] = grad_norm
        return norms 