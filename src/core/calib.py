#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VA校准层实现
在模型推理阶段对VA值进行线性变换，以提高准确性
"""

import torch
import torch.nn as nn
import numpy as np

class VACalib(nn.Module):
    """
    VA校准层，使用线性变换校准VA预测值
    在训练完成后进行微调，单独训练该层参数
    """
    def __init__(self):
        super(VACalib, self).__init__()
        # 初始化为恒等变换
        self.a = nn.Parameter(torch.ones(2))  # 缩放参数
        self.b = nn.Parameter(torch.zeros(2)) # 偏置参数
        
    def forward(self, va):
        """
        对VA预测值应用线性变换
        
        参数:
            va: 形状为[batch_size, 2]的VA预测值
            
        返回:
            校准后的VA值
        """
        return self.a * va + self.b
    
    def reset_parameters(self):
        """重置参数为恒等变换"""
        nn.init.ones_(self.a)
        nn.init.zeros_(self.b)
        
    def apply_boundary_buffer(self, va, threshold=0.05):
        """
        应用边界缓冲，避免微小值引起的象限错误
        
        参数:
            va: 形状为[batch_size, 2]的VA预测值
            threshold: 边界阈值，小于此值视为中性
            
        返回:
            应用边界缓冲后的VA值
        """
        # 创建掩码，标识接近零的值
        v_mask = (va[:, 0].abs() < threshold)
        a_mask = (va[:, 1].abs() < threshold)
        
        # 创建结果，初始化为原始值
        result = va.clone()
        
        # 应用边界缓冲
        result[v_mask, 0] = 0.0  # 将接近零的V值设为0
        result[a_mask, 1] = 0.0  # 将接近零的A值设为0
        
        return result
    
    def get_quadrant_confidence(self, va):
        """
        计算象限预测的置信度
        
        参数:
            va: 形状为[batch_size, 2]的VA预测值
            
        返回:
            每个样本的象限置信度得分 (0-1)
        """
        # VA值的绝对值作为距离
        v_abs = va[:, 0].abs()
        a_abs = va[:, 1].abs()
        
        # 使用欧几里得距离作为置信度
        confidence = torch.sqrt(v_abs**2 + a_abs**2) / np.sqrt(2)
        
        # 将置信度限制在[0, 1]范围内
        return torch.clamp(confidence, 0, 1)
        
    def save(self, path):
        """保存校准层参数"""
        torch.save({
            'a': self.a.data,
            'b': self.b.data
        }, path)
        
    def load(self, path):
        """加载校准层参数"""
        checkpoint = torch.load(path)
        self.a.data = checkpoint['a']
        self.b.data = checkpoint['b']
        
    def __repr__(self):
        """打印友好的模型描述"""
        return f"VACalib(a=[{self.a[0].item():.4f}, {self.a[1].item():.4f}], b=[{self.b[0].item():.4f}, {self.b[1].item():.4f}])" 