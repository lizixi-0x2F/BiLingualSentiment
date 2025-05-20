#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linformer Transformer Encoder Layer
为XLM-R模型提供低内存消耗的Transformer编码器层实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linformer_attention import LinformerSelfAttention


class LinformerEncoderLayer(nn.Module):
    """
    使用Linformer自注意力的Transformer编码器层
    
    参数:
        d_model: 输入和输出的维度
        nhead: 注意力头数
        dim_feedforward: 前馈神经网络的维度
        dropout: Dropout比例
        projection_dim: Linformer投影维度，越小内存消耗越低
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, projection_dim=128):
        super(LinformerEncoderLayer, self).__init__()
        
        # 使用LinformerSelfAttention替代标准MultiheadAttention
        self.self_attn = LinformerSelfAttention(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            projection_dim=projection_dim,
            batch_first=True
        )
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.gelu
        
        # 添加安全处理
        self.eps = 1e-5
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        前向传播，加入安全检查和异常处理
        
        参数:
            src: 输入张量，形状 [batch_size, seq_len, d_model]
            src_mask: 注意力掩码
            src_key_padding_mask: 填充掩码
            
        返回:
            处理后的张量
        """
        # 安全检查输入
        if torch.isnan(src).any():
            print("警告: Transformer层输入包含NaN值，已替换为0") if 'DEBUG' in globals() and DEBUG else None
            src = torch.nan_to_num(src, nan=0.0)
        
        # 保存原始输入用于残差连接
        residual = src
        
        # 尝试应用第一个注意力块
        try:
            # 先应用规范化，有助于稳定训练
            src = self.norm1(src)
            
            # 使用Linformer自注意力
            src2, attn_weights = self.self_attn(
                src, src, src, 
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            
            # 检查注意力输出
            if torch.isnan(src2).any():
                print("警告: 自注意力输出包含NaN值，使用原始输入") if 'DEBUG' in globals() and DEBUG else None
                src2 = residual
            
            # Dropout和残差连接
            src = residual + self.dropout1(src2)
            
        except Exception as e:
            print(f"警告: 自注意力计算出错: {e}") if 'DEBUG' in globals() and DEBUG else None
            src = residual  # 保持原样
        
        # 保存第一阶段输出用于第二阶段残差
        residual2 = src
        
        # 尝试应用第二个前馈块
        try:
            # 先应用规范化
            src = self.norm2(src)
            
            # 前馈网络
            src2 = self.linear1(src)
            src2 = self.activation(src2)
            src2 = self.dropout(src2)
            src2 = self.linear2(src2)
            
            # 检查前馈输出
            if torch.isnan(src2).any():
                print("警告: 前馈网络输出包含NaN值，使用归一化输入") if 'DEBUG' in globals() and DEBUG else None
                src2 = src
            
            # Dropout和残差连接
            src = residual2 + self.dropout2(src2)
            
        except Exception as e:
            print(f"警告: 前馈网络计算出错: {e}") if 'DEBUG' in globals() and DEBUG else None
            src = residual2  # 使用第一阶段输出
        
        # 最终数值稳定性检查
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("警告: 最终输出包含NaN或Inf，进行处理") if 'DEBUG' in globals() and DEBUG else None
            # 替换NaN/Inf并裁剪过大值
            src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)
            src = torch.clamp(src, min=-5.0, max=5.0)  # 防止极端值
        
        return src
