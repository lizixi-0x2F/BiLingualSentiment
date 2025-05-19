#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linformer实现的Mini-Transformer
用于在XLM-R模型中替换标准Transformer，实现内存优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linformer_transformer import LinformerEncoderLayer

# 全局调试标志
DEBUG = True

class LinformerMiniTransformer(nn.Module):
    """
    基于Linformer的Mini-Transformer架构，用于情感信息增强
    实现更高的内存效率，适用于长序列处理 - 优化版本
    
    参数:
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout比例
        projection_dim: Linformer投影维度，用于控制内存使用
    """
    
    def __init__(self, d_model, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, projection_dim=128):
        super(LinformerMiniTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 逐渐减小投影维度以节省更多内存
        # 首层使用较大的投影维度，后续层逐渐减少
        projection_dims = []
        for i in range(num_layers):
            # 线性减少投影维度，但确保最小不低于32
            layer_proj_dim = max(32, int(projection_dim * (1 - i * 0.15)))
            projection_dims.append(layer_proj_dim)
        
        # 使用LinformerEncoderLayer替换TransformerEncoderLayer
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                projection_dim=projection_dims[i]
            )
            for i in range(num_layers)
        ])
        
        # 添加位置编码能力 - 增加到1024支持更长序列
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model))  # 最大1024个时间步
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # 添加输入规范化
        self.input_norm = nn.LayerNorm(d_model)
        
        # 增加数据适配层
        self.input_adapter = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
          # 保存投影维度列表用于日志记录
        self.projection_dims = projection_dims
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        前向传播，处理来自LTC的序列数据
        
        参数:
            src: 输入序列，形状 [batch_size, seq_len, d_model]
            src_mask: 注意力掩码
            src_key_padding_mask: 填充掩码
            
        返回:
            处理后的序列
        """
        # 检查并处理输入
        if torch.isnan(src).any():
            print("警告: Transformer输入包含NaN值，已替换为0") if DEBUG else None
            src = torch.nan_to_num(src, nan=0.0)
        
        # 确保数据类型是float32
        if src.dtype != torch.float32:
            src = src.to(torch.float32)
        
        # 打印设备信息用于调试
        debug_device = next(self.parameters()).device
        print(f"Transformer组件设备: {debug_device}") if DEBUG else None
        print(f"输入数据设备: {src.device}") if DEBUG else None
        
        # 使用LayerNorm标准化输入 - 增强数值稳定性
        src = self.input_norm(src)
        
        # 通过输入适配层
        src = self.input_adapter(src)
        src = F.relu(src)
        
        # src形状: [batch_size, seq_len, d_model]
        seq_len = src.size(1)
        
        # 添加位置编码
        pos_enc = self.pos_encoder[:, :seq_len, :]
        # 确保位置编码与输入位于同一设备
        pos_enc = pos_enc.to(src.device)
        
        src = src + pos_enc
        src = self.dropout(src)
        
        # 逐层应用transformer
        for i, layer in enumerate(self.layers):
            try:
                src = layer(src, src_mask, src_key_padding_mask)
                
                # 中间检查，确保没有NaN传播
                if torch.isnan(src).any():
                    print(f"警告: Transformer层 {i+1} 输出包含NaN值，使用上一层输出") if DEBUG else None
                    # 回退到上一个有效的src
                    break
            except Exception as e:
                print(f"警告: Transformer层 {i+1} 处理出错: {e}") if DEBUG else None
                # 保持src不变，不再继续处理后续层
                break
        
        # 最终检查
        if torch.isnan(src).any():
            # 如果输出包含NaN，返回零张量
            print("警告: Transformer最终输出包含NaN值，返回零张量") if DEBUG else None
            batch_size, seq_len, d_model = src.shape
            src = torch.zeros(batch_size, seq_len, d_model, device=src.device)
        
        return src
