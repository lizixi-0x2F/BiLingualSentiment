#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linformer Self-Attention Module
为XLM-R模型提供低内存消耗的自注意力实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinformerSelfAttention(nn.Module):
    """
    Linformer自注意力实现
    使用线性投影减少序列长度，实现O(n)而非O(n²)的复杂度
    
    参数:
        d_model: 输入和输出的维度
        nhead: 注意力头数
        dropout: Dropout比例
        projection_dim: 序列长度投影维度，越小内存消耗越低 (k in paper)
        batch_first: 输入张量是否以batch为第一维
    """
    def __init__(self, d_model, nhead, dropout=0.1, projection_dim=128, batch_first=True):
        super(LinformerSelfAttention, self).__init__()
        
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.projection_dim = projection_dim
        self.batch_first = batch_first
        self.scaling = float(self.head_dim) ** -0.5
        
        # 直接创建不同的投影，这样CPU->GPU转移时内存消耗更小
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Linformer投影矩阵E_1和E_2，用于减少键和值的序列长度
        # 每个头共享投影矩阵以减少内存使用
        self.E_1 = nn.Parameter(torch.Tensor(projection_dim, d_model))
        self.E_2 = nn.Parameter(torch.Tensor(projection_dim, d_model))
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout层，使用较大的dropout以减少过拟合风险
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化投影矩阵参数"""
        # 使用Xavier均匀分布初始化线性层参数
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        
        # 初始化Linformer投影矩阵 - 更精细的初始化
        nn.init.xavier_normal_(self.E_1)
        nn.init.xavier_normal_(self.E_2)
    
    def _in_projection(self, q, k, v):
        """
        执行分离的线性投影，以优化内存使用
        """
        # 分别投影查询、键、值
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        return q, k, v
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, seq_len, d_model] 或 [seq_len, batch_size, d_model]
            key, value: 与query相同形状            attn_mask: 注意力掩码，形状为 [seq_len, seq_len]
            key_padding_mask: 键填充掩码，形状为 [batch_size, seq_len]
            
        返回:
            output: 与输入相同形状
            attn_weights: 注意力权重 [batch_size, nhead, seq_len, projection_dim]
        """
        # 处理输入张量形状
        if not self.batch_first:
            # 如果输入是 [seq_len, batch_size, d_model]，转换为 [batch_size, seq_len, d_model]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
          # 获取批量大小和序列长度
        batch_size, seq_len, _ = query.shape
        
        # 优化的线性投影 - 一次性计算q, k, v
        q, k, v = self._in_projection(query, key, value)
        
        # 重塑张量以适应多头格式
        q = q.reshape(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)        # q, k, v: [batch_size, nhead, seq_len, head_dim]
        
        # 应用Linformer投影矩阵
        k_projected = torch.matmul(k, self.E_1.t())
        v_projected = torch.matmul(v, self.E_2.t())
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k_projected.transpose(-2, -1)) * self.scaling
        
        # 应用注意力掩码（如果提供）- 这里需要修改掩码应用方式以适应Linformer
        if attn_mask is not None:
            attn_scores += attn_mask
        
        # 应用键填充掩码（如果提供）
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # 将注意力分数转换为概率
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 与值相乘并合并头
        # [batch_size, nhead, seq_len, projection_dim] x [batch_size, nhead, projection_dim, head_dim]
        # -> [batch_size, nhead, seq_len, head_dim]
        output = torch.matmul(attn_weights, v_projected)
        
        # 重塑回原始维度
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 应用输出投影
        output = self.out_proj(output)
        
        # 如果需要，转换回原始形状
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, attn_weights
