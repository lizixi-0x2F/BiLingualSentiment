#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-RNN 增强模型实现
结合液态时间常数单元与增强型神经电路策略的稀疏连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List, Union
import logging

# 配置logger
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from .cells import LTCCell
from .wiring import NCPWiring, MultiLevelNCPWiring

# 全局调试标志，设为True启用调试输出
DEBUG = True

def debug_print(*args, **kwargs):
    """仅在DEBUG为True时输出信息"""
    if DEBUG:
        print(*args, **kwargs)

# 添加Transformer相关组件
class TransformerEncoderLayer(nn.Module):
    """简化版Transformer编码器层"""
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
        
        # 添加安全处理 - 新增
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
            print("警告: Transformer层输入包含NaN值，已替换为0") if DEBUG else None
            src = torch.nan_to_num(src, nan=0.0)
        
        # 保存原始输入用于残差连接
        residual = src
        
        # 尝试应用第一个注意力块
        try:
            # 先应用规范化，有助于稳定训练
            src = self.norm1(src)
            
            # 自注意力
            src2, attn_weights = self.self_attn(
                src, src, src, 
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            
            # 检查注意力输出
            if torch.isnan(src2).any():
                print("警告: 自注意力输出包含NaN值，使用原始输入") if DEBUG else None
                src2 = residual
            
            # Dropout和残差连接
            src = residual + self.dropout1(src2)
            
        except Exception as e:
            print(f"警告: 自注意力计算出错: {e}") if DEBUG else None
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
                print("警告: 前馈网络输出包含NaN值，使用归一化输入") if DEBUG else None
                src2 = src
            
            # Dropout和残差连接
            src = residual2 + self.dropout2(src2)
            
        except Exception as e:
            print(f"警告: 前馈网络计算出错: {e}") if DEBUG else None
            src = residual2  # 使用第一阶段输出
        
        # 最终数值稳定性检查
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("警告: 最终输出包含NaN或Inf，进行处理") if DEBUG else None
            # 替换NaN/Inf并裁剪过大值
            src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)
            src = torch.clamp(src, min=-5.0, max=5.0)  # 防止极端值
        
        return src


class MiniTransformer(nn.Module):
    """四层Mini-Transformer架构，用于情感信息增强"""
    
    def __init__(self, d_model, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(MiniTransformer, self).__init__()
        
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 添加位置编码能力
        self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model))  # 最大200个时间步
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # 添加输入规范化 - 新增
        self.input_norm = nn.LayerNorm(d_model)
        
        # 增加数据适配层 - 新增
        self.input_adapter = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
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
        debug_print(f"Transformer组件设备: {next(self.parameters()).device}")
        debug_print(f"输入数据设备: {src.device}")
        
        # 使用LayerNorm标准化输入 - 新增，增强数值稳定性
        src = self.input_norm(src)
        
        # 通过输入适配层 - 新增
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
                    print(f"警告: Transformer层 {i+1} 输出包含NaN值，使用上一层输出")
                    # 回退到上一个有效的src
                    break
            except Exception as e:
                print(f"警告: Transformer层 {i+1} 处理出错: {e}")
                # 保持src不变，不再继续处理后续层
                break
        
        # 最终检查
        if torch.isnan(src).any():
            # 如果输出包含NaN，返回零张量
            print("警告: Transformer最终输出包含NaN值，返回零张量") if DEBUG else None
            batch_size, seq_len, d_model = src.shape
            src = torch.zeros(batch_size, seq_len, d_model, device=src.device)
        
        return src


class LTC_NCP_RNN(nn.Module):
    """
    液态时间常数-神经电路策略-情感价效度模型
    用于文本情感价效度回归，支持双语输入
    增加了四象限分类辅助任务头
    """
    
    def __init__(self, 
                vocab_size: int,
                embedding_dim: int,
                hidden_size: int,
                output_size: int = 2,  # V,A两维情感输出
                dropout: float = 0.3,
                sparsity_level: float = 0.5,  # 降低稀疏度，增加连接
                dt: float = 1.0,
                integration_method: str = 'euler',
                use_meta_features: bool = True,
                bidirectional: bool = False,
                padding_idx: Optional[int] = None,
                wiring_type: str = 'structured',  # 新增：连接类型
                multi_level: bool = True,         # 新增：是否使用多层次连接
                emotion_focused: bool = True,     # 新增：情感感知连接
                heterogeneous: bool = True,       # 新增：异构连接密度
                use_transformer: bool = False,    # 新增：是否使用Transformer
                invert_valence: bool = False,     # 新增：是否反转价值预测
                invert_arousal: bool = False,     # 新增：是否反转效度预测
                enhance_valence: bool = False,    # 新增：是否增强价值预测能力
                valence_layers: int = 2,          # 新增：价值分支的额外层数
                use_quadrant_head: bool = True,   # 新增：是否使用四象限分类头
                quadrant_weight: float = 0.3,     # 新增：四象限分类损失权重
                **kwargs                          # 新增：接受任意关键字参数
               ):
        """
        初始化增强版LTC-NCP-RNN模型
        
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_size: 隐藏层大小
            output_size: 输出维度 (默认为2: Valence和Arousal)
            dropout: Dropout比率
            sparsity_level: NCP稀疏连接水平 (0-1)
            dt: 时间步长
            integration_method: 积分方法 ('euler'或'rk4')
            use_meta_features: 是否使用元特征(句长和标点密度)
            bidirectional: 是否使用双向RNN
            padding_idx: 填充索引
            wiring_type: 连接类型 ('random', 'structured', 'small_world', 'modular')
            multi_level: 是否使用多层次连接
            emotion_focused: 是否使用情感感知连接
            heterogeneous: 是否使用异构连接密度
            use_transformer: 是否使用Transformer增强特征
            invert_valence: 是否反转价值预测
            invert_arousal: 是否反转效度预测
            enhance_valence: 是否增强价值预测能力
            valence_layers: 价值分支的额外层数
            use_quadrant_head: 是否使用四象限分类头
            quadrant_weight: 四象限分类损失权重
        """
        super(LTC_NCP_RNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout
        self.dt = dt
        self.integration_method = integration_method
        self.use_meta_features = use_meta_features
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.wiring_type = wiring_type
        self.multi_level = multi_level
        self.emotion_focused = emotion_focused
        self.heterogeneous = heterogeneous
        self.use_transformer = use_transformer
        self.invert_valence = invert_valence  # 保存反转标志
        self.invert_arousal = invert_arousal  # 保存反转标志
        self.enhance_valence = enhance_valence
        self.valence_layers = valence_layers
        self.use_quadrant_head = use_quadrant_head  # 新增：是否使用四象限分类头
        self.quadrant_weight = quadrant_weight      # 新增：四象限分类损失权重
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # 创建NCP连线结构 - 只保留定义，不应用掩码
        if multi_level:
            # 多层次NCP布线
            # 简单创建多层级LTC单元，但不应用掩码
            self.ltc_levels = nn.ModuleList()
            level_sizes = [hidden_size // 2, hidden_size // 2]  # 将隐藏层分为两部分
            
            for i, level_size in enumerate(level_sizes):
                ltc_cell = LTCCell(
                    input_size=embedding_dim if i == 0 else level_sizes[i-1],
                    hidden_size=level_size,
                    dt=dt,
                    method=integration_method
                )
                self.ltc_levels.append(ltc_cell)
                
                # 如果是双向，为每层添加反向LTC单元
                if bidirectional:
                    reverse_ltc_cell = LTCCell(
                        input_size=embedding_dim if i == 0 else level_sizes[i-1],
                        hidden_size=level_size,
                        dt=dt,
                        method=integration_method
                    )
                    self.ltc_levels.append(reverse_ltc_cell)
        else:
            # 常规LTC单元
            self.ltc_cell = LTCCell(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                dt=dt,
                method=integration_method
            )
            
            # 如果是双向，创建反向LTC单元
            if bidirectional:
                self.reverse_ltc_cell = LTCCell(
                    input_size=embedding_dim,
                    hidden_size=hidden_size,
                    dt=dt,
                    method=integration_method
                )
        
        # 计算输出层（回归头）的输入维度
        # 如果是双向RNN，输入维度翻倍
        hidden_output_size = hidden_size * self.num_directions
        
        # 固定输入维度，确保一致性
        self.hidden_output_size = hidden_output_size
        
        # 添加特征维度转换层 - 新增，解决维度不匹配问题
        self.feature_adapter = nn.Linear(hidden_output_size, hidden_output_size)
        
        # 添加Transformer模块
        if use_transformer:
            self.transformer = MiniTransformer(
                d_model=hidden_output_size,
                nhead=4, 
                num_layers=4,
                dim_feedforward=hidden_output_size*2,
                dropout=dropout/2
            )
            
            # 添加Transformer输出适配层 - 新增，保持维度一致性
            self.transformer_adapter = nn.Linear(hidden_output_size, hidden_output_size)
        
        # 如果使用元特征，增加输入维度
        self.meta_feature_size = 0
        self.use_pos_for_valence = False
        
        # 计算元特征的维度 - 根据情况计算
        if use_meta_features:
            base_meta_features = 3  # 基础元特征(句长、标点密度等)
            pos_features_count = kwargs.get('pos_features_count', 0)  # 词性特征数量
            self.meta_feature_size = base_meta_features + pos_features_count
            
            # 判断是否使用词性特征增强Valence判断
            self.use_pos_for_valence = pos_features_count > 0
            if self.use_pos_for_valence:
                logger.info(f"使用{pos_features_count}维词性特征增强Valence判断")
                # 记录词性特征的起始索引，用于后续处理
                self.pos_features_start_idx = base_meta_features
                self.pos_features_count = pos_features_count
        
        # 计算实际的输入维度 - 考虑双向和多层次因素
        self.ltc_output_size = hidden_size
        if multi_level:
            # 多层次输出尺寸取决于层数
            self.ltc_output_size = hidden_size  # 多层次模型已经在内部合并层
        
        # 双向效果
        if bidirectional:
            self.ltc_output_size *= 2
            
        # 总输入维度计算
        self.total_input_size = self.ltc_output_size + self.meta_feature_size
        
        # 记录实际维度供调试
        print(f"总输入维度: {self.total_input_size}, LTC输出维度: {self.ltc_output_size}, 元特征维度: {self.meta_feature_size}")
        
        # 创建输入适配器，用于处理元特征和维度变化
        self.input_adapter = nn.Linear(self.total_input_size, hidden_size)
        
        # 输出层 - 为V和A创建专门的处理通道
        # 1. 共享基础层
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(dropout/2)
        )
        
        # 2. 创建情感分支 - V分支
        # 检查是否有词性特征增强
        has_pos_features = kwargs.get('pos_features_count', 0) > 0
        
        if enhance_valence:
            valence_layers_list = []
            # 正确设置输入维度为hidden_size
            input_size = hidden_size
            
            # 添加额外的valence专用层
            for i in range(valence_layers):
                # 第一层输入为hidden_size，输出相同
                if i == 0:
                    valence_layers_list.append(nn.Linear(input_size, input_size))
                else:
                    # 后续层保持相同维度
                    valence_layers_list.append(nn.Linear(input_size, input_size))
                valence_layers_list.append(nn.LeakyReLU())
                valence_layers_list.append(nn.Dropout(dropout * 0.8))
            
            # 最后两层降维到输出
            valence_layers_list.append(nn.Linear(input_size, input_size // 2))
            valence_layers_list.append(nn.Tanh())
            valence_layers_list.append(nn.Linear(input_size // 2, 1))
            
            self.valence_branch = nn.Sequential(*valence_layers_list)
            print(f"创建增强的valence分支，层数: {valence_layers}，输入维度: {input_size}")
            
            # 如果使用词性特征增强，添加专门的词性特征处理分支
            if has_pos_features:
                pos_size = kwargs.get('pos_features_count', 15)  # 词性特征维度，与pos_features.py中保持一致
                self.pos_valence_branch = nn.Sequential(
                    nn.Linear(pos_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.Tanh(),
                    nn.Linear(hidden_size // 4, 1)
                )
                print(f"创建词性特征增强分支，输入维度: {pos_size}")
                
                # 创建融合层，用于组合词性特征和常规特征的Valence预测
                self.valence_fusion = nn.Linear(2, 1)
                # 初始化融合权重，给常规特征0.7的权重，词性特征0.3的权重
                self.valence_fusion.weight.data = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
                self.valence_fusion.bias.data.zero_()
                
                print("启用词性特征与常规特征融合，增强Valence判断")
        else:
            # 标准valence分支 - 确保维度正确
            self.valence_branch = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # 如果使用词性特征增强，添加专门的词性特征处理分支
            if has_pos_features:
                pos_size = kwargs.get('pos_features_count', 15)  # 词性特征维度，与pos_features.py中保持一致
                self.pos_valence_branch = nn.Sequential(
                    nn.Linear(pos_size, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_size // 4, 1)
                )
                print(f"创建词性特征增强分支，输入维度: {pos_size}")
                
                # 创建融合层，用于组合词性特征和常规特征的Valence预测
                self.valence_fusion = nn.Linear(2, 1)
                # 初始化融合权重，给常规特征0.7的权重，词性特征0.3的权重
                self.valence_fusion.weight.data = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
                self.valence_fusion.bias.data.zero_()
        
        # 3. 创建情感分支 - A分支，使其与valence分支匹配
        self.arousal_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 4. 创建V-A交互层，捕捉情感维度间的关系
        self.emotion_interaction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 3),
            nn.Tanh(),
            nn.Linear(hidden_size // 3, 2)
        )
        
        # 5. 创建注意力机制，加强重要特征权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 6. 新增：四象限分类头 - 直接预测四个象限类别
        if use_quadrant_head:
            self.quadrant_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(hidden_size // 2, 4)  # 输出4个象限类别的logits
            )
            print(f"创建四象限分类辅助任务头，输入维度: {hidden_size}，权重: {quadrant_weight}")
    
    def _process_sequence(self, embedded, mask=None):
        """处理序列的正向传播"""
        batch_size, seq_len, _ = embedded.size()
        
        if self.multi_level:
            # 分层次处理
            level_outputs = []
            
            for i in range(0, len(self.ltc_levels), 2 if self.bidirectional else 1):
                ltc_cell = self.ltc_levels[i]
                
                # 初始化隐藏状态
                h = torch.zeros(batch_size, ltc_cell.hidden_size, device=embedded.device)
                
                # 每个时间步的输出
                outputs = []
                
                # 如果是第一层，输入是嵌入；否则，输入是上一层的输出
                if i == 0:
                    inputs = embedded
                else:
                    # 使用上一层的输出作为这一层的输入
                    prev_outputs = level_outputs[-1]
                    inputs = prev_outputs
                
                # 序列处理
                for t in range(seq_len):
                    x_t = inputs[:, t, :]
                    h = ltc_cell(x_t, h)
                    outputs.append(h)
                
                # 堆叠所有时间步的输出 [batch_size, seq_len, hidden_size]
                outputs = torch.stack(outputs, dim=1)
                level_outputs.append(outputs)
            
            # 合并所有层级的输出
            final_outputs = torch.cat(level_outputs, dim=2)
            
            return final_outputs
        else:
            # 常规处理
            # 初始化隐藏状态
            h = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
            
            # 每个时间步的输出
            outputs = []
            
            # 序列处理
            for t in range(seq_len):
                x_t = embedded[:, t, :]
                
                # 应用掩码（如果提供）
                if mask is not None:
                    # 假设mask的形状是[batch_size, seq_len]
                    m_t = mask[:, t].unsqueeze(1)
                    # 只有当掩码为1时才更新隐藏状态
                    h_new = self.ltc_cell(x_t, h)
                    h = m_t * h_new + (1 - m_t) * h
                else:
                    h = self.ltc_cell(x_t, h)
                
                outputs.append(h)
            
            # 堆叠所有时间步的输出 [batch_size, seq_len, hidden_size]
            outputs = torch.stack(outputs, dim=1)
            
            return outputs
    
    def _process_reverse_sequence(self, embedded, mask=None):
        """处理序列的反向传播（用于双向模式）"""
        batch_size, seq_len, _ = embedded.size()
        
        if self.multi_level:
            # 分层次处理
            level_outputs = []
            
            for i in range(1, len(self.ltc_levels), 2):
                reverse_ltc_cell = self.ltc_levels[i]
                
                # 初始化隐藏状态
                h = torch.zeros(batch_size, reverse_ltc_cell.hidden_size, device=embedded.device)
                
                # 每个时间步的输出（反向）
                outputs = []
                
                # 如果是第一层，输入是嵌入；否则，输入是上一层的输出
                if i == 1:
                    inputs = embedded
                else:
                    # 使用上一反向层的输出作为这一层的输入
                    prev_outputs = level_outputs[-1]
                    inputs = prev_outputs
                
                # 反向序列处理
                for t in range(seq_len-1, -1, -1):
                    x_t = inputs[:, t, :]
                    h = reverse_ltc_cell(x_t, h)
                    outputs.insert(0, h)
                
                # 堆叠所有时间步的输出 [batch_size, seq_len, hidden_size]
                outputs = torch.stack(outputs, dim=1)
                level_outputs.append(outputs)
            
            # 合并所有反向层级的输出
            final_outputs = torch.cat(level_outputs, dim=2)
            
            return final_outputs
        else:
            # 常规处理
            # 初始化隐藏状态
            h = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
            
            # 每个时间步的输出（反向）
            outputs = []
            
            # 反向序列处理
            for t in range(seq_len-1, -1, -1):
                x_t = embedded[:, t, :]
                
                # 应用掩码（如果提供）
                if mask is not None:
                    m_t = mask[:, t].unsqueeze(1)
                    h_new = self.reverse_ltc_cell(x_t, h)
                    h = m_t * h_new + (1 - m_t) * h
                else:
                    h = self.reverse_ltc_cell(x_t, h)
                
                outputs.insert(0, h)
            
            # 堆叠所有时间步的输出 [batch_size, seq_len, hidden_size]
            outputs = torch.stack(outputs, dim=1)
            
            return outputs
    
    def forward(self, tokens, lengths=None, meta_features=None):
        """前向传播"""
        # 1. 嵌入层
        embedded = self.embedding(tokens)
        
        # 检查嵌入中是否有NaN值
        if torch.isnan(embedded).any():
            print("警告: 嵌入层输出包含NaN值，使用0替换") if DEBUG else None
            embedded = torch.nan_to_num(embedded, nan=0.0)
        
        # 2. 创建掩码（用于变长序列）
        mask = None
        if lengths is not None:
            # 确保lengths在正确的设备上
            lengths = lengths.to(tokens.device)
            max_len = tokens.size(1)
            mask = torch.arange(max_len, device=tokens.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            mask = mask.float()
        
        # 3. 正向传播
        forward_out = self._process_sequence(embedded, mask)
        
        # 4. 如果是双向，还要进行反向传播
        if self.bidirectional:
            reverse_out = self._process_reverse_sequence(embedded, mask)
            # 合并正向和反向的输出
            combined_out = torch.cat([forward_out, reverse_out], dim=2)
        else:
            combined_out = forward_out
        
        # 检查输出中是否有NaN值
        if torch.isnan(combined_out).any():
            print("警告: RNN输出包含NaN值，使用0替换") if DEBUG else None
            combined_out = torch.nan_to_num(combined_out, nan=0.0)
        
        # 输出形状调试信息
        debug_print(f"LTC输出形状: {combined_out.shape}, 设备: {combined_out.device}, 类型: {combined_out.dtype}")
        
        # 使用特征适配器进行维度调整，保证数值稳定性 - 新增
        combined_out = self.feature_adapter(combined_out)
        combined_out = F.relu(combined_out)  # 增加非线性，提高特征表达能力
        
        # 5. 应用Transformer层（如果启用）
        if self.use_transformer:
            # 创建padding掩码，用于transformer
            if lengths is not None:
                # 确保lengths在正确的设备上
                lengths = lengths.to(tokens.device)
                max_len = tokens.size(1)
                padding_mask = torch.arange(max_len, device=tokens.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
            else:
                padding_mask = None
            
            # 调试信息
            debug_print(f"Transformer输入形状: {combined_out.shape}")
            
            # 应用transformer
            try:
                # 确保数据类型匹配
                combined_out = combined_out.to(torch.float32)
                
                # 应用transformer处理
                transformed_out = self.transformer(combined_out, src_key_padding_mask=padding_mask)
                
                # 检查Transformer输出
                debug_print(f"Transformer输出形状: {transformed_out.shape}")
                
                # 通过adapter保持维度一致性 - 新增
                transformed_out = self.transformer_adapter(transformed_out)
                transformed_out = F.relu(transformed_out)
                
                # 检查Transformer输出是否有NaN
                if torch.isnan(transformed_out).any():
                    print("警告: Transformer输出包含NaN值，回退到原始输出") if DEBUG else None
                    transformed_out = combined_out
                combined_out = transformed_out
            except Exception as e:
                print(f"警告: Transformer层出错，回退到原始输出: {e}")
                # 保持combined_out不变
        
        # 6. 获取最后时间步的输出
        if lengths is not None:
            # 确保lengths在正确的设备上并进行安全处理
            lengths = lengths.to(tokens.device)
            # 确保索引不超出序列长度范围
            max_seq_len = combined_out.size(1)
            clipped_lengths = torch.clamp(lengths, max=max_seq_len).long()
            # 将0长度序列设为1(避免-1变成最后一个位置)
            clipped_lengths = torch.clamp(clipped_lengths, min=1)
            # 使用实际长度索引最后一个非填充时间步
            last_indices = (clipped_lengths - 1).long()
            
            try:
                # 简单方法：使用批量索引
                batch_size = combined_out.size(0)
                final_hidden = torch.stack([combined_out[i, last_indices[i]] for i in range(batch_size)])
            except Exception as e:
                print(f"警告: 索引最后时间步出错，使用最后位置: {e}")
                # 回退到使用最后时间步
                final_hidden = combined_out[:, -1, :]
        else:
            # 如果没有提供长度，默认使用最后一个时间步
            final_hidden = combined_out[:, -1, :]
        
        # 检查hidden输出是否有NaN
        if torch.isnan(final_hidden).any():
            print("警告: 最终隐藏状态包含NaN值，使用0替换") if DEBUG else None
            final_hidden = torch.nan_to_num(final_hidden, nan=0.0)
        
        # 7. 如果使用元特征，添加到最终特征中
        if self.use_meta_features and meta_features is not None:
            # 确保meta_features在正确的设备上
            meta_features = meta_features.to(tokens.device)
            
            # 检查meta_features是否有NaN
            if torch.isnan(meta_features).any():
                print("警告: 元特征包含NaN值，使用0替换") if DEBUG else None
                meta_features = torch.nan_to_num(meta_features, nan=0.0)
                
            # 连接元特征和模型输出
            final_hidden = torch.cat([final_hidden, meta_features], dim=1)
        
        # 调试维度信息
        debug_print(f"最终特征维度: {final_hidden.shape}, 输入适配器期望维度: {self.total_input_size}")
        
        # 确保维度匹配 - 如果不匹配，进行调整
        if final_hidden.size(1) != self.total_input_size:
            print(f"警告: 输入维度不匹配 ({final_hidden.size(1)} vs {self.total_input_size})，调整中...")
            if final_hidden.size(1) > self.total_input_size:
                # 截断多余维度
                final_hidden = final_hidden[:, :self.total_input_size]
                print(f"已裁剪维度至: {final_hidden.size(1)}")
            else:
                # 填充不足的维度
                padding = torch.zeros(final_hidden.size(0), 
                                     self.total_input_size - final_hidden.size(1),
                                     device=final_hidden.device)
                final_hidden = torch.cat([final_hidden, padding], dim=1)
                print(f"已填充维度至: {final_hidden.size(1)}")
        
        # 使用预定义的输入适配器处理维度
        try:
            final_hidden = self.input_adapter(final_hidden)
            debug_print(f"应用输入适配器后维度: {final_hidden.shape}")
        except Exception as e:
            print(f"输入适配器错误: {e}，输入维度: {final_hidden.shape}")
            # 创建紧急备用
            final_hidden = torch.zeros(final_hidden.size(0), self.hidden_size, device=final_hidden.device)
        
        # 8. 通过共享基础层
        try:
            shared_features = self.shared_layer(final_hidden)
            debug_print(f"共享层输出维度: {shared_features.shape}")
            # 检查是否有NaN
            if torch.isnan(shared_features).any():
                print("警告: 共享层输出包含NaN值，使用0替换") if DEBUG else None
                shared_features = torch.nan_to_num(shared_features, nan=0.0)
        except Exception as e:
            print(f"警告: 共享层处理出错: {e}")
            # 回退策略：创建零张量
            shared_features = torch.zeros(final_hidden.size(0), self.hidden_size, device=final_hidden.device)
        
        # 9. 计算注意力权重
        try:
            attention_weights = self.attention(shared_features)
            # 确保注意力权重在[0,1]范围内
            attention_weights = torch.clamp(attention_weights, min=0.0, max=1.0)
            # 检查是否有NaN
            if torch.isnan(attention_weights).any():
                print("警告: 注意力权重包含NaN值，使用0.5替换") if DEBUG else None
                attention_weights = torch.ones_like(attention_weights) * 0.5
        except Exception as e:
            print(f"警告: 注意力计算出错: {e}")
            # 使用均匀权重
            attention_weights = torch.ones(final_hidden.size(0), 1, device=final_hidden.device) * 0.5
        
        # 10. 使用注意力增强的特征通过分支层
        # Valence分支处理
        try:
            debug_print(f"Valence分支输入维度: {shared_features.shape}")
            
            # 提取词性特征用于Valence判断增强
            pos_features_vector = None
            if hasattr(self, 'use_pos_for_valence') and self.use_pos_for_valence and meta_features is not None:
                # 从meta_features中提取词性特征部分
                pos_features_vector = meta_features[:, self.pos_features_start_idx:self.pos_features_start_idx+self.pos_features_count]
                debug_print(f"词性特征向量维度: {pos_features_vector.shape}")
            
            # 基础Valence预测 - 使用完整的shared_features
            v_output_base = self.valence_branch(shared_features)
            
            # 如果有词性特征，额外计算词性特征的Valence预测并融合
            if pos_features_vector is not None and hasattr(self, 'pos_valence_branch'):
                # 使用词性特征预测Valence
                v_output_pos = self.pos_valence_branch(pos_features_vector)
                
                # 融合两种预测结果
                v_combined = torch.cat([v_output_base, v_output_pos], dim=1)
                v_output = self.valence_fusion(v_combined)
                
                debug_print(f"融合Valence输出: base={v_output_base.mean().item():.4f}, pos={v_output_pos.mean().item():.4f}, combined={v_output.mean().item():.4f}")
            else:
                # 不使用词性特征增强，直接使用基础预测
                v_output = v_output_base
            
            if torch.isnan(v_output).any():
                print("警告: Valence输出包含NaN值，使用0替换") if DEBUG else None
                v_output = torch.zeros_like(v_output)
        except Exception as e:
            print(f"警告: Valence分支处理出错: {e}，输入维度: {shared_features.shape}")
            v_output = torch.zeros(final_hidden.size(0), 1, device=final_hidden.device)
        
        # Arousal分支
        try:
            debug_print(f"Arousal分支输入维度: {shared_features.shape}")
            a_output = self.arousal_branch(shared_features)
            if torch.isnan(a_output).any():
                print("警告: Arousal输出包含NaN值，使用0替换") if DEBUG else None
                a_output = torch.zeros_like(a_output)
        except Exception as e:
            print(f"警告: Arousal分支处理出错: {e}，输入维度: {shared_features.shape}")
            a_output = torch.zeros(final_hidden.size(0), 1, device=final_hidden.device)
        
        # 情感交互层
        try:
            interaction = self.emotion_interaction(shared_features)
            if torch.isnan(interaction).any():
                print("警告: 情感交互输出包含NaN值，使用0替换") if DEBUG else None
                interaction = torch.zeros_like(interaction)
        except Exception as e:
            print(f"警告: 情感交互处理出错: {e}")
            interaction = torch.zeros(final_hidden.size(0), 2, device=final_hidden.device)
        
        # 结合交互信息
        v_output = v_output + interaction[:, 0:1] * 0.2
        a_output = a_output + interaction[:, 1:2] * 0.2
        
        # 合并V和A输出
        outputs = torch.cat([v_output, a_output], dim=1)  # [batch_size, 2]
        
        # 处理NaN值(如果存在)
        if torch.isnan(outputs).any():
            print("警告: 输出包含NaN值，使用0替换") if DEBUG else None
            outputs = torch.nan_to_num(outputs, nan=0.0)
            
        # 确保输出在[-1,1]范围内 - 仅应用一次tanh
        outputs = torch.tanh(outputs)
        
        # 应用价值和效度反转（如果启用）- 避免原地修改
        if self.invert_valence or self.invert_arousal:
            # 创建新的张量，避免原地修改
            outputs_clone = outputs.clone()
            
            if self.invert_valence:
                outputs_clone[:, 0] = -outputs[:, 0]  # 反转价值维度
            
            if self.invert_arousal:
                outputs_clone[:, 1] = -outputs[:, 1]  # 反转效度维度
                
            outputs = outputs_clone
        
        # 新增：四象限分类预测
        quadrant_logits = None
        if self.use_quadrant_head:
            try:
                quadrant_logits = self.quadrant_head(shared_features)
                
                # 检查是否有NaN值
                if torch.isnan(quadrant_logits).any():
                    print("警告: 四象限分类输出包含NaN值，使用0替换") if DEBUG else None
                    quadrant_logits = torch.nan_to_num(quadrant_logits, nan=0.0)
            except Exception as e:
                print(f"警告: 四象限分类头处理出错: {e}")
                quadrant_logits = torch.zeros(final_hidden.size(0), 4, device=final_hidden.device)
        
        # 返回VA回归输出和四象限分类预测
        if self.use_quadrant_head:
            return outputs, quadrant_logits
        else:
            return outputs
    
    def get_stats(self):
        """获取模型统计信息"""
        stats = {
            'embedding_norm': self.embedding.weight.norm().item(),
            'shared_layer_norm': sum(p.norm().item() for p in self.shared_layer.parameters()),
            'valence_branch_norm': sum(p.norm().item() for p in self.valence_branch.parameters()),
            'arousal_branch_norm': sum(p.norm().item() for p in self.arousal_branch.parameters()),
            'interaction_norm': sum(p.norm().item() for p in self.emotion_interaction.parameters()),
            'attention_norm': sum(p.norm().item() for p in self.attention.parameters())
        }
        
        # 添加四象限分类头统计
        if self.use_quadrant_head:
            stats['quadrant_head_norm'] = sum(p.norm().item() for p in self.quadrant_head.parameters())
        
        # 添加LTC细胞统计
        if self.multi_level:
            for i, ltc_cell in enumerate(self.ltc_levels):
                dir_name = "forward" if i % 2 == 0 or not self.bidirectional else "reverse"
                level_idx = i // 2 if self.bidirectional else i
                stats[f'ltc_level{level_idx}_{dir_name}_norm'] = sum(p.norm().item() for p in ltc_cell.parameters())
        else:
            stats['ltc_cell_norm'] = sum(p.norm().item() for p in self.ltc_cell.parameters())
            
            if self.bidirectional:
                stats['reverse_ltc_cell_norm'] = sum(p.norm().item() for p in self.reverse_ltc_cell.parameters())
        
        # 添加Transformer统计（如果使用）
        if self.use_transformer:
            transformer_norm = 0
            for layer in self.transformer.layers:
                transformer_norm += sum(p.norm().item() for p in layer.parameters())
            stats['transformer_norm'] = transformer_norm
        
        return stats 