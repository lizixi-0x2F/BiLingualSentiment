#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer分支模块: XLM-RoBERTa预训练模型
用于情感分析的多语言Transformer编码器，前6层冻结，后6层可训练
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('transformer_branch')

class XLMRTransformerBranch(nn.Module):
    """
    XLM-RoBERTa编码器分支，前6层冻结，后6层可训练
    包含与预训练模型共享的嵌入层
    
    参数：
        output_dim: 输出维度
        dropout: Dropout比例
        use_pooler: 是否使用RoBERTa的pooler输出
        freeze_layers: 指定要冻结的层数量（前N层）
    """
    def __init__(self, output_dim=768, dropout=0.1, use_pooler=False, freeze_layers=6):
        super(XLMRTransformerBranch, self).__init__()
        
        # 加载预训练模型
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.use_pooler = use_pooler
        
        # 获取编码器隐藏层维度
        self.hidden_size = self.encoder.config.hidden_size
        
        # 重要: 直接引用XLM-R的嵌入层，确保权重共享
        self.embedding = self.encoder.embeddings.word_embeddings
        
        # 冻结前N层
        self._freeze_layers(freeze_layers)
        
        # 输出投影层（如果需要改变维度）
        if output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 记录模型信息
        self._log_model_info()
        
        # 验证嵌入层权重共享
        logger.info(f"嵌入层权重ID检查: {id(self.embedding.weight) == id(self.encoder.embeddings.word_embeddings.weight)}")
        logger.info(f"Embedding weight id: {id(self.embedding.weight)}")
        logger.info(f"Encoder embedding weight id: {id(self.encoder.embeddings.word_embeddings.weight)}")
    
    def _freeze_layers(self, num_layers):
        """冻结前num_layers层参数"""
        if num_layers <= 0:
            return
        
        # 冻结嵌入层
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结指定层数的编码器层
        for i in range(min(num_layers, len(self.encoder.encoder.layer))):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        
        logger.info(f"已冻结嵌入层和前{num_layers}层Transformer层")
    
    def _log_model_info(self):
        """打印模型信息，包括冻结与非冻结参数数量"""
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = frozen_params + trainable_params
        
        logger.info(f"XLM-RoBERTa模型信息:")
        logger.info(f"  冻结参数:   {frozen_params:,d} ({frozen_params/total_params:.1%})")
        logger.info(f"  可训练参数: {trainable_params:,d} ({trainable_params/total_params:.1%})")
        logger.info(f"  总参数:     {total_params:,d}")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        """
        前向传播
        
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            position_ids: 位置编码ID
            
        返回:
            sequence_output 或 pooled_output
        """
        # 输入参数校验
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 可以直接使用self.embedding获取嵌入表示
        # 但为了保持完整的模型流程，使用encoder处理完整序列
        
        # 调用XLM-RoBERTa编码器
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 选择输出类型
        if self.use_pooler:
            # 使用[CLS]的池化输出
            output = outputs.pooler_output
        else:
            # 使用最后一层的序列输出
            output = outputs.last_hidden_state
        
        # 应用dropout和投影
        output = self.dropout(output)
        output = self.projection(output)
        
        return output

# 调试函数：验证模型参数冻结情况
def debug_transformer_branch():
    """创建模型实例并验证参数冻结情况"""
    model = XLMRTransformerBranch(output_dim=512, freeze_layers=6)
    
    # 统计每一层的参数情况
    layer_info = []
    
    # 检查嵌入层
    emb_frozen = sum(1 for p in model.encoder.embeddings.parameters() if not p.requires_grad) == \
                 sum(1 for _ in model.encoder.embeddings.parameters())
    layer_info.append(("嵌入层", "已冻结" if emb_frozen else "可训练"))
    
    # 检查每一个编码器层
    for i, layer in enumerate(model.encoder.encoder.layer):
        frozen = sum(1 for p in layer.parameters() if not p.requires_grad) == \
                 sum(1 for _ in layer.parameters())
        layer_info.append((f"层 {i}", "已冻结" if frozen else "可训练"))
    
    # 检查输出投影层
    proj_frozen = sum(1 for p in model.projection.parameters() if not p.requires_grad) == \
                 sum(1 for _ in model.projection.parameters())
    layer_info.append(("输出投影层", "已冻结" if proj_frozen else "可训练"))
    
    # 打印详细信息
    print("模型层状态:")
    for layer_name, status in layer_info:
        print(f"  {layer_name:<15}: {status}")
    
    # 创建一个小批量数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # 测试前向传播
    outputs = model(input_ids, attention_mask)
    
    print(f"\n前向传播测试:")
    if model.use_pooler:
        print(f"  输出形状: {outputs.shape} (批量大小, 输出维度)")
    else:
        print(f"  输出形状: {outputs.shape} (批量大小, 序列长度, 输出维度)")
    
    return model

if __name__ == "__main__":
    debug_transformer_branch()
