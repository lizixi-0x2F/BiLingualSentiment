#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接测试XLM-R Transformer分支模块 - 独立脚本
验证层冻结是否正确实现
"""

import os
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XLMRTransformerBranch(nn.Module):
    """
    XLM-RoBERTa编码器分支，前6层冻结，后6层可训练
    
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
        """前向传播"""
        # 输入参数校验
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
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

def test_transformer_branch():
    """测试XLM-R Transformer分支"""
    print("=" * 50)
    print("开始测试XLM-R Transformer分支...")
    print("=" * 50)
    
    # 创建模型
    model = XLMRTransformerBranch(output_dim=512, freeze_layers=6)
    
    # 验证各层的冻结状态
    print("\n检查每一层的冻结状态:")
    
    # 检查嵌入层
    emb_params = list(model.encoder.embeddings.parameters())
    emb_frozen = all(not p.requires_grad for p in emb_params)
    print(f"嵌入层: {'已冻结' if emb_frozen else '可训练'}")
    
    # 检查编码器层
    for i, layer in enumerate(model.encoder.encoder.layer):
        layer_params = list(layer.parameters())
        layer_frozen = all(not p.requires_grad for p in layer_params)
        frozen_status = "已冻结" if layer_frozen else "可训练"
        expected = (i < 6 and layer_frozen) or (i >= 6 and not layer_frozen)
        print(f"第 {i} 层: {frozen_status} {'✓' if expected else '✗'}")
    
    # 统计冻结和可训练参数
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    
    print("\n参数统计:")
    print(f"冻结参数:   {frozen_params:,d} ({frozen_params/total_params:.1%})")
    print(f"可训练参数: {trainable_params:,d} ({trainable_params/total_params:.1%})")
    print(f"总参数:     {total_params:,d}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    try:
        outputs = model(input_ids, attention_mask)
        print("\n前向传播测试成功!")
        print(f"输出张量形状: {outputs.shape}")
        validation_success = True
    except Exception as e:
        print(f"\n前向传播测试失败: {str(e)}")
        validation_success = False
        
    # 验证冻结和非冻结层的要求
    frozen_layers_correct = all(not param.requires_grad for layer in model.encoder.encoder.layer[:6] 
                              for param in layer.parameters())
    trainable_layers_correct = all(param.requires_grad for layer in model.encoder.encoder.layer[6:] 
                                 for param in layer.parameters())
    
    print("\n验证结果:")
    print(f"前6层已冻结: {'通过 ✓' if frozen_layers_correct else '失败 ✗'}")
    print(f"后6层可训练: {'通过 ✓' if trainable_layers_correct else '失败 ✗'}")
    print(f"前向传播测试: {'通过 ✓' if validation_success else '失败 ✗'}")
    
    overall_success = frozen_layers_correct and trainable_layers_correct and validation_success
    print(f"\n总体结果: {'成功 ✓' if overall_success else '失败 ✗'}")
    print("=" * 50)

if __name__ == "__main__":
    test_transformer_branch()
