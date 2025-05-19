#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证脚本：检查XLMRTransformerBranch的嵌入层权重是否与编码器嵌入层权重绑定
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import XLMRobertaModel

# 添加项目路径到sys.path确保可以导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入transformer_branch或直接复制类定义到此处
try:
    from src.core.transformer_branch import XLMRTransformerBranch
except ImportError:
    print("无法导入XLMRTransformerBranch，将使用内联定义")
    
    # 如果导入失败，使用内联定义
    class XLMRTransformerBranch(nn.Module):
        """
        XLM-RoBERTa编码器分支，前6层冻结，后6层可训练
        包含与预训练模型共享的嵌入层
        """
        def __init__(self, output_dim=768, dropout=0.1, use_pooler=False, freeze_layers=6):
            super(XLMRTransformerBranch, self).__init__()
            
            # 加载预训练模型
            print("正在加载XLM-R预训练模型...")
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
            
            print("模型初始化完成！")
        
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
            
            print(f"已冻结嵌入层和前{num_layers}层Transformer层")
        
        def forward(self, input_ids, attention_mask=None):
            """前向传播"""
            # 输入参数校验
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            # 调用XLM-RoBERTa编码器
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # 选择输出类型
            if self.use_pooler:
                output = outputs.pooler_output
            else:
                output = outputs.last_hidden_state
            
            # 应用dropout和投影
            output = self.dropout(output)
            output = self.projection(output)
            
            return output

def validate_embedding_tie():
    """验证嵌入层权重绑定"""
    print("=" * 50)
    print("开始验证嵌入层权重绑定...")
    print("=" * 50)
    
    # 创建模型
    model = XLMRTransformerBranch(output_dim=512)
    
    # 检查嵌入层权重是否绑定
    embedding_id = id(model.embedding.weight)
    encoder_embedding_id = id(model.encoder.embeddings.word_embeddings.weight)
    
    print(f"\n嵌入层权重ID: {embedding_id}")
    print(f"编码器嵌入层权重ID: {encoder_embedding_id}")
    
    if embedding_id == encoder_embedding_id:
        print("\n✓ 成功: 嵌入层权重已成功绑定!")
    else:
        print("\n✗ 失败: 嵌入层权重未绑定!")
    
    # 测试forward
    print("\n测试前向传播...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    try:
        outputs = model(input_ids, attention_mask)
        print(f"✓ 前向传播成功! 输出形状: {outputs.shape}")
        
        # 验证输出形状
        expected_shape = (batch_size, seq_len, 512)  # 512是我们设置的output_dim
        if outputs.shape == expected_shape:
            print(f"✓ 输出形状正确: {outputs.shape}")
        else:
            print(f"✗ 输出形状不正确: 期望 {expected_shape}, 得到 {outputs.shape}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {str(e)}")
    
    print("\n验证完成!")
    print("=" * 50)

if __name__ == "__main__":
    validate_embedding_tie()
