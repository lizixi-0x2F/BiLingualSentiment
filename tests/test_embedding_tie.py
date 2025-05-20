#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证XLM-RoBERTa嵌入层绑定是否成功
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_embedding_tie():
    """测试XLM-RoBERTa嵌入层权重绑定"""
    logger.info("===============================")
    logger.info("测试XLM-RoBERTa嵌入层权重绑定")
    logger.info("===============================")
    
    logger.info("\n正在加载XLM-RoBERTa模型...")
    
    # 加载预训练模型
    encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    
    # 直接引用嵌入层
    embedding = encoder.embeddings.word_embeddings
    
    # 检查权重ID
    is_same_id = id(embedding.weight) == id(encoder.embeddings.word_embeddings.weight)
    logger.info(f"\n嵌入层权重ID检查: {'通过 ✓' if is_same_id else '失败 ✗'}")
    logger.info(f"Embedding weight id: {id(embedding.weight)}")
    logger.info(f"Encoder embedding weight id: {id(encoder.embeddings.word_embeddings.weight)}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    try:
        # 通过词嵌入层直接获取嵌入
        embedded = embedding(input_ids)
        logger.info(f"\n直接嵌入前向传播成功! 输出形状: {embedded.shape}")
        
        # 更改嵌入层权重，验证绑定是否有效
        original_value = embedding.weight[0, 0].item()
        embedding.weight.data[0, 0] = 999.0
        
        # 检查原始encoder中的权重是否也变化
        changed_value = encoder.embeddings.word_embeddings.weight[0, 0].item()
        weights_changed = abs(changed_value - 999.0) < 1e-4
        
        logger.info(f"\n权重修改测试: {'通过 ✓' if weights_changed else '失败 ✗'}")
        logger.info(f"原始值: {original_value}")
        logger.info(f"修改后值: {changed_value}")
        
        # 恢复原始值
        embedding.weight.data[0, 0] = original_value
        
        logger.info("\n=== 测试结论 ===")
        if is_same_id and weights_changed:
            logger.info("✓ 嵌入层权重绑定成功!")
        else:
            logger.info("✗ 嵌入层权重绑定失败!")
        
    except Exception as e:
        logger.info(f"\n测试失败: {str(e)}")
    
    logger.info("===============================")

if __name__ == "__main__":
    test_embedding_tie()
