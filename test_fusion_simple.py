#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版融合门控测试脚本
"""

import os
import sys
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    print("正在尝试导入LTC_NCP_RNN...")
    from src.core import LTC_NCP_RNN
    print("导入成功！")

    # 创建模型
    model = LTC_NCP_RNN(
        vocab_size=1000,
        embedding_dim=64,
        hidden_size=128,
        output_size=2,
        dropout=0.3,
        use_transformer=True
    )
    
    # 检查模型中是否有融合门控层
    if hasattr(model, "fusion_gate_linear"):
        print("融合门控层检测: 成功")
        print(f"融合门控层形状: 输入 {model.fusion_gate_linear.in_features}, 输出 {model.fusion_gate_linear.out_features}")
    else:
        print("融合门控层检测: 失败")
    
    # 尝试简单前向传播
    batch_size = 2
    seq_length = 5
    tokens = torch.randint(0, 1000, (batch_size, seq_length))
    lengths = torch.ones(batch_size) * seq_length
    
    print("尝试前向传播...")
    outputs = model(tokens, lengths)
    print(f"输出形状: {outputs.shape}")
    print("前向传播成功!")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
