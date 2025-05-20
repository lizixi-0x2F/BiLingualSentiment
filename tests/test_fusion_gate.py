#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试融合门控实现 - 简单版本

这个脚本创建一个小批量数据并验证融合门控是否正常工作，打印门控值和形状信息
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core import LTC_NCP_RNN

def test_fusion_gate():
    """测试融合门控实现"""
    print("=" * 80)
    print("测试融合门控实现")
    print("=" * 80)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    vocab_size = 1000
    embedding_dim = 64
    hidden_size = 128
    output_size = 2  # Valence, Arousal
    
    # 创建模型
    print("创建模型...")
    model = LTC_NCP_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=0.3,
        sparsity_level=0.5,
        dt=1.0,
        integration_method="euler",
        use_meta_features=True,
        bidirectional=False,
        padding_idx=0,
        wiring_type="structured",
        multi_level=False,
        emotion_focused=False,
        heterogeneous=False,
        use_transformer=True,  # 启用Transformer
        use_moe=False,
        invert_valence=False,
        invert_arousal=False,
        enhance_valence=False,
        valence_layers=1,
        use_quadrant_head=False,
        quadrant_weight=0.0
    )
    
    # 确认融合门已添加
    if hasattr(model, 'fusion_gate_linear'):
        print("✓ 融合门控已成功添加到模型")
    else:
        print("✗ 模型中未找到融合门控")
        return
    
    # 将模型移到设备
    model.to(device)
    model.eval()
      # 创建用于收集融合门控值的变量和钩子
    fusion_gate_values = []
    
    # 定义一个自定义钩子，用于捕获融合门控值
    def capture_fusion_gate(module, input_tensor, output_tensor):
        # 注意：这会捕获sigmoid应用前的原始输出
        # 我们需要在测试中手动应用sigmoid
        fusion_gate_values.append(output_tensor.detach().cpu())
    
    # 定制前向方法以获取融合门控值
    original_forward = model.forward
    
    def forward_with_gate_capture(*args, **kwargs):
        # 调用原始forward方法
        result = original_forward(*args, **kwargs)
        # 在这里，融合门控值将通过钩子捕获
        return result
    
    # 临时替换forward方法
    model.forward = forward_with_gate_capture
    
    # 注册钩子到fusion_gate_linear层
    hook = model.fusion_gate_linear.register_forward_hook(capture_fusion_gate)
    
    # 创建测试批次
    batch_size = 4
    seq_length = 10
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    lengths = torch.randint(5, seq_length + 1, (batch_size,), device=device)
    meta_features = torch.randn(batch_size, 3, device=device)  # 3个元特征
    
    # 运行模型前向传递
    print("运行模型前向传递...")
    with torch.no_grad():
        outputs = model(tokens, lengths, meta_features)
      # 移除钩子并恢复原始forward方法
    hook.remove()
    model.forward = original_forward
    
    # 检查输出
    print(f"模型输出形状: {outputs.shape}")
      # 分析融合门控值
    if fusion_gate_values:
        fusion_gate = fusion_gate_values[0]  # 取第一个（也是唯一一个）批次
        print(f"融合门控形状: {fusion_gate.shape}")
        
        # 基本统计信息
        gate_mean = fusion_gate.mean().item()
        gate_min = fusion_gate.min().item()
        gate_max = fusion_gate.max().item()
        gate_std = fusion_gate.std().item()
        
        print(f"融合门控统计: 平均值={gate_mean:.4f}, 最小值={gate_min:.4f}, 最大值={gate_max:.4f}, 标准差={gate_std:.4f}")
        
        # 简单可视化
        plt.figure(figsize=(10, 6))
        
        # 检查张量维度并进行适当的可视化
        if len(fusion_gate.shape) == 2:
            # 如果是二维张量 [batch*seq_len, hidden_size]
            # 选择一些特征维度可视化
            dims_to_plot = min(5, fusion_gate.shape[1])
            for i in range(dims_to_plot):
                values = fusion_gate[:, i].numpy()
                plt.plot(range(len(values)), values, label=f'维度 {i}')
        elif len(fusion_gate.shape) == 3:
            # 如果是三维张量 [batch, seq_len, hidden_size]
            sample_idx = 0
            dims_to_plot = min(5, fusion_gate.shape[2])
            for i in range(dims_to_plot):
                plt.plot(fusion_gate[sample_idx, :, i].numpy(), label=f'维度 {i}')
        
        plt.title('融合门控值随序列位置的变化')
        plt.xlabel('序列位置')
        plt.ylabel('门控值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plt.savefig('fusion_gate_test_visualization.png')
        plt.close()
        print("可视化已保存为 'fusion_gate_test_visualization.png'")
    else:
        print("未捕获到融合门控值")
    
    print("=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_fusion_gate()
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
