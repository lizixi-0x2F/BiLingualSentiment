#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试融合门控实现 - 单独脚本

这个脚本创建一个模型实例，并直接测试融合门控的行为
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core import LTC_NCP_RNN

def main():
    """主函数"""
    print("=" * 50)
    print("  测试融合门控实现  ")
    print("=" * 50)
    
    # 创建输出目录
    output_dir = "results/fusion_gate_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    vocab_size = 1000
    embedding_dim = 64
    hidden_size = 128
    
    # 创建模型
    print("\n创建模型...")
    model = LTC_NCP_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=2,  # VA情感输出
        dropout=0.3,
        sparsity_level=0.5,
        dt=1.0,
        integration_method="euler",
        use_meta_features=True,
        bidirectional=True,
        padding_idx=0,
        wiring_type="structured",
        multi_level=False,
        emotion_focused=False,
        heterogeneous=False,
        use_transformer=True,  # 启用Transformer
        use_moe=False
    ).to(device)
    
    # 确认模型架构
    print("\n模型架构检查:")
    if hasattr(model, "fusion_gate_linear"):
        print("✓ 融合门控层已添加到模型")
        print(f"  融合门控层形状: 输入 {model.fusion_gate_linear.in_features}, 输出 {model.fusion_gate_linear.out_features}")
    else:
        print("✗ 模型中未找到融合门控层")
        return
    
    # 准备测试数据
    print("\n生成测试数据...")
    batch_size = 8
    seq_length = 20
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    lengths = torch.randint(5, seq_length + 1, (batch_size,), device=device)
    meta_features = torch.randn(batch_size, 3, device=device)  # 3个元特征
    
    # 准备钩子
    fusion_gate_values = []
    
    def hook_fn(module, input_tensor, output_tensor):
        # 捕获融合门控的输入值（在应用sigmoid之前）
        fusion_gate_values.append(output_tensor.detach().cpu())
    
    # 注册钩子
    hook = model.fusion_gate_linear.register_forward_hook(hook_fn)
    
    # 前向传播
    print("\n运行前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(tokens, lengths, meta_features)
    
    # 移除钩子
    hook.remove()
    
    # 收集结果
    print(f"\n模型输出形状: {outputs.shape}")
    
    # 处理融合门控值
    if fusion_gate_values:
        raw_gate_values = fusion_gate_values[0]  # 取第一个批次的值
        print(f"原始融合门控值形状 (线性层输出): {raw_gate_values.shape}")
        
        # 应用sigmoid函数获取实际的门控值
        gate_values = torch.sigmoid(raw_gate_values)
        
        # 统计信息
        gate_mean = gate_values.mean().item()
        gate_std = gate_values.std().item()
        gate_min = gate_values.min().item()
        gate_max = gate_values.max().item()
        
        print(f"\n融合门控统计:")
        print(f"  平均值: {gate_mean:.4f}")
        print(f"  标准差: {gate_std:.4f}")
        print(f"  最小值: {gate_min:.4f}")
        print(f"  最大值: {gate_max:.4f}")
        
        # 可视化
        if len(raw_gate_values.shape) == 3:  # [batch, seq_len, hidden]
            # 绘制不同批次的门控值分布
            plt.figure(figsize=(12, 8))
            for i in range(min(4, batch_size)):
                # 选择序列中间位置的门控值
                middle_pos = lengths[i].item() // 2
                if middle_pos >= seq_length:
                    middle_pos = seq_length - 1
                
                # 为每个批次绘制门控值的直方图
                plt.subplot(2, 2, i+1)
                gate_vals = gate_values[i, middle_pos].numpy()
                plt.hist(gate_vals, bins=30)
                plt.title(f'批次 {i+1} 门控值分布 (位置 {middle_pos})')
                plt.xlabel('门控值')
                plt.ylabel('频率')
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fusion_gate_distribution.png'))
            print(f"\n分布图已保存到 {os.path.join(output_dir, 'fusion_gate_distribution.png')}")
            
            # 绘制序列位置和门控值的关系
            plt.figure(figsize=(12, 6))
            sample_idx = 0
            seq_len = lengths[sample_idx].item()
            
            # 计算每个位置的门控值平均值
            position_means = []
            for pos in range(min(seq_len, seq_length)):
                position_means.append(gate_values[sample_idx, pos].mean().item())
            
            plt.plot(range(len(position_means)), position_means, marker='o')
            plt.title('融合门控平均值随序列位置的变化')
            plt.xlabel('序列位置')
            plt.ylabel('平均门控值')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'fusion_gate_by_position.png'))
            print(f"位置图已保存到 {os.path.join(output_dir, 'fusion_gate_by_position.png')}")
    else:
        print("\n✗ 未捕获到融合门控值")
    
    print("\n" + "="*50)
    print("  测试完成  ")
    print("="*50)

if __name__ == "__main__":
    main()
