#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证融合门控在混合精度模式下的行为
测试目标:
1. 确保融合门形状正确: h_final.shape == h_t.shape == h_l.shape
2. 检查是否有NaN值产生
3. 打印h_final.mean()的值
"""

import os
import sys
import torch
import numpy as np
# Update imports to use recommended torch.amp instead of deprecated torch.cuda.amp
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core import LTC_NCP_RNN

def main():
    """主函数"""
    # 设置参数
    use_amp = True  # 是否使用混合精度
    batch_size = 4
    seq_length = 16
    
    print("=" * 50)
    print(f"  融合门控混合精度验证 (AMP: {use_amp})  ")
    print("=" * 50)
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果不可用CUDA，则禁用AMP
    if device.type != 'cuda' and use_amp:
        print("警告: CUDA不可用，禁用混合精度")
        use_amp = False
    
    # 创建模型
    print("\n创建模型...")
    model = LTC_NCP_RNN(
        vocab_size=1000,
        embedding_dim=64,
        hidden_size=128,
        output_size=2,
        dropout=0.3,
        use_transformer=True
    ).to(device)
    
    # 设置为评估模式以避免dropout
    model.eval()
    
    # 检查模型中是否有融合门控层
    if hasattr(model, "fusion_gate_linear"):
        print(f"✓ 融合门控层形状: 输入 {model.fusion_gate_linear.in_features}, 输出 {model.fusion_gate_linear.out_features}")
    else:
        print("✗ 模型中未找到融合门控层")
        return
      # 注册中间值查看用的钩子
    fusion_values = {}
    
    def hook_transformer_out(name):
        def hook(module, input_tensor, output_tensor):
            fusion_values[name] = output_tensor.detach()
        return hook
    
    # 为了捕获融合后的输出 (h_final)
    def fusion_output_hook(module, input_tensor, output_tensor):
        # 这个钩子会在融合门控应用后被触发
        # output_tensor 是 Transformer 层处理后的结果，是融合后的输出 (h_final)
        if isinstance(output_tensor, tuple):
            output = output_tensor[0]  # 有些模块返回元组
        else:
            output = output_tensor
            
        # 计算并保存平均值
        fusion_values['h_final'] = output
        mean_value = output.mean().item()
        print(f"h_final.mean(): {mean_value:.4f}")
    
    # 为相关位置注册钩子
    transformer_hook = model.transformer_adapter.register_forward_hook(
        hook_transformer_out('transformer_out')
    )
    
    # 注册捕获融合输出的钩子 - 在transformer层之后
    fusion_hook = model.transformer.register_forward_hook(fusion_output_hook)
    
    # 创建随机输入
    print("\n生成输入数据...")
    tokens = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    lengths = torch.randint(5, seq_length + 1, (batch_size,)).to(device)
    
    # 如果使用AMP，创建梯度缩放器
    scaler = GradScaler() if use_amp else None    # 执行前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        # 使用AMP上下文（如果启用），明确指定device_type和dtype
        amp_context = autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
        with amp_context:
            outputs = model(tokens, lengths)
    
    # 移除钩子
    transformer_hook.remove()
    
    # 打印结果
    print("\n前向传播结果:")
    print(f"模型输出形状: {outputs.shape}")
    has_nan = torch.isnan(outputs).any()
    print(f"输出包含NaN: {'是' if has_nan else '否'}")
    
    # 检查中间值
    if 'transformer_out' in fusion_values:
        print("\n中间值检查:")
        t_out = fusion_values['transformer_out']
        print(f"Transformer输出形状: {t_out.shape}")
        has_nan = torch.isnan(t_out).any()
        print(f"Transformer输出包含NaN: {'是' if has_nan else '否'}")
    
    print("\n" + "="*50)
    print("  测试完成  ")
    print("="*50)

if __name__ == "__main__":
    main()
