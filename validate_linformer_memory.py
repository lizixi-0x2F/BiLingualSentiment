#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存效率验证脚本
比较标准Transformer与Linformer Transformer的内存使用情况
"""

import os
import sys
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import time
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入必要模块
from src.core.model import TransformerEncoderLayer, MiniTransformer
from src.core.linformer_transformer import LinformerEncoderLayer

def format_memory(bytes):
    """将字节转换为可读格式"""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes/(1024*1024):.2f} MB"
    else:
        return f"{bytes/(1024*1024*1024):.2f} GB"

def measure_memory_usage(model, input_tensor, backward=True):
    """测量模型前向和后向传播的内存使用情况"""
    # 清理缓存以准确测量
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 记录初始内存
    start_memory = torch.cuda.memory_allocated()
    
    # 前向传播
    if backward:
        output = model(input_tensor)
        
        # 计算损失并反向传播
        if isinstance(output, tuple):
            output = output[0]
        
        loss = output.sum()
        loss.backward()
    else:
        with torch.no_grad():
            model(input_tensor)
    
    # 记录峰值内存
    peak_memory = torch.cuda.max_memory_allocated()
    
    # 清理
    torch.cuda.empty_cache()
    
    return peak_memory - start_memory

def compare_models(seq_lengths, batch_size=8, d_model=512, use_amp=False):
    """比较不同序列长度下的内存使用情况"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("警告: CUDA不可用，无法准确测量GPU内存使用情况")
        return
    
    results = {
        'standard': [],
        'linformer': [],
        'seq_lengths': seq_lengths
    }
    
    # 设置投影维度为序列长度的25%，确保至少为64
    for seq_len in seq_lengths:
        print(f"\n--- 测试序列长度: {seq_len} ---")
        
        # 创建随机输入数据
        input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # 计算Linformer投影维度
        projection_dim = max(64, int(seq_len * 0.25))
        print(f"Linformer投影维度: {projection_dim}")
        
        # 创建标准Transformer层
        standard_transformer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=d_model*4,
            dropout=0.1
        ).to(device)
        
        # 创建Linformer层
        linformer_transformer = LinformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=d_model*4, 
            dropout=0.1,
            projection_dim=projection_dim
        ).to(device)
        
        # 设置AMP上下文
        amp_context = autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
        
        # 测量标准Transformer内存使用
        with amp_context:
            standard_memory = measure_memory_usage(standard_transformer, input_tensor)
            
        print(f"标准Transformer内存使用: {format_memory(standard_memory)}")
        
        # 测量Linformer内存使用
        with amp_context:
            linformer_memory = measure_memory_usage(linformer_transformer, input_tensor)
            
        print(f"Linformer内存使用: {format_memory(linformer_memory)}")
        
        # 计算节省百分比
        savings = (standard_memory - linformer_memory) / standard_memory * 100
        print(f"内存节省: {savings:.2f}%")
        
        results['standard'].append(standard_memory)
        results['linformer'].append(linformer_memory)
    
    return results

def plot_results(results):
    """绘制内存使用对比图"""
    plt.figure(figsize=(10, 6))
    
    # 转换为MB
    standard_mb = [mem / (1024 * 1024) for mem in results['standard']]
    linformer_mb = [mem / (1024 * 1024) for mem in results['linformer']]
    
    # 计算节省百分比
    savings = [(s - l) / s * 100 for s, l in zip(results['standard'], results['linformer'])]
    
    # 绘制条形图
    x = np.arange(len(results['seq_lengths']))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制内存使用
    rects1 = ax1.bar(x - width/2, standard_mb, width, label='标准Transformer', color='royalblue', alpha=0.8)
    rects2 = ax1.bar(x + width/2, linformer_mb, width, label='Linformer', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('内存使用 (MB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results['seq_lengths'])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加第二个Y轴显示节省百分比
    ax2 = ax1.twinx()
    ax2.plot(x, savings, 'ro-', label='节省百分比', linewidth=2)
    ax2.set_ylabel('内存节省 (%)')
    ax2.legend(loc='upper right')
    
    # 添加标题和标签
    fig.tight_layout()
    plt.title('Linformer vs 标准Transformer的内存使用对比')
    
    # 在条形顶部添加数值
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}MB',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_labels(rects1)
    add_labels(rects2)
    
    # 为节省百分比添加标签
    for i, v in enumerate(savings):
        ax2.annotate(f'{v:.1f}%', 
                    xy=(i, v),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.savefig('memory_comparison.png')
    print(f"结果图表已保存至 {os.path.join(os.getcwd(), 'memory_comparison.png')}")
    
    return fig

def main():
    """主函数"""
    print("=" * 60)
    print("  Linformer vs 标准Transformer内存效率测试  ")
    print("=" * 60)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法测量GPU内存使用情况")
        return
    
    # 打印GPU信息
    device_name = torch.cuda.get_device_name(0)
    print(f"使用GPU: {device_name}")
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512, 1024]
    batch_size = 16
    d_model = 512
    use_amp = False
    
    print(f"\n配置:")
    print(f"- 批量大小: {batch_size}")
    print(f"- 模型维度: {d_model}")
    print(f"- 使用混合精度: {'是' if use_amp else '否'}")
    
    # 运行对比测试
    results = compare_models(
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        d_model=d_model,
        use_amp=use_amp
    )
    
    # 绘制结果
    if results:
        fig = plot_results(results)
        
        # 生成总结
        for i, seq_len in enumerate(seq_lengths):
            standard_mem = results['standard'][i]
            linformer_mem = results['linformer'][i]
            savings = (standard_mem - linformer_mem) / standard_mem * 100
            
            print(f"\n序列长度 {seq_len}:")
            print(f"- 标准Transformer: {format_memory(standard_mem)}")
            print(f"- Linformer: {format_memory(linformer_mem)}")
            print(f"- 节省: {savings:.2f}%")
        
        # 检查是否达到目标
        target_seq_len = 512
        target_idx = seq_lengths.index(target_seq_len) if target_seq_len in seq_lengths else -1
        
        if target_idx >= 0:
            target_savings = (results['standard'][target_idx] - results['linformer'][target_idx]) / results['standard'][target_idx] * 100
            print(f"\n目标检查 (序列长度={target_seq_len}):")
            print(f"- 内存节省: {target_savings:.2f}%")
            if target_savings >= 40:
                print("✓ 达成目标: 内存减少 ≥40%")
            else:
                print(f"✗ 未达成目标: 内存减少 {target_savings:.2f}% < 40%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
