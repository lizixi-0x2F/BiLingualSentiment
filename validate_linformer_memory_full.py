#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全面验证Linformer与标准Transformer在序列长度为512时的内存使用情况
并确保前向和后向传播正确工作
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import argparse
import time
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入必要模块
from src.core import LTC_NCP_RNN, LinformerMiniTransformer
from src.core.model import MiniTransformer, TransformerEncoderLayer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('linformer_validation')

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

def memory_report():
    """生成内存使用报告"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法获取内存信息")
        return {}
    
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    
    logger.info(f"当前内存使用:")
    logger.info(f"  已分配: {format_memory(allocated)} (峰值: {format_memory(max_allocated)})")
    logger.info(f"  已保留: {format_memory(reserved)} (峰值: {format_memory(max_reserved)})")
    
    return {
        "allocated": allocated,
        "max_allocated": max_allocated,
        "reserved": reserved,
        "max_reserved": max_reserved
    }

def create_standard_model(seq_length=512, batch_size=4):
    """创建使用标准Transformer的模型"""
    standard_model = LTC_NCP_RNN(
        vocab_size=1000,
        embedding_dim=64,
        hidden_size=128,
        output_size=2,
        dropout=0.3,
        use_transformer=True
    )
    
    # 确保使用的是标准MiniTransformer
    d_model = standard_model.transformer.d_model
    nhead = len(standard_model.transformer.layers) > 0 and hasattr(standard_model.transformer.layers[0], 'self_attn') and getattr(standard_model.transformer.layers[0].self_attn, 'num_heads', 4)
    num_layers = len(standard_model.transformer.layers)
    dim_feedforward = standard_model.transformer.layers[0].linear1.out_features if len(standard_model.transformer.layers) > 0 else d_model * 2
    dropout = standard_model.transformer.dropout.p
    
    # 创建一个新的标准MiniTransformer
    standard_transformer = MiniTransformer(
        d_model=d_model,
        nhead=nhead, 
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 替换模型中的transformer
    standard_model.transformer = standard_transformer
    
    return standard_model

def create_linformer_model(seq_length=512, batch_size=4, projection_dim=None):
    """创建使用LinformerMiniTransformer的模型"""
    linformer_model = LTC_NCP_RNN(
        vocab_size=1000,
        embedding_dim=64,
        hidden_size=128,
        output_size=2,
        dropout=0.3,
        use_transformer=True
    )
    
    # 确定投影维度
    if projection_dim is None:
        projection_dim = max(64, int(seq_length * 0.25))
    
    # 确定模型参数
    d_model = linformer_model.transformer.d_model
    nhead = len(linformer_model.transformer.layers) > 0 and hasattr(linformer_model.transformer.layers[0], 'self_attn') and getattr(linformer_model.transformer.layers[0].self_attn, 'num_heads', 4)
    num_layers = len(linformer_model.transformer.layers)
    dim_feedforward = linformer_model.transformer.layers[0].linear1.out_features if len(linformer_model.transformer.layers) > 0 else d_model * 2
    dropout = linformer_model.transformer.dropout.p
    
    # 创建一个新的LinformerMiniTransformer
    linformer_transformer = LinformerMiniTransformer(
        d_model=d_model,
        nhead=nhead, 
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        projection_dim=projection_dim
    )
    
    # 替换模型中的transformer
    linformer_model.transformer = linformer_transformer
    
    return linformer_model

def validate_model_memory(model_type, model, seq_length=512, batch_size=4, use_amp=False):
    """验证模型的内存使用情况并测试前向和后向传播"""
    logger.info(f"\n--- 验证{model_type}模型 ---")
    
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，跳过内存测量")
        return None
    
    # 准备设备和输入
    device = torch.device("cuda")
    model = model.to(device)
    
    # 重置内存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 准备输入数据
    tokens = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    lengths = torch.randint(5, seq_length + 1, (batch_size,), device=device)
    
    # AMP设置
    scaler = GradScaler() if use_amp else None
    amp_context = autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    
    # 1. 验证前向传播
    try:
        model.train()  # 设置为训练模式
        
        # 内存使用前
        before_forward = torch.cuda.memory_allocated()
        
        # 前向传播
        with amp_context:
            outputs = model(tokens, lengths)
        
        # 内存使用后
        after_forward = torch.cuda.memory_allocated()
        
        # 计算使用的内存
        forward_memory = after_forward - before_forward
        logger.info(f"前向传播内存使用: {format_memory(forward_memory)}")
        
        # 2. 验证后向传播
        loss = outputs.sum()  # 简单损失函数
        
        # 内存使用前
        before_backward = torch.cuda.memory_allocated()
        
        # 后向传播
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 内存使用后
        after_backward = torch.cuda.memory_allocated()
        
        # 计算使用的内存
        backward_memory = after_backward - before_backward
        logger.info(f"后向传播内存使用: {format_memory(backward_memory)}")
        
        # 总内存使用
        peak_memory = torch.cuda.max_memory_allocated()
        logger.info(f"峰值内存使用: {format_memory(peak_memory)}")
        
        logger.info(f"✓ {model_type}模型验证通过！")
        
        # 记录完整的内存报告
        memory_stats = memory_report()
        
        return {
            "success": True,
            "forward_memory": forward_memory,
            "backward_memory": backward_memory,
            "peak_memory": peak_memory,
            "memory_stats": memory_stats
        }
    
    except Exception as e:
        logger.error(f"✗ {model_type}模型验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }

def plot_memory_comparison(standard_results, linformer_results, seq_length=512, save_path="memory_comparison.png"):
    """绘制内存使用对比图"""
    if standard_results is None or linformer_results is None:
        logger.error("无法生成对比图：模型验证结果不完整")
        return
    
    if not standard_results["success"] or not linformer_results["success"]:
        logger.error("无法生成对比图：模型验证失败")
        return
    
    # 提取内存使用数据 (转换为MB)
    to_mb = lambda x: x / (1024 * 1024)
    
    std_forward = to_mb(standard_results["forward_memory"])
    std_backward = to_mb(standard_results["backward_memory"])
    std_peak = to_mb(standard_results["peak_memory"])
    
    lin_forward = to_mb(linformer_results["forward_memory"])
    lin_backward = to_mb(linformer_results["backward_memory"])
    lin_peak = to_mb(linformer_results["peak_memory"])
    
    # 计算节省百分比
    forward_savings = (std_forward - lin_forward) / std_forward * 100
    backward_savings = (std_backward - lin_backward) / std_backward * 100
    peak_savings = (std_peak - lin_peak) / std_peak * 100
    
    # 绘图设置
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(3)
    
    # 绘制条形图
    standard_bars = ax.bar(index, [std_forward, std_backward, std_peak], bar_width,
                         label='标准Transformer', color='royalblue', alpha=0.8)
    linformer_bars = ax.bar(index + bar_width, [lin_forward, lin_backward, lin_peak], bar_width,
                          label='Linformer', color='lightgreen', alpha=0.8)
    
    # 添加节省百分比标签
    for i, saving in enumerate([forward_savings, backward_savings, peak_savings]):
        ax.annotate(f'-{saving:.1f}%', 
                   xy=(i + bar_width/2, lin_forward if i == 0 else lin_backward if i == 1 else lin_peak + 5),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center', va='bottom', color='green', fontweight='bold')
    
    # 添加图表元素
    ax.set_xlabel('内存使用类型')
    ax.set_ylabel('内存使用 (MB)')
    ax.set_title(f'序列长度{seq_length}的内存使用对比')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['前向传播', '后向传播', '峰值使用'])
    ax.legend()
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加总结信息
    summary_text = f"序列长度: {seq_length}\n"
    summary_text += f"峰值内存节省: {peak_savings:.1f}%\n"
    summary_text += f"Linformer投影维度: {linformer_results.get('projection_dim', int(seq_length * 0.25))}"
    
    # 标注目标线
    if peak_savings >= 40:
        target_text = f"✓ 达到目标: ≥40% 内存减少"
        ax.text(0.98, 0.05, target_text, transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.1),
               color='green', fontsize=12, fontweight='bold')
    else:
        target_text = f"✗ 未达到目标: <40% 内存减少"
        ax.text(0.98, 0.05, target_text, transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.1),
               color='red', fontsize=12, fontweight='bold')
    
    # 添加注释框
    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.1),
           fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"对比图已保存至: {save_path}")
    
    return fig

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证Linformer内存效率')
    parser.add_argument('--seq-length', type=int, default=512, help='序列长度 (默认: 512)')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小 (默认: 4)')
    parser.add_argument('--projection-dim', type=int, default=None, 
                       help='Linformer投影维度 (默认: seq_length * 0.25, 最小64)')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度')
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("错误: CUDA不可用，无法进行内存验证")
        return
    
    # 打印硬件信息
    logger.info("=" * 60)
    logger.info("  Linformer内存效率验证  ")
    logger.info("=" * 60)
    
    # 获取设备信息
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA设备: {device_name}")
    
    # 计算投影维度
    if args.projection_dim is None:
        projection_dim = max(64, int(args.seq_length * 0.25))
    else:
        projection_dim = args.projection_dim
    
    logger.info(f"测试配置:")
    logger.info(f"- 序列长度: {args.seq_length}")
    logger.info(f"- 批量大小: {args.batch_size}")
    logger.info(f"- Linformer投影维度: {projection_dim}")
    logger.info(f"- 使用AMP: {'是' if args.use_amp else '否'}")
    
    # 1. 创建并验证标准模型
    logger.info("\n[1/4] 创建标准Transformer模型...")
    standard_model = create_standard_model(args.seq_length, args.batch_size)
    
    # 2. 验证标准模型内存使用
    logger.info("\n[2/4] 验证标准Transformer模型内存使用...")
    standard_results = validate_model_memory(
        "标准Transformer",
        standard_model,
        args.seq_length, 
        args.batch_size,
        args.use_amp
    )
    
    # 清理内存
    standard_model = standard_model.cpu()
    del standard_model
    torch.cuda.empty_cache()
    
    # 3. 创建并验证Linformer模型
    logger.info("\n[3/4] 创建Linformer模型...")
    linformer_model = create_linformer_model(
        args.seq_length, 
        args.batch_size,
        projection_dim
    )
    
    # 添加投影维度到结果
    if standard_results:
        standard_results["projection_dim"] = None
    
    # 4. 验证Linformer模型内存使用
    logger.info("\n[4/4] 验证Linformer模型内存使用...")
    linformer_results = validate_model_memory(
        "Linformer",
        linformer_model,
        args.seq_length, 
        args.batch_size,
        args.use_amp
    )
    
    # 添加投影维度到结果
    if linformer_results:
        linformer_results["projection_dim"] = projection_dim
    
    # 绘制内存使用对比图
    if standard_results and linformer_results and standard_results["success"] and linformer_results["success"]:
        plot_memory_comparison(
            standard_results, 
            linformer_results, 
            args.seq_length,
            "linformer_memory_comparison.png"
        )
        
        # 计算总体节省百分比
        std_peak = standard_results["peak_memory"]
        lin_peak = linformer_results["peak_memory"]
        savings = (std_peak - lin_peak) / std_peak * 100
        
        logger.info("\n总体结果:")
        logger.info(f"- 标准Transformer峰值内存: {format_memory(std_peak)}")
        logger.info(f"- Linformer峰值内存: {format_memory(lin_peak)}")
        logger.info(f"- 内存节省: {savings:.2f}%")
        
        if savings >= 40:
            logger.info(f"✓ 成功达成目标: 内存减少 {savings:.2f}% ≥ 40%")
        else:
            logger.warning(f"✗ 未达成目标: 内存减少 {savings:.2f}% < 40%")
    else:
        logger.error("无法生成内存对比，部分模型验证失败")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"验证过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
