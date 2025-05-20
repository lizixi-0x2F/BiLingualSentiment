#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XLM-R模型Linformer注意力转换脚本
用于将标准注意力机制替换为内存效率更高的Linformer注意力
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入必要的模块
from src.core import LTC_NCP_RNN
from src.core.linformer_mini_transformer import LinformerMiniTransformer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('linformer_conversion')

def memory_usage_report():
    """报告当前CUDA内存使用情况 - 优化版本"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法报告GPU内存使用情况")
        return {}
    
    # 强制释放未使用的缓存
    torch.cuda.empty_cache()
    
    # 获取内存信息
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    
    # 格式化为MB
    to_mb = lambda x: x / (1024 * 1024)
    
    report = {
        "allocated": to_mb(allocated),
        "max_allocated": to_mb(max_allocated),
        "reserved": to_mb(reserved),
        "max_reserved": to_mb(max_reserved)
    }
    
    logger.info(f"CUDA内存使用 (MB):")
    logger.info(f"  已分配: {report['allocated']:.2f} (峰值: {report['max_allocated']:.2f})")
    logger.info(f"  已保留: {report['reserved']:.2f} (峰值: {report['max_reserved']:.2f})")
    
    # 计算活跃的内存块数量
    active_blocks = torch.cuda.memory_snapshot()
    active_block_count = len(active_blocks)
    active_block_size = sum(b.get('size', 0) for b in active_blocks) / (1024 * 1024)
    
    logger.info(f"  活跃内存块: {active_block_count} (总计: {active_block_size:.2f} MB)")
    
    return report

def convert_to_linformer(model, projection_dim=None):
    """
    将标准transformer注意力转换为linformer注意力
    
    参数:
        model: 要转换的模型
        projection_dim: Linformer投影维度，如果为None则自动设置为序列长度的25%，最小64
    """
    logger.info("开始将标准Transformer转换为Linformer...")
    
    if not hasattr(model, 'transformer') or not hasattr(model, 'use_transformer') or not model.use_transformer:
        logger.error("模型中未找到Transformer组件或未启用")
        return model, False
    
    # 备份原始配置
    d_model = model.transformer.d_model
    nhead = len(model.transformer.layers) > 0 and hasattr(model.transformer.layers[0], 'self_attn') and model.transformer.layers[0].self_attn.num_heads
    if nhead is None or nhead == 0:
        nhead = 4  # 默认值
    num_layers = len(model.transformer.layers)
    dim_feedforward = model.transformer.layers[0].linear1.out_features if len(model.transformer.layers) > 0 else d_model * 2
    dropout = model.transformer.dropout.p
    
    logger.info(f"Transformer参数信息:")
    logger.info(f"  模型维度: {d_model}")
    logger.info(f"  注意力头数: {nhead}")
    logger.info(f"  层数: {num_layers}")
    logger.info(f"  前馈网络维度: {dim_feedforward}")
    logger.info(f"  Dropout比例: {dropout}")
    
    # 如果未指定projection_dim，设置为序列长度的25%，最小64
    if projection_dim is None:
        # 假定默认序列长度为512
        seq_len = 512
        projection_dim = max(64, int(seq_len * 0.25))
    
    logger.info(f"Linformer投影维度: {projection_dim}")
    
    # 创建新的LinformerMiniTransformer
    linformer_transformer = LinformerMiniTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        projection_dim=projection_dim
    )
    
    # 复制位置编码和规范化层权重
    linformer_transformer.pos_encoder.data.copy_(model.transformer.pos_encoder.data)
    linformer_transformer.input_norm.weight.data.copy_(model.transformer.input_norm.weight.data)
    linformer_transformer.input_norm.bias.data.copy_(model.transformer.input_norm.bias.data)
    linformer_transformer.input_adapter.weight.data.copy_(model.transformer.input_adapter.weight.data)
    linformer_transformer.input_adapter.bias.data.copy_(model.transformer.input_adapter.bias.data)
    
    # 替换模型中的transformer
    model.transformer = linformer_transformer
    
    logger.info("Transformer替换完成！")
    return model, True

def validate_model(model, seq_length=512, batch_size=4, use_amp=False):
    """验证模型可以通过前向和后向传播"""
    logger.info(f"验证模型 (seq_len={seq_length}, batch_size={batch_size})...")
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"使用设备: {device}")
    
    # 创建随机输入
    tokens = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    lengths = torch.randint(5, seq_length + 1, (batch_size,), device=device)
    
    # AMP设置
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    amp_context = autocast(device_type=device.type, dtype=torch.float16) if use_amp and device.type == 'cuda' else nullcontext()
    
    model.train()  # 确保处于训练模式
    
    try:
        # 前向传播
        with amp_context:
            outputs = model(tokens, lengths)
            loss = outputs.mean()  # 简单损失函数用于测试
        
        # 后向传播
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(torch.optim.Adam(model.parameters(), lr=1e-4))
            scaler.update()
        else:
            loss.backward()
            torch.optim.Adam(model.parameters(), lr=1e-4).step()
        
        logger.info(f"✓ 模型验证通过！前向+后向传播成功")
        logger.info(f"  输出形状: {outputs.shape}")
        return True
    
    except Exception as e:
        logger.error(f"✗ 模型验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='XLM-R模型LinformerSelfAttention转换')
    parser.add_argument('--projection-dim', type=int, default=None, 
                        help='Linformer投影维度 (默认为序列长度的25%，最小64)')
    parser.add_argument('--seq-length', type=int, default=512, 
                        help='用于验证的序列长度 (默认: 512)')
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='用于验证的批量大小 (默认: 4)')
    parser.add_argument('--use-amp', action='store_true', 
                        help='使用自动混合精度')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  XLM-R模型 Linformer Self-Attention 转换  ")
    print("=" * 60)
    
    # 创建标准模型
    logger.info("创建原始模型...")
    standard_model = LTC_NCP_RNN(
        vocab_size=1000,
        embedding_dim=64,
        hidden_size=128,
        output_size=2,
        dropout=0.3,
        use_transformer=True
    )
    
    # 测量原始模型内存
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_model = standard_model.to(device)
    
    # 1. 测量原始模型的内存使用
    logger.info("\n[测量1] 原始模型内存使用:")
    torch.cuda.reset_peak_memory_stats()
    standard_memory_before = memory_usage_report()
    
    # 测试原始模型
    validate_model(
        standard_model, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size,
        use_amp=args.use_amp
    )
    
    standard_memory_after = memory_usage_report()
    
    # 2. 转换为Linformer
    torch.cuda.empty_cache()
    logger.info("\n转换为Linformer模型...")
    linformer_model, success = convert_to_linformer(standard_model, args.projection_dim)
    
    if not success:
        logger.error("模型转换失败，退出")
        return
    
    # 3. 测量Linformer模型的内存使用
    logger.info("\n[测量2] Linformer模型内存使用:")
    torch.cuda.reset_peak_memory_stats()
    linformer_memory_before = memory_usage_report()
    
    # 测试Linformer模型
    success = validate_model(
        linformer_model, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size,
        use_amp=args.use_amp
    )
    
    if not success:
        logger.error("Linformer模型验证失败")
        return
    
    linformer_memory_after = memory_usage_report()
    
    # 计算内存节省
    standard_peak = standard_memory_after["max_allocated"]
    linformer_peak = linformer_memory_after["max_allocated"]
    savings = (standard_peak - linformer_peak) / standard_peak * 100
    
    logger.info("\n内存使用对比:")
    logger.info(f"  标准模型峰值内存: {standard_peak:.2f} MB")
    logger.info(f"  Linformer模型峰值内存: {linformer_peak:.2f} MB")
    logger.info(f"  内存节省: {savings:.2f}%")
    
    if savings >= 40:
        logger.info(f"✓ 成功达成目标: 内存减少 {savings:.2f}% ≥ 40%")
    else:
        logger.warning(f"✗ 未达成目标: 内存减少 {savings:.2f}% < 40%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
