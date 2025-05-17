#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 评估和可视化脚本
用于测试集上的性能评估和结果可视化
"""

import os
import yaml
import json
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.core import LTC_NCP_RNN
from src.train import EmotionDataset, SimpleTokenizer, compute_metrics, prepare_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LTC-NCP-VA-Eval')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LTC-NCP-VA 评估脚本')
    parser.add_argument('--ckpt', type=str, required=True, help='检查点路径')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--n_samples', type=int, default=100, help='可视化样本数量')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    return parser.parse_args()


def load_checkpoint(ckpt_path, device):
    """加载检查点"""
    # 检查路径是否存在
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件不存在: {ckpt_path}")
    
    logger.info(f"加载检查点: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 提取配置和状态字典
    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']
    
    # 创建模型
    model_config = config['model']
    
    # 需要知道词汇表大小，先加载训练数据
    train_en_df = pd.read_csv(config['data']['train']['en'])
    train_zh_df = pd.read_csv(config['data']['train']['zh'])
    
    # 创建分词器并构建词汇表
    text_col = config['data']['text_col']
    all_texts = list(train_en_df[text_col]) + list(train_zh_df[text_col])
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.build_vocab(all_texts, min_freq=2)
    
    # 初始化模型
    model = LTC_NCP_RNN(
        vocab_size=tokenizer.vocab_size_actual,
        embedding_dim=model_config['embedding_dim'],
        hidden_size=model_config['hidden_size'],
        output_size=2,  # V和A
        dropout=model_config['dropout'],
        sparsity_level=model_config['sparsity_level'],
        dt=model_config['dt'],
        integration_method=model_config['integration_method'],
        use_meta_features=config['data'].get('use_meta_features', False),
        bidirectional=model_config.get('bidirectional', False),
        padding_idx=0  # PAD token ID
    )
    
    # 加载状态字典
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer, config


def evaluate_model(model, test_loader, device, config):
    """在测试集上评估模型"""
    model.eval()
    all_preds = []
    all_targets = []
    all_texts = []
    
    # 进度条
    pbar = tqdm(test_loader, desc="评估")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            tokens = batch['tokens'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['length']
            
            # 提取并处理元特征
            meta_features = None
            if config['data'].get('use_meta_features', False) and 'meta_features' in batch:
                meta_features = batch['meta_features'].to(device)
            
            # 前向传播
            outputs = model(tokens, lengths, meta_features)
            
            # 收集预测、目标和文本
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    # 拼接数组
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算指标
    metrics = compute_metrics(all_targets, all_preds)
    
    return all_preds, all_targets, metrics


def visualize_va_space(predictions, targets, output_path, n_samples=100, title="VA空间预测"):
    """可视化VA空间预测与目标"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果样本数量大于n_samples，随机选择n_samples个样本
    if len(predictions) > n_samples:
        indices = np.random.choice(len(predictions), n_samples, replace=False)
        pred_subset = predictions[indices]
        target_subset = targets[indices]
    else:
        pred_subset = predictions
        target_subset = targets
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    
    # 绘制四象限边界
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # 绘制目标和预测
    plt.scatter(target_subset[:, 0], target_subset[:, 1], c='blue', alpha=0.5, label='目标')
    plt.scatter(pred_subset[:, 0], pred_subset[:, 1], c='red', alpha=0.5, label='预测')
    
    # 添加线条连接目标和对应的预测
    for i in range(len(pred_subset)):
        plt.plot([target_subset[i, 0], pred_subset[i, 0]], 
                 [target_subset[i, 1], pred_subset[i, 1]], 
                 'gray', alpha=0.3)
    
    # 绘制评价空间的四个象限
    plt.text(0.8, 0.8, '高V高A\n(激动/兴奋)', ha='center', va='center', fontsize=12)
    plt.text(-0.8, 0.8, '低V高A\n(愤怒/恐惧)', ha='center', va='center', fontsize=12)
    plt.text(-0.8, -0.8, '低V低A\n(悲伤/疲惫)', ha='center', va='center', fontsize=12)
    plt.text(0.8, -0.8, '高V低A\n(平静/放松)', ha='center', va='center', fontsize=12)
    
    # 添加标签和标题
    plt.xlabel('Valence (愉悦度)')
    plt.ylabel('Arousal (唤起度)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置轴范围
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"保存VA空间可视化到：{output_path}")
    plt.close()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    plots_dir = os.path.join(args.output, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载检查点和模型
    model, tokenizer, config = load_checkpoint(args.ckpt, device)
    
    # 准备数据
    _, _, _, test_loader = prepare_data(config)
    
    # 评估模型
    predictions, targets, metrics = evaluate_model(model, test_loader, device, config)
    
    # 输出评估指标
    logger.info("测试集评估结果:")
    logger.info(f"整体 RMSE: {metrics['overall']['rmse']:.4f}")
    logger.info(f"V-CCC: {metrics['V']['ccc']:.4f}, A-CCC: {metrics['A']['ccc']:.4f}")
    logger.info(f"平均CCC: {metrics['avg_ccc']:.4f}")
    
    # 保存指标到JSON文件
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"保存指标到: {metrics_path}")
    
    # 可视化VA空间
    va_plot_path = os.path.join(plots_dir, 'va_space.png')
    visualize_va_space(
        predictions, targets, va_plot_path, 
        n_samples=args.n_samples, 
        title=f"VA空间预测 (平均CCC: {metrics['avg_ccc']:.4f})"
    )
    
    # 分别可视化V和A的预测
    for i, dim in enumerate(['V', 'A']):
        # 根据指标格式化标题
        title = f"{dim} 预测 (CCC: {metrics[dim]['ccc']:.4f}, RMSE: {metrics[dim]['rmse']:.4f})"
        
        # 对样本进行排序以便绘制
        if len(predictions) > args.n_samples:
            indices = np.random.choice(len(predictions), args.n_samples, replace=False)
            sorted_indices = indices[np.argsort(targets[indices, i])]
        else:
            sorted_indices = np.argsort(targets[:, i])
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制目标和预测
        plt.plot(targets[sorted_indices, i], 'b-', label='目标', linewidth=2)
        plt.plot(predictions[sorted_indices, i], 'r--', label='预测', linewidth=2)
        
        # 添加阴影区域表示误差
        plt.fill_between(
            range(len(sorted_indices)),
            targets[sorted_indices, i],
            predictions[sorted_indices, i],
            color='gray', alpha=0.2
        )
        
        # 添加标签和标题
        plt.xlabel('样本索引（按目标值排序）')
        plt.ylabel(f'{dim} 值')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        plot_path = os.path.join(plots_dir, f'{dim}_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"保存{dim}预测可视化到：{plot_path}")
        plt.close()


if __name__ == "__main__":
    main() 