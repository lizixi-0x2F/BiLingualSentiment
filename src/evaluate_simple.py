#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 简化版评估脚本
适用于中英双语情感价效度回归任务
"""

import os
import argparse
import yaml
import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

from src.core import LTC_NCP_RNN
from src.train_simple import SimpleTokenizer, EmotionDataset, concordance_correlation_coefficient
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('LTC-NCP-VA-Eval')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估LTC-NCP-VA模型')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data', type=str, help='测试数据路径')
    parser.add_argument('--output', type=str, default='results/evaluation', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--device', type=str, default=None, help='设备(cuda或cpu)')
    
    return parser.parse_args()

def visualize_va_space(predictions, targets, output_path, n_samples=100, title="VA空间预测"):
    """可视化VA空间预测与目标"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果样本数量超过n_samples，随机选择n_samples个样本
    if len(predictions) > n_samples:
        indices = np.random.choice(len(predictions), n_samples, replace=False)
        pred_subset = predictions[indices]
        targets_subset = targets[indices]
    else:
        pred_subset = predictions
        targets_subset = targets
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    
    # 绘制网格和象限分界线
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制目标值
    plt.scatter(targets_subset[:, 0], targets_subset[:, 1], c='blue', marker='o', label='目标', alpha=0.5)
    
    # 绘制预测值
    plt.scatter(pred_subset[:, 0], pred_subset[:, 1], c='red', marker='x', label='预测', alpha=0.5)
    
    # 绘制连接线
    for i in range(len(pred_subset)):
        plt.plot([targets_subset[i, 0], pred_subset[i, 0]], 
                 [targets_subset[i, 1], pred_subset[i, 1]], 
                 'gray', alpha=0.2)
    
    # 标注象限
    plt.text(0.8, 0.8, '高V高A\n(喜悦/兴奋)', ha='center', va='center', fontsize=12)
    plt.text(0.8, -0.8, '高V低A\n(平静/放松)', ha='center', va='center', fontsize=12)
    plt.text(-0.8, 0.8, '低V高A\n(愤怒/恐惧)', ha='center', va='center', fontsize=12)
    plt.text(-0.8, -0.8, '低V低A\n(悲伤/疲惫)', ha='center', va='center', fontsize=12)
    
    # 添加标签和标题
    plt.xlabel('Valence (愉悦度)')
    plt.ylabel('Arousal (唤起度)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围，确保四象限被完整显示
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"VA空间可视化已保存至 {output_path}")

def evaluate(model, test_loader, device, output_dir):
    """评估模型性能"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # 禁用梯度计算
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            # 将数据移到设备
            tokens = batch['tokens'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['length']
            
            # 提取并处理元特征（如果使用）
            meta_features = None
            if model.use_meta_features and 'meta_features' in batch:
                meta_features = batch['meta_features'].to(device)
            
            # 计算模型输出
            outputs = model(tokens, lengths, meta_features)
            
            # 收集预测和目标
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
    
    # 将预测和目标拼接为两个大张量
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # 分别计算V和A的指标
    v_mse = mean_squared_error(all_targets[:, 0], all_predictions[:, 0])
    a_mse = mean_squared_error(all_targets[:, 1], all_predictions[:, 1])
    v_rmse = np.sqrt(v_mse)
    a_rmse = np.sqrt(a_mse)
    v_mae = mean_absolute_error(all_targets[:, 0], all_predictions[:, 0])
    a_mae = mean_absolute_error(all_targets[:, 1], all_predictions[:, 1])
    
    # 相关系数
    v_spearman = spearmanr(all_targets[:, 0], all_predictions[:, 0])[0]
    a_spearman = spearmanr(all_targets[:, 1], all_predictions[:, 1])[0]
    v_pearson = pearsonr(all_targets[:, 0], all_predictions[:, 0])[0]
    a_pearson = pearsonr(all_targets[:, 1], all_predictions[:, 1])[0]
    
    # CCC
    v_ccc = concordance_correlation_coefficient(all_targets[:, 0], all_predictions[:, 0])
    a_ccc = concordance_correlation_coefficient(all_targets[:, 1], all_predictions[:, 1])
    
    # 平均指标
    avg_mse = (v_mse + a_mse) / 2
    avg_rmse = (v_rmse + a_rmse) / 2
    avg_mae = (v_mae + a_mae) / 2
    avg_spearman = (v_spearman + a_spearman) / 2
    avg_pearson = (v_pearson + a_pearson) / 2
    avg_ccc = (v_ccc + a_ccc) / 2
    
    # 整理并打印结果
    results = {
        'v_mse': v_mse,
        'a_mse': a_mse,
        'v_rmse': v_rmse,
        'a_rmse': a_rmse,
        'v_mae': v_mae,
        'a_mae': a_mae,
        'v_spearman': v_spearman,
        'a_spearman': a_spearman,
        'v_pearson': v_pearson,
        'a_pearson': a_pearson,
        'v_ccc': v_ccc,
        'a_ccc': a_ccc,
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_spearman': avg_spearman,
        'avg_pearson': avg_pearson,
        'avg_ccc': avg_ccc
    }
    
    # 打印评估结果
    logger.info("评估结果:")
    logger.info(f"  Valence (愉悦度): RMSE={v_rmse:.4f}, MAE={v_mae:.4f}, CCC={v_ccc:.4f}")
    logger.info(f"  Arousal (唤起度): RMSE={a_rmse:.4f}, MAE={a_mae:.4f}, CCC={a_ccc:.4f}")
    logger.info(f"  平均指标: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, CCC={avg_ccc:.4f}")
    
    # 保存结果为CSV
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    pd.DataFrame([results]).to_csv(results_path, index=False)
    logger.info(f"评估指标已保存至 {results_path}")
    
    # 可视化VA空间
    vis_path = os.path.join(output_dir, 'va_space_visualization.png')
    visualize_va_space(all_predictions, all_targets, vis_path, n_samples=200, 
                       title="VA空间预测 vs 目标")
    
    return results, all_predictions, all_targets

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型检查点
    logger.info(f"加载模型检查点 {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    config = checkpoint.get('config', {})
    
    # 获取模型配置
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    # 创建分词器
    tokenizer = SimpleTokenizer(vocab_size=10000, min_freq=2)
    
    # 加载测试数据
    test_path = args.data if args.data else data_config.get('val_path')  # 默认使用验证集
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_path}")
    
    # 加载测试数据集
    logger.info(f"加载测试数据 {test_path}...")
    test_df = pd.read_csv(test_path)
    
    # 构建分词器词汇表
    tokenizer.fit(test_df[data_config.get('text_col', 'text')].fillna("").tolist())
    
    # 创建测试数据集
    test_dataset = EmotionDataset(
        test_path, 
        tokenizer, 
        data_config, 
        max_length=data_config.get('max_length', 100),
        is_training=False
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = LTC_NCP_RNN(
        vocab_size=len(tokenizer.token2idx),
        embedding_dim=model_config.get('embedding_dim', 256),
        hidden_size=model_config.get('hidden_size', 128),
        output_size=2,  # VA预测
        dropout=model_config.get('dropout', 0.3),
        sparsity_level=model_config.get('sparsity_level', 0.5),
        dt=model_config.get('dt', 0.1),
        integration_method=model_config.get('integration_method', 'euler'),
        use_meta_features=data_config.get('use_meta_features', False),
        bidirectional=model_config.get('bidirectional', True),
        padding_idx=0,  # <PAD>的索引
        wiring_type=model_config.get('wiring_type', 'structured'),
        multi_level=model_config.get('multi_level', False),
        emotion_focused=model_config.get('emotion_focused', False),
        heterogeneous=model_config.get('heterogeneous', False),
        use_transformer=model_config.get('use_transformer', False),
        invert_valence=model_config.get('invert_valence', False),
        invert_arousal=model_config.get('invert_arousal', False),
        enhance_valence=model_config.get('enhance_valence', False),
        valence_layers=model_config.get('valence_layers', 1),
        use_quadrant_head=model_config.get('use_quadrant_head', False),
        quadrant_weight=model_config.get('quadrant_weight', 0.0),
        meta_features_count=len(data_config.get('meta_features', [])) if data_config.get('use_meta_features', False) else 0
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 评估模型
    logger.info("开始评估...")
    results, predictions, targets = evaluate(model, test_loader, device, output_dir)
    
    logger.info(f"评估完成! 结果保存在 {output_dir}")
    return results

if __name__ == '__main__':
    main()
