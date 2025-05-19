#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 训练脚本 - 简化版
适用于中英双语情感价效度回归任务
"""

import os
import yaml
import argparse
import logging
import time
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from src.core import LTC_NCP_RNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

# 获取日志级别环境变量
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 设置日志
logging.basicConfig(
    level=LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('LTC-NCP-VA')

# 禁用模型内部调试输出
try:
    import src.core.model
    src.core.model.DEBUG = False
except ImportError:
    pass

# 检查添加的额外元特征是否可用
def add_sentence_count(df, text_col="text"):
    """添加句子数量特征"""
    import re
    # 简单句子分割正则表达式，支持中英文句子
    sentence_splitter = re.compile(r'[.!?。！？]+')
    
    # 计算句子数量
    def count_sentences(text):
        if not isinstance(text, str):
            return 1
        # 分割句子并计数非空句子
        sentences = [s.strip() for s in sentence_splitter.split(text)]
        return sum(1 for s in sentences if s)
    
    df['sentence_count'] = df[text_col].apply(count_sentences)
    return df

# 计算CCC (Concordance Correlation Coefficient)
def concordance_correlation_coefficient(y_true, y_pred):
    """
    计算一致性相关系数 (CCC)
    """
    # 将输入转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 确保是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算均值
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # 计算方差
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # 计算协方差
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # 计算CCC
    numerator = 2 * covar
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    # 避免除零错误
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator

class EmotionDataset(Dataset):
    """情感数据集"""
    
    def __init__(self, csv_path, tokenizer, config, max_length=100, is_training=True):
        """
        初始化数据集
        
        参数:
            csv_path: CSV文件路径
            tokenizer: 分词器
            config: 数据配置
            max_length: 最大序列长度
            is_training: 是否为训练集
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.is_training = is_training
        
        # 定义列名
        self.text_col = config['text_col']
        self.v_col = config['v_col']
        self.a_col = config['a_col']
        
        # 检查数据集是否包含必要的列
        for col in [self.text_col, self.v_col, self.a_col]:
            if col not in self.df.columns:
                raise ValueError(f"数据集缺少列: {col}")
        
        # 添加额外元特征（如果在配置中但不在数据中）
        if 'sentence_count' in config.get('meta_features', []) and 'sentence_count' not in self.df.columns:
            self.df = add_sentence_count(self.df, self.text_col)
                
        # 元特征（如果启用）
        self.use_meta_features = config.get('use_meta_features', False)
        self.meta_features = config.get('meta_features', [])
        
        # 检查元特征列是否存在
        if self.use_meta_features:
            for col in self.meta_features:
                if col not in self.df.columns:
                    raise ValueError(f"数据集缺少元特征列: {col}")
    
    def _clean_nan_values(self):
        """清理NaN值"""
        # 检查并替换文本列中的NaN
        if self.text_col in self.df.columns:
            self.df[self.text_col] = self.df[self.text_col].fillna("")
        
        # 检查并替换VA列中的NaN
        for col in [self.v_col, self.a_col]:
            if col in self.df.columns and self.df[col].isna().any():
                logger.warning(f"列 {col} 包含NaN值，使用0替代")
                self.df[col] = self.df[col].fillna(0.0)
        
        # 检查并替换元特征列中的NaN
        if self.use_meta_features:
            for col in self.meta_features:
                if col in self.df.columns and self.df[col].isna().any():
                    logger.warning(f"元特征列 {col} 包含NaN值，使用均值替代")
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取数据集的一个样本"""
        row = self.df.iloc[idx]
        
        # 获取文本并进行分词
        text = str(row[self.text_col])
        tokens = self.tokenizer.tokenize(text, self.max_length)
        
        # 获取VA值
        v_value = float(row[self.v_col])
        a_value = float(row[self.a_col])
        targets = torch.tensor([v_value, a_value], dtype=torch.float32)
        
        # 构造基本样本
        sample = {
            'tokens': tokens,
            'targets': targets,
            'length': len(tokens)
        }
        
        # 添加元特征（如果启用）
        if self.use_meta_features:
            meta_features = [float(row[col]) for col in self.meta_features]
            sample['meta_features'] = torch.tensor(meta_features, dtype=torch.float32)
        
        return sample

class SimpleTokenizer:
    """简单分词器，适用于英文和中文"""
    
    def __init__(self, vocab_size=10000, min_freq=2):
        """
        初始化分词器
        
        参数:
            vocab_size: 词汇表大小
            min_freq: 最小词频
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.token_freqs = {}
        self.is_fitted = False
    
    def fit(self, texts):
        """
        构建词汇表
        
        参数:
            texts: 文本列表
        """
        # 统计词频
        for text in texts:
            for char in text:
                if char not in self.token_freqs:
                    self.token_freqs[char] = 0
                self.token_freqs[char] += 1
        
        # 按词频排序，只保留高频词
        sorted_tokens = sorted(self.token_freqs.items(), key=lambda x: x[1], reverse=True)
        for i, (token, freq) in enumerate(sorted_tokens):
            if i >= self.vocab_size - 4 or freq < self.min_freq:  # 4是特殊token的数量
                break
            self.token2idx[token] = i + 4  # 从4开始，前4个是特殊token
            self.idx2token[i + 4] = token
        
        self.is_fitted = True
        logger.info(f"词汇表构建完成，大小: {len(self.token2idx)}")
    
    def tokenize(self, text, max_length):
        """
        将文本转换为token索引
        
        参数:
            text: 输入文本
            max_length: 最大长度
        
        返回:
            token索引列表
        """
        if not self.is_fitted:
            raise ValueError("请先调用fit()构建词汇表")
        
        # 字符级分词
        tokens = [self.token2idx.get(char, self.token2idx['<UNK>']) for char in text[:max_length]]
        
        # 填充或截断到指定长度
        if len(tokens) < max_length:
            tokens = tokens + [self.token2idx['<PAD>']] * (max_length - len(tokens))
        elif len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def vocab_size(self):
        """返回词汇表大小"""
        return len(self.token2idx)

def parse_args():
    parser = argparse.ArgumentParser(description='训练LTC-NCP-VA模型')
    parser.add_argument('--config', type=str, default='configs/simple_base.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数(覆盖配置文件)')
    parser.add_argument('--batch_size', type=int, help='批量大小(覆盖配置文件)')
    parser.add_argument('--lr', type=float, help='学习率(覆盖配置文件)')
    parser.add_argument('--save_dir', type=str, help='模型保存目录(覆盖配置文件)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='设备(覆盖配置文件)')
    parser.add_argument('--seed', type=int, help='随机种子(覆盖配置文件)')
    parser.add_argument('--debug', action='store_true', help='调试模式(减少数据)')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(config):
    """
    准备数据集
    
    参数:
        config: 数据配置
    
    返回:
        train_dataset: 训练集
        val_dataset: 验证集
        tokenizer: 分词器
    """
    # 获取数据路径
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']
    
    # 检查文件是否存在
    for path in [train_path, val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
    
    # 加载数据集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # 构建分词器
    tokenizer = SimpleTokenizer(vocab_size=10000, min_freq=2)
    tokenizer.fit(train_df[config['data']['text_col']].fillna("").tolist())
    
    # 创建数据集
    train_dataset = EmotionDataset(
        train_path, 
        tokenizer, 
        config['data'], 
        max_length=config['data'].get('max_length', 100),
        is_training=True
    )
    
    val_dataset = EmotionDataset(
        val_path, 
        tokenizer, 
        config['data'], 
        max_length=config['data'].get('max_length', 100),
        is_training=False
    )
    
    return train_dataset, val_dataset, tokenizer

def create_model(config, vocab_size):
    """
    创建模型
    
    参数:
        config: 模型配置
        vocab_size: 词汇表大小
    
    返回:
        model: LTC-NCP-RNN模型
    """
    # 获取模型配置
    model_config = config['model']
    
    # 创建模型
    model = LTC_NCP_RNN(
        vocab_size=vocab_size,
        embedding_dim=model_config.get('embedding_dim', 128),
        hidden_size=model_config.get('hidden_size', 128),
        output_size=2,  # VA预测
        dropout=model_config.get('dropout', 0.3),
        sparsity_level=model_config.get('sparsity_level', 0.5),
        dt=model_config.get('dt', 0.1),
        integration_method=model_config.get('integration_method', 'euler'),
        use_meta_features=config['data'].get('use_meta_features', False),
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
        use_quadrant_head=model_config.get('use_quadrant_head', False),  # 不使用四象限分类头
        quadrant_weight=model_config.get('quadrant_weight', 0.0),
        meta_features_count=len(config['data'].get('meta_features', [])) if config['data'].get('use_meta_features', False) else 0
    )
    
    return model

def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler=None, scaler=None):
    """训练一个轮次"""
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    all_preds = []
    all_targets = []
    
    # 获取训练配置
    grad_accum_steps = config.get('optimization', {}).get('gradient_accumulation_steps', 1)
    use_amp = config.get('training', {}).get('use_amp', False)
    
    # 创建训练进度条
    pbar = tqdm(train_loader, desc="训练")
    
    # 确定是否使用自动混合精度
    from contextlib import nullcontext
    
    # 冻结A头相关参数
    freeze_a_head = False
    freeze_a_head_epochs = config.get('optimization', {}).get('freeze_a_head_epochs', 0)
    if freeze_a_head_epochs > 0 and model.current_epoch < freeze_a_head_epochs:
        freeze_a_head = True
    
    # 如果需要冻结A头，先冻结相关参数
    if freeze_a_head:
        for name, param in model.named_parameters():
            if 'arousal_branch' in name:
                param.requires_grad = False
        logger.info("已冻结Arousal头部参数")
    
    optimizer.zero_grad()  # 在循环外清零梯度，以支持梯度累积
    
    for i, batch in enumerate(pbar):
        # 将数据移到设备
        tokens = batch['tokens'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['length']
        
        # 确保tokens是长整型
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        
        # 提取并处理元特征（如果使用）
        meta_features = None
        if config['data'].get('use_meta_features', False) and 'meta_features' in batch:
            meta_features = batch['meta_features'].to(device)
        
        # 计算模型输出和损失
        with autocast() if use_amp else nullcontext():
            # 前向传播
            outputs = model(tokens, lengths, meta_features)
            
            # 检查输出中是否包含NaN 
            if torch.isnan(outputs).any():
                logger.debug(f"警告: 批次{i}包含NaN输出，跳过此批次...")
                # 更新进度条显示
                pbar.set_postfix({'loss': 'NaN-跳过'})
                continue
            
            # 计算损失
            loss = criterion(outputs, targets)
        
        # 反向传播和优化
        if use_amp:
            # 使用梯度缩放器进行反向传播
            scaler.scale(loss / grad_accum_steps).backward()
            
            # 每grad_accum_steps步更新一次参数
            if (i + 1) % grad_accum_steps == 0:
                # 裁剪梯度
                if config['training'].get('clip_grad_norm', 0) > 0:
                    # 需要先取消缩放再裁剪
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
                
                # 使用scaler更新参数
                scaler.step(optimizer)
                scaler.update()
                
                # 更新学习率（如果使用批次级调度器）
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                optimizer.zero_grad()
        else:
            # 普通反向传播
            (loss / grad_accum_steps).backward()
            
            # 每grad_accum_steps步更新一次参数
            if (i + 1) % grad_accum_steps == 0:
                # 裁剪梯度
                if config['training'].get('clip_grad_norm', 0) > 0:
                    clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
                
                optimizer.step()
                
                # 更新学习率（如果使用批次级调度器）
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                optimizer.zero_grad()
        
        # 更新统计信息
        total_loss += loss.item() * grad_accum_steps
        valid_batches += 1
        
        # 收集预测和目标用于指标计算
        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # 更新进度条显示
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算平均损失
    avg_loss = total_loss / max(valid_batches, 1)
    
    # 将预测和目标拼接为两个大张量
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
        
        # 计算训练集上的指标
        v_mse = mean_squared_error(all_targets[:, 0], all_preds[:, 0])
        a_mse = mean_squared_error(all_targets[:, 1], all_preds[:, 1])
        v_ccc = concordance_correlation_coefficient(all_targets[:, 0], all_preds[:, 0])
        a_ccc = concordance_correlation_coefficient(all_targets[:, 1], all_preds[:, 1])
        
        # 返回训练结果
        train_results = {
            'loss': avg_loss,
            'v_mse': v_mse,
            'a_mse': a_mse,
            'v_ccc': v_ccc,
            'a_ccc': a_ccc,
            'avg_mse': (v_mse + a_mse) / 2,
            'avg_ccc': (v_ccc + a_ccc) / 2
        }
    else:
        train_results = {'loss': avg_loss}
    
    return train_results

def validate(model, val_loader, criterion, device, config):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    all_predictions = []
    all_targets = []
    
    # 禁用梯度计算
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证"):
            # 将数据移到设备
            tokens = batch['tokens'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['length']
            
            # 确保tokens是长整型
            if tokens.dtype != torch.long:
                tokens = tokens.long()
            
            # 提取并处理元特征（如果使用）
            meta_features = None
            if config['data'].get('use_meta_features', False) and 'meta_features' in batch:
                meta_features = batch['meta_features'].to(device)
            
            # 计算模型输出
            outputs = model(tokens, lengths, meta_features)
            
            # 检查输出中是否包含NaN
            if torch.isnan(outputs).any():
                logger.debug("警告: 验证批次包含NaN输出，跳过...")
                continue
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 更新统计信息
            total_loss += loss.item()
            valid_batches += 1
            
            # 收集预测和目标
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
    
    # 计算平均损失
    avg_loss = total_loss / max(valid_batches, 1)
    
    # 计算指标
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
    
    # 返回结果
    results = {
        'loss': avg_loss,
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
        'avg_ccc': avg_ccc,
        'all_predictions': all_predictions,
        'all_targets': all_targets
    }
    
    return results

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    torch.save(checkpoint, save_path)
    logger.info(f"模型检查点已保存到 {save_path}")

def train_model(config, train_dataset, val_dataset, model, device, output_dir):
    """训练模型的主函数"""
    # 获取训练配置
    train_config = config['training']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 0.0)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['hardware'].get('num_workers', 0),
        drop_last=False,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时可以用更大的批量
        shuffle=False,
        num_workers=config['hardware'].get('num_workers', 0),
        drop_last=False,
        pin_memory=True
    )
    
    # 创建优化器
    optimizer_type = train_config.get('optimizer', 'adam').lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    # 创建学习率调度器
    scheduler = None
    scheduler_config = train_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', '').lower()
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=lr * 0.01
        )
    elif scheduler_type == 'cosine_with_warmup':
        # 线性预热+余弦退火
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'cosine_with_restarts':
        # 带重启的余弦退火
        restarts = scheduler_config.get('restarts', 2)
        T_0 = epochs // (restarts + 1)  # 初始周期长度
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, T_0),
            T_mult=1,
            eta_min=lr * 0.01
        )
    
    # 创建损失函数
    criterion = nn.MSELoss()
    
    # 评估配置
    eval_config = config.get('evaluation', {})
    best_metric_name = eval_config.get('best_metric', 'avg_rmse')
    higher_better = eval_config.get('higher_better', False)
    patience = eval_config.get('patience', 10)
    
    # 初始化最佳指标和耐心计数器
    best_metric = float('-inf') if higher_better else float('inf')
    patience_counter = 0
    
    # 创建TensorBoard日志记录器
    log_dir = os.path.join(output_dir, config.get('logging', {}).get('log_dir', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 创建自动混合精度缩放器
    scaler = None
    if config['training'].get('use_amp', False):
        scaler = GradScaler()
    
    # 主训练循环
    for epoch in range(epochs):
        logger.info(f"开始第 {epoch+1}/{epochs} 轮训练")
        
        # 存储当前轮次
        model.current_epoch = epoch
        
        # 训练一个轮次
        start_time = time.time()
        train_results = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            config, 
            scheduler=scheduler, 
            scaler=scaler
        )
        train_time = time.time() - start_time
        
        # 验证模型
        start_time = time.time()
        val_results = validate(model, val_loader, criterion, device, config)
        val_time = time.time() - start_time
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练和验证结果
        logger.info(f"轮次 {epoch+1}/{epochs} 结果:")
        logger.info(f"  训练: 损失={train_results['loss']:.4f}, V-CCC={train_results.get('v_ccc', 0):.4f}, A-CCC={train_results.get('a_ccc', 0):.4f}, 用时={train_time:.2f}s")
        logger.info(f"  验证: 损失={val_results['loss']:.4f}, V-CCC={val_results['v_ccc']:.4f}, A-CCC={val_results['a_ccc']:.4f}, V-RMSE={val_results['v_rmse']:.4f}, A-RMSE={val_results['a_rmse']:.4f}, 用时={val_time:.2f}s")
        logger.info(f"  学习率: {current_lr:.6f}")
        
        # 记录指标到TensorBoard
        for key, value in train_results.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_results.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'val/{key}', value, epoch)
        
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # 更新学习率（如果使用epoch级调度器）
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # 检查是否为最佳模型
        current_metric = val_results[best_metric_name]
        is_best = (higher_better and current_metric > best_metric) or (not higher_better and current_metric < best_metric)
        
        if is_best:
            logger.info(f"新的最佳模型! {best_metric_name} 从 {best_metric:.6f} 改善到 {current_metric:.6f}")
            best_metric = current_metric
            patience_counter = 0
            
            # 保存最佳模型
            save_path = os.path.join(output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, save_path)
            
            # 保存最佳指标
            with open(os.path.join(output_dir, 'best_metrics.json'), 'w') as f:
                # 过滤掉非标量值
                metrics_to_save = {k: v for k, v in val_results.items() if isinstance(v, (int, float))}
                json.dump(metrics_to_save, f, indent=4)
        else:
            patience_counter += 1
            logger.info(f"{best_metric_name} 没有改善。当前: {current_metric:.6f}, 最佳: {best_metric:.6f}, 耐心: {patience_counter}/{patience}")
        
        # 检查是否应该提前停止
        if patience_counter >= patience:
            logger.info(f"早停! {patience} 轮没有改善。")
            break
        
        # 定期保存检查点
        if config.get('logging', {}).get('save_checkpoints', False):
            checkpoint_interval = config.get('logging', {}).get('checkpoints_interval', 5)
            if (epoch + 1) % checkpoint_interval == 0:
                save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, save_path)
    
    # 保存最终模型
    save_path = os.path.join(output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, scheduler, epochs-1, best_metric, config, save_path)
    
    # 关闭TensorBoard writer
    writer.close()
    
    logger.info(f"训练完成! 最佳{best_metric_name}: {best_metric:.6f}")
    return best_metric

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 用命令行参数覆盖配置（如果提供）
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.save_dir:
        config['logging']['save_dir'] = args.save_dir
    if args.device:
        config['hardware']['device'] = args.device
    if args.seed:
        config['hardware']['seed'] = args.seed
    
    # 设置随机种子
    set_seed(config['hardware'].get('seed', 42))
    
    # 设置设备
    device_name = config['hardware'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    
    # 创建输出目录
    output_dir = config.get('output_dir', 'results/default')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存使用的配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 准备数据
    logger.info("准备数据...")
    train_dataset, val_dataset, tokenizer = prepare_data(config)
    logger.info(f"数据准备完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config, len(tokenizer.token2idx))
    model.to(device)
    logger.info("模型创建完成")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 训练模型
    logger.info("开始训练...")
    best_metric = train_model(config, train_dataset, val_dataset, model, device, output_dir)
    logger.info(f"训练结束! 最佳指标: {best_metric:.6f}")

if __name__ == '__main__':
    main()
