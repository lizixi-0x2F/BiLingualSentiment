#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 训练脚本
适用于中英双语情感价效度回归任务
增强版: 支持高级优化技术
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from ltc_ncp import LTC_NCP_RNN
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
    import ltc_ncp.model
    ltc_ncp.model.DEBUG = False
except ImportError:
    pass
except AttributeError:
    pass

# 检查添加的额外元特征是否可用
def add_sentence_count(df, text_col="text"):
    """添加句子数量特征"""
    # 简单计算句号、问号、感叹号的数量作为句子数的近似
    df["sentence_count"] = df[text_col].apply(
        lambda x: len([c for c in str(x) if c in ['.', '!', '?', '。', '！', '？']])
    )
    return df

class LabelSmoothingLoss(nn.Module):
    """带标签平滑的MSE损失"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # 在[-1,1]范围内应用标签平滑
        smoothed_targets = targets * (1 - self.smoothing) + 0 * self.smoothing
        return self.mse(predictions, smoothed_targets)

class MSE_CCC_Loss(nn.Module):
    """结合MSE和CCC的损失函数"""
    def __init__(self, mse_weight=1.0, ccc_weight=0.0):  # 改为只使用MSE
        super(MSE_CCC_Loss, self).__init__()
        self.mse_weight = mse_weight
        self.ccc_weight = ccc_weight
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        
        # 如果ccc_weight为0，则跳过CCC计算以提高效率
        if self.ccc_weight <= 0.0:
            return mse_loss
            
        # 分别计算V和A的CCC
        v_ccc = concordance_correlation_coefficient(predictions[:, 0], targets[:, 0])
        a_ccc = concordance_correlation_coefficient(predictions[:, 1], targets[:, 1])
        
        # 平均CCC
        avg_ccc = (v_ccc + a_ccc) / 2
        
        # 计算综合损失：Loss = mse_weight * MSE + ccc_weight * (1-CCC)
        loss = self.mse_weight * mse_loss + self.ccc_weight * (1 - avg_ccc)
        
        return loss

def mixup_data(x, y, alpha=1.0):
    """执行Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 注意：x可能是整数类型的tokens，不应该进行mixup操作
    if x.dtype == torch.long or x.dtype == torch.int:
        # 对于整数类型的tokens，不进行混合，直接返回原始x
        return x, lam * y + (1 - lam) * y[index, :]
    else:
        # 对于浮点类型的数据（如嵌入或特征），正常进行mixup
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y

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
        
        # 清理NaN值 - 新增
        self._clean_nan_values()
                
        # 元特征（如果启用）
        self.use_meta_features = config.get('use_meta_features', False)
        self.meta_features = config.get('meta_features', [])
        
        # 检查元特征列是否存在
        if self.use_meta_features:
            for col in self.meta_features:
                if col not in self.df.columns:
                    raise ValueError(f"数据集缺少元特征列: {col}")
        
        # 对Valence极端值样本进行上采样（仅在训练集中进行）
        if is_training and config.get('oversample_extreme_valence', False):
            self._oversample_extreme_valence()
    
    # 添加清理NaN值的方法
    def _clean_nan_values(self):
        """清理数据集中的NaN值"""
        # 1. 对文本列进行清理
        self.df[self.text_col] = self.df[self.text_col].fillna("")
        
        # 2. 对情感标签列进行清理
        # 检查并报告NaN值
        v_nan_count = self.df[self.v_col].isna().sum()
        a_nan_count = self.df[self.a_col].isna().sum()
        
        if v_nan_count > 0 or a_nan_count > 0:
            print(f"警告: 发现NaN值 - V列: {v_nan_count}, A列: {a_nan_count}")
            
        # 用0填充NaN值(情感中性值)
        self.df[self.v_col] = self.df[self.v_col].fillna(0.0)
        self.df[self.a_col] = self.df[self.a_col].fillna(0.0)
        
        # 3. 清理元特征
        if 'meta_features' in self.config:
            for col in self.config['meta_features']:
                if col in self.df.columns:
                    # 报告NaN值
                    nan_count = self.df[col].isna().sum()
                    if nan_count > 0:
                        print(f"警告: 元特征 {col} 中有 {nan_count} 个NaN值")
                    # 填充NaN值
                    if col == 'text_length':
                        # 文本长度为0
                        self.df[col] = self.df[col].fillna(0)
                    elif col == 'punct_density':
                        # 标点密度为0
                        self.df[col] = self.df[col].fillna(0)
                    elif col == 'sentence_count':
                        # 句子数量为1
                        self.df[col] = self.df[col].fillna(1)
                    else:
                        # 其他特征填0
                        self.df[col] = self.df[col].fillna(0)
    
    def _oversample_extreme_valence(self):
        """对极端Valence值（大于0.7或小于-0.7）的样本进行上采样，每种上采样两倍"""
        # 获取V值极高的样本
        high_v_samples = self.df[self.df[self.v_col] > 0.7]
        # 获取V值极低的样本
        low_v_samples = self.df[self.df[self.v_col] < -0.7]
        
        # 添加上采样的极端样本
        if not high_v_samples.empty:
            # 上采样两倍意味着添加2份原始样本
            self.df = pd.concat([self.df, high_v_samples, high_v_samples], ignore_index=True)
            print(f"上采样高V值样本 ({len(high_v_samples)} -> {len(high_v_samples)*3})")
        
        if not low_v_samples.empty:
            # 上采样两倍意味着添加2份原始样本
            self.df = pd.concat([self.df, low_v_samples, low_v_samples], ignore_index=True)
            print(f"上采样低V值样本 ({len(low_v_samples)} -> {len(low_v_samples)*3})")
        
        print(f"上采样后数据集大小: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取文本并确保是字符串类型
        text = self.df.iloc[idx][self.text_col]
        if not isinstance(text, str):
            if pd.isna(text):
                text = ""
            else:
                text = str(text)
                
        # 获取情感标签
        valence = float(self.df.iloc[idx][self.v_col])  # V值
        arousal = float(self.df.iloc[idx][self.a_col])  # A值
        
        # 再次确保不会有NaN值 - 新增安全检查
        if math.isnan(valence) or math.isinf(valence):
            valence = 0.0
        if math.isnan(arousal) or math.isinf(arousal):
            arousal = 0.0
        
        # 对文本进行标记化
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.max_length, 
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        # 提取元特征
        meta_features = None
        if self.use_meta_features and len(self.meta_features) > 0:
            try:
                # 确保所有元特征都是浮点数且没有NaN
                meta_values = []
                for col in self.meta_features:
                    val = float(self.df.iloc[idx][col])
                    # 检查并处理可能的NaN或无限值
                    if math.isnan(val) or math.isinf(val):
                        val = 0.0
                    meta_values.append(val)
                
                meta_features = torch.tensor(meta_values, dtype=torch.float32)
            except Exception as e:
                # 如果出错，使用默认值
                print(f"警告: 处理元特征时出错: {str(e)}")
                meta_features = torch.zeros(len(self.meta_features), dtype=torch.float32)
        
        # 构建结果字典
        result = {
            'tokens': tokens,
            'targets': torch.tensor([valence, arousal], dtype=torch.float32),
            'length': min(len(text), self.max_length)
        }
        
        if meta_features is not None:
            result['meta_features'] = meta_features
        
        return result


class SimpleTokenizer:
    """简单分词器，适用于英文和中文"""
    
    def __init__(self, vocab_size=10000):
        """初始化分词器"""
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.vocab_size_actual = 2  # 初始词汇表大小
    
    def build_vocab(self, texts, min_freq=2):
        """从文本构建词汇表"""
        # 统计词频
        for text in texts:
            # 确保文本是字符串
            if not isinstance(text, str):
                if pd.isna(text):
                    text = ""
                else:
                    text = str(text)
            
            for char in text:
                if char not in self.word_counts:
                    self.word_counts[char] = 0
                self.word_counts[char] += 1
        
        # 按频率排序
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 添加到词汇表（满足最小频率要求）
        for word, count in sorted_words:
            if count < min_freq:
                continue
            if word not in self.word2idx and self.vocab_size_actual < self.vocab_size:
                self.word2idx[word] = self.vocab_size_actual
                self.idx2word[self.vocab_size_actual] = word
                self.vocab_size_actual += 1
    
    def encode(self, text, max_length=100, truncation=True, padding='max_length', return_tensors=None):
        """将文本编码为ID序列"""
        # 确保文本是字符串类型
        if not isinstance(text, str):
            if pd.isna(text):
                text = ""
            else:
                text = str(text)
                
        # 将文本转换为token IDs
        token_ids = []
        for char in text:
            if char in self.word2idx:
                token_ids.append(self.word2idx[char])
            else:
                token_ids.append(self.unk_token_id)
        
        # 截断
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # 填充
        if padding == 'max_length':
            padded_tokens = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
            token_ids = padded_tokens[:max_length]
        
        # 转换为张量
        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            return token_ids


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LTC-NCP-VA 训练脚本')
    parser.add_argument('--config', type=str, default='configs/bilingual_base.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch', type=int, help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, help='学习率（覆盖配置文件）')
    parser.add_argument('--seed', type=int, help='随机种子（覆盖配置文件）')
    parser.add_argument('--method', choices=['euler', 'rk4'], help='积分方法（覆盖配置文件）')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--amp', action='store_true', help='使用自动混合精度')
    return parser.parse_args()


def load_config(config_path, args):
    """加载并更新配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 用命令行参数覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch:
        config['training']['batch_size'] = args.batch
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.seed:
        config['hardware']['seed'] = args.seed
    if args.method:
        config['model']['integration_method'] = args.method
    if args.cpu:
        config['hardware']['device'] = 'cpu'
    
    return config


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(config):
    """准备数据集和分词器"""
    # 1. 从训练集文件加载文本
    train_en_df = pd.read_csv(config['data']['train']['en'])
    train_zh_df = pd.read_csv(config['data']['train']['zh'])
    
    # 2. 创建分词器并构建词汇表
    text_col = config['data']['text_col']
    # 确保文本列都是字符串类型
    train_en_df[text_col] = train_en_df[text_col].fillna("").astype(str)
    train_zh_df[text_col] = train_zh_df[text_col].fillna("").astype(str)
    
    all_texts = list(train_en_df[text_col]) + list(train_zh_df[text_col])
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.build_vocab(all_texts, min_freq=2)
    
    logger.info(f"词汇表大小: {tokenizer.vocab_size_actual}")
    
    # 3. 创建数据集
    max_length = config['data'].get('max_length', 100)
    
    # 英文数据集
    train_en_dataset = EmotionDataset(
        config['data']['train']['en'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=True
    )
    val_en_dataset = EmotionDataset(
        config['data']['val']['en'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=False
    )
    test_en_dataset = EmotionDataset(
        config['data']['test']['en'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=False
    )
    
    # 中文数据集
    train_zh_dataset = EmotionDataset(
        config['data']['train']['zh'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=True
    )
    val_zh_dataset = EmotionDataset(
        config['data']['val']['zh'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=False
    )
    test_zh_dataset = EmotionDataset(
        config['data']['test']['zh'], 
        tokenizer, 
        config['data'], 
        max_length, 
        is_training=False
    )
    
    # 4. 合并数据集
    train_dataset = ConcatDataset([train_en_dataset, train_zh_dataset])
    val_dataset = ConcatDataset([val_en_dataset, val_zh_dataset])
    test_dataset = ConcatDataset([test_en_dataset, test_zh_dataset])
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 5. 创建数据加载器
    batch_size = config['training']['batch_size']
    num_workers = config['hardware'].get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return tokenizer, train_loader, val_loader, test_loader


def create_model(config, vocab_size):
    """创建模型"""
    model_config = config['model']
    
    # 使用Transformer相关设置
    use_transformer = model_config.get('use_transformer', False)
    
    # 记录创建细节
    logger.info(f"创建模型: LTC-NCP-RNN | 词汇表大小: {vocab_size}")
    logger.info(f"集成方法: {model_config['integration_method']} | 隐藏层大小: {model_config['hidden_size']}")
    
    if use_transformer:
        transformer_config = model_config.get('transformer', {})
        num_layers = transformer_config.get('num_layers', 4)
        nhead = transformer_config.get('num_heads', 4)
        dim_feedforward = transformer_config.get('dim_feedforward', model_config['hidden_size']*2)
        transformer_dropout = transformer_config.get('dropout', model_config['dropout']/2)
        
        logger.info(f"启用Transformer架构: {num_layers}层, {nhead}头, 前馈维度: {dim_feedforward}")
    
    try:
        model = LTC_NCP_RNN(
            vocab_size=vocab_size,
            embedding_dim=model_config['embedding_dim'],
            hidden_size=model_config['hidden_size'],
            output_size=2,  # V和A
            dropout=model_config['dropout'],
            sparsity_level=model_config['sparsity_level'],
            dt=model_config['dt'],
            integration_method=model_config['integration_method'],
            use_meta_features=config['data'].get('use_meta_features', False),
            bidirectional=model_config.get('bidirectional', False),
            padding_idx=0,  # PAD token ID
            wiring_type=model_config.get('wiring_type', 'structured'),
            multi_level=model_config.get('multi_level', True),
            emotion_focused=model_config.get('emotion_focused', True),
            heterogeneous=model_config.get('heterogeneous', True),
            use_transformer=use_transformer
        )
        
        # 输出双向信息
        if model_config.get('bidirectional', False):
            logger.info("使用双向RNN结构")
        
        # 输出多层次信息
        if model_config.get('multi_level', True):
            logger.info("启用多层次LTC结构")
            
    except Exception as e:
        logger.error(f"创建模型时出错: {e}")
        raise
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 记录子模块信息
    try:
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            logger.info(f"模块 {name}: {params:,} 参数")
    except Exception as e:
        logger.warning(f"无法记录模块信息: {e}")
    
    # 验证模型结构
    device = config['hardware']['device']
    logger.info(f"模型将部署到设备: {device}")
    
    # 检查transformer组件设备一致性
    if use_transformer:
        logger.info("验证Transformer组件设备一致性...")
        # 移动模型到目标设备
        model = model.to(device)
        # 检查transformer组件
        transformer_device = next(model.transformer.parameters()).device
        logger.info(f"Transformer组件位于设备: {transformer_device}")
        # 检查特征适配器
        adapter_device = next(model.feature_adapter.parameters()).device
        logger.info(f"特征适配器位于设备: {adapter_device}")
    
    return model


def concordance_correlation_coefficient(y_true, y_pred):
    """计算一致性相关系数 (CCC)"""
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    
    y_true_var = torch.var(y_true, unbiased=False)
    y_pred_var = torch.var(y_pred, unbiased=False)
    
    covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc


def compute_metrics(y_true, y_pred):
    """计算所有评估指标"""
    # 转为NumPy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 检查并替换NaN值
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # 计算MSE、RMSE和MAE
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
    except Exception as e:
        print(f"计算整体指标时出错: {str(e)}")
        mse = 1.0
        rmse = 1.0
        mae = 1.0
    
    # 分别计算V和A的指标
    metrics_v = {}
    metrics_a = {}
    
    for i, dim in enumerate(['V', 'A']):
        try:
            y_true_dim = y_true[:, i]
            y_pred_dim = y_pred[:, i]
            
            # 再次检查并替换NaN值
            y_true_dim = np.nan_to_num(y_true_dim, nan=0.0)
            y_pred_dim = np.nan_to_num(y_pred_dim, nan=0.0)
            
            # 一致性相关系数
            ccc = concordance_correlation_coefficient(
                torch.tensor(y_true_dim),
                torch.tensor(y_pred_dim)
            ).item()
            
            # 确保CCC是有效值
            if math.isnan(ccc) or math.isinf(ccc):
                ccc = 0.0
            
            # Spearman相关系数
            try:
                spearman_corr, _ = spearmanr(y_true_dim, y_pred_dim)
                if math.isnan(spearman_corr) or math.isinf(spearman_corr):
                    spearman_corr = 0.0
            except Exception:
                spearman_corr = 0.0
            
            # Pearson相关系数（如果需要）
            try:
                pearson_corr, _ = pearsonr(y_true_dim, y_pred_dim)
                if math.isnan(pearson_corr) or math.isinf(pearson_corr):
                    pearson_corr = 0.0
            except Exception:
                pearson_corr = 0.0
            
            # MSE、RMSE和MAE
            mse_dim = mean_squared_error(y_true_dim, y_pred_dim)
            rmse_dim = np.sqrt(mse_dim)
            mae_dim = mean_absolute_error(y_true_dim, y_pred_dim)
            
            if dim == 'V':
                metrics_v = {
                    'mse': mse_dim,
                    'rmse': rmse_dim,
                    'mae': mae_dim,
                    'ccc': ccc,
                    'spearman': spearman_corr,
                    'pearson': pearson_corr
                }
            else:
                metrics_a = {
                    'mse': mse_dim,
                    'rmse': rmse_dim,
                    'mae': mae_dim,
                    'ccc': ccc,
                    'spearman': spearman_corr,
                    'pearson': pearson_corr
                }
        except Exception as e:
            print(f"计算{dim}维度指标时出错: {str(e)}")
            # 使用默认值
            default_metrics = {
                'mse': 1.0,
                'rmse': 1.0,
                'mae': 1.0,
                'ccc': 0.0,
                'spearman': 0.0,
                'pearson': 0.0
            }
            if dim == 'V':
                metrics_v = default_metrics
            else:
                metrics_a = default_metrics
    
    # 确保两个指标字典都存在
    if not metrics_v:
        metrics_v = {
            'mse': 1.0, 'rmse': 1.0, 'mae': 1.0,
            'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0
        }
    if not metrics_a:
        metrics_a = {
            'mse': 1.0, 'rmse': 1.0, 'mae': 1.0,
            'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0
        }
    
    # 合并指标
    metrics = {
        'overall': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'V': metrics_v,
        'A': metrics_a,
        'avg_ccc': (metrics_v['ccc'] + metrics_a['ccc']) / 2
    }
    
    return metrics


def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler=None, scaler=None):
    """训练一个轮次"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    valid_batches = 0  # 跟踪有效批次数量
    
    # 进度条
    pbar = tqdm(train_loader, desc="训练")
    
    # 获取优化配置
    grad_accum_steps = config.get('optimization', {}).get('gradient_accumulation_steps', 1)
    mixup_alpha = config.get('optimization', {}).get('mixup_alpha', 0)
    use_amp = config.get('hardware', {}).get('precision', '') == 'mixed' or scaler is not None
    freeze_a_head = config.get('optimization', {}).get('freeze_a_head', False)
    
    # 设置日志打印频率 - 减少日志输出
    log_interval = max(1, len(train_loader) // 4)  # 每批次的25%打印一次，改为每1/4打印
    
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
            
            # 检查元特征中是否有NaN
            if torch.isnan(meta_features).any():
                # 替换NaN为0
                meta_features = torch.nan_to_num(meta_features, nan=0.0)
                logger.debug(f"警告: 批次{i}中发现NaN元特征，已替换为0")
        
        # 检查目标值中是否有NaN
        if torch.isnan(targets).any():
            # 替换NaN为0（中性情感）
            targets = torch.nan_to_num(targets, nan=0.0)
            logger.debug(f"警告: 批次{i}中发现NaN目标值，已替换为0")
        
        # 应用Mixup数据增强（如果启用）
        if mixup_alpha > 0 and i % grad_accum_steps == 0:
            # 不要对tokens应用mixup，只对targets应用
            # tokens 保持不变，targets 进行混合
            _, mixed_targets = mixup_data(tokens, targets, mixup_alpha)
            targets = mixed_targets
        
        # 使用混合精度（如果启用）- 修复弃用警告
        with autocast('cuda') if use_amp else nullcontext():
            # 前向传播
            outputs = model(tokens, lengths, meta_features)
            
            # 检查输出是否包含NaN
            if torch.isnan(outputs).any():
                logger.debug(f"警告: 批次{i}包含NaN输出，跳过此批次...")
                # 更新进度条显示
                pbar.set_postfix({'loss': 'NaN-跳过'})
                continue
                
            loss = criterion(outputs, targets)
            
            # 检查损失是否为NaN
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.debug(f"警告: 批次{i}产生NaN损失，跳过此批次...")
                # 更新进度条显示
                pbar.set_postfix({'loss': 'NaN-跳过'})
                continue
                
            loss = loss / grad_accum_steps  # 归一化损失以支持梯度累积
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 检查梯度是否包含NaN
        valid_grads = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.debug(f"警告: 参数{name}的梯度包含NaN或Inf")
                    valid_grads = False
                    break
        
        # 如果梯度有问题，跳过此批次更新
        if not valid_grads:
            logger.debug(f"警告: 批次{i}梯度无效，跳过参数更新")
            # 清零梯度并继续
            optimizer.zero_grad()
            continue
        
        # 每grad_accum_steps步更新一次参数
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
            # 梯度裁剪
            if 'clip_grad_norm' in config['training']:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            
            # 使用scaler更新参数（如果启用AMP）
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
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
        
        # 更新进度条
        pbar.set_postfix({'loss': total_loss / max(1, valid_batches)})
        
        # 打印日志 - 减少记录频率
        if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
            logger.info(f"TRAIN: 批次 {i+1}/{len(train_loader)} | 损失: {loss.item() * grad_accum_steps:.4f}")
    
    # 确保我们至少处理了一些有效批次
    if valid_batches == 0:
        raise RuntimeError("所有批次都包含NaN，无法继续训练")
    
    # 计算平均损失
    avg_loss = total_loss / valid_batches
    
    # 计算指标 - 确保输入没有NaN
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 检查并过滤掉NaN
    valid_indices = ~(torch.isnan(all_preds).any(dim=1) | torch.isnan(all_targets).any(dim=1))
    
    if valid_indices.sum() == 0:
        logger.warning("警告: 所有预测或目标都包含NaN，无法计算指标")
        # 返回占位符指标
        placeholder_metrics = {
            'overall': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0},
            'V': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0},
            'A': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0},
            'avg_ccc': 0.0
        }
        return avg_loss, placeholder_metrics
    
    # 使用有效数据计算指标
    valid_preds = all_preds[valid_indices]
    valid_targets = all_targets[valid_indices]
    
    try:
        metrics = compute_metrics(valid_targets, valid_preds)
    except Exception as e:
        logger.error(f"计算指标时发生错误: {str(e)}")
        # 返回占位符指标
        placeholder_metrics = {
            'overall': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0},
            'V': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0},
            'A': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'ccc': 0.0, 'spearman': 0.0, 'pearson': 0.0},
            'avg_ccc': 0.0
        }
        return avg_loss, placeholder_metrics
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device, config):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # 获取硬件配置
    use_amp = config.get('hardware', {}).get('precision', '') == 'mixed'
    
    # 减少日志频率，默认只在开始和结束时记录
    log_interval = max(1, len(val_loader) // 2)  # 每一半记录一次
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="验证")):
            # 将数据移到设备
            tokens = batch['tokens'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['length']
            
            # 提取并处理元特征
            meta_features = None
            if config['data'].get('use_meta_features', False) and 'meta_features' in batch:
                meta_features = batch['meta_features'].to(device)
            
            # 使用混合精度（如果启用）
            with autocast('cuda') if use_amp else nullcontext():
                # 前向传播
                outputs = model(tokens, lengths, meta_features)
                loss = criterion(outputs, targets)
            
            # 更新统计信息
            total_loss += loss.item()
            
            # 收集预测和目标用于指标计算
            all_preds.append(outputs)
            all_targets.append(targets)
            
            # 偶尔记录进度
            if (i+1) % log_interval == 0 or (i+1) == len(val_loader):
                logger.debug(f"VAL: 批次 {i+1}/{len(val_loader)} | 损失: {loss.item():.4f}")
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_targets, all_preds)
    
    return avg_loss, metrics


# 创建空上下文管理器（用于条件autocast）
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return None


def create_scheduler(optimizer, config, train_loader):
    """创建学习率调度器"""
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', None)
    epochs = config['training']['epochs']
    
    if not scheduler_type:
        return None
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - scheduler_config.get('warmup_epochs', 0)
        )
    elif scheduler_type == 'cosine_with_restarts':
        warmup = scheduler_config.get('warmup_epochs', 0)
        restarts = scheduler_config.get('restarts', 1)
        restart_decay = scheduler_config.get('restart_decay', 1.0)
        
        # 计算每个重启周期的长度，确保至少为1
        cycle_length = max(1, int((epochs - warmup) / restarts))
        
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cycle_length,
            T_mult=1,
            eta_min=1e-6
        )
    elif scheduler_type == 'one_cycle':
        steps_per_epoch = len(train_loader)
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config['training']['learning_rate'],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3
        )
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min' if not config['evaluation']['higher_better'] else 'max',
            factor=0.5,
            patience=3
        )
    else:
        logger.warning(f"未知的调度器类型: {scheduler_type}，不使用调度器")
        return None


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config, args)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(config['logging']['save_dir'], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 设置随机种子
    set_seed(config['hardware']['seed'])
    
    # 设置设备
    device = torch.device(config['hardware']['device'])
    logger.info(f"设备: {device}")
    
    # 准备数据
    tokenizer, train_loader, val_loader, test_loader = prepare_data(config)
    
    # 创建模型
    model = create_model(config, tokenizer.vocab_size_actual)
    
    # 冻结嵌入层（如果配置）
    freeze_epochs = config.get('optimization', {}).get('freeze_embedding_epochs', 0)
    if freeze_epochs > 0:
        logger.info(f"前{freeze_epochs}轮冻结嵌入层")
        model.embedding.weight.requires_grad = False
    
    model.to(device)
    
    # 创建损失函数
    label_smoothing = config.get('optimization', {}).get('label_smoothing', 0)
    if label_smoothing > 0:
        criterion = LabelSmoothingLoss(smoothing=label_smoothing)
        logger.info(f"使用标签平滑: {label_smoothing}")
    else:
        criterion = MSE_CCC_Loss()
    
    # 创建优化器
    weight_decay = config['training'].get('weight_decay', 0)
    # 确保weight_decay是浮点数
    if isinstance(weight_decay, str):
        try:
            weight_decay = float(weight_decay)
        except ValueError:
            logger.warning(f"无效的weight_decay值: {weight_decay}，使用默认值0")
            weight_decay = 0.0
            
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=weight_decay
    )
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config, train_loader)
    if scheduler:
        scheduler_name = scheduler.__class__.__name__
        logger.info(f"学习率调度: {scheduler_name}")
    
    # 设置混合精度训练 - 更新GradScaler创建
    use_amp = config.get('hardware', {}).get('precision', '') == 'mixed' or args.amp
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("启用混合精度训练")
    
    # 创建TensorBoard摘要写入器
    if config['logging'].get('tensorboard', False):
        writer = SummaryWriter(log_dir=save_dir)
    else:
        writer = None
    
    # 训练循环
    best_metric = float('-inf') if config['evaluation']['higher_better'] else float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = config['evaluation'].get('patience', float('inf'))
    
    epochs = config['training']['epochs']
    logger.info(f"开始训练: {epochs}轮 | 批次大小: {config['training']['batch_size']}")
    
    # 冻结A头的轮数
    freeze_a_head_epochs = config.get('optimization', {}).get('freeze_a_head_epochs', 0)
    
    for epoch in range(1, epochs + 1):
        logger.info(f"===== Epoch {epoch}/{epochs} =====")
        
        # 解冻嵌入层（如果达到指定轮数）
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.info("解冻嵌入层")
            model.embedding.weight.requires_grad = True
        
        # 配置是否冻结A头
        if epoch <= freeze_a_head_epochs:
            config['optimization']['freeze_a_head'] = True
            logger.info(f"轮次 {epoch}/{freeze_a_head_epochs}: 冻结Arousal头部")
        else:
            # 解冻A头
            if epoch == freeze_a_head_epochs + 1 and freeze_a_head_epochs > 0:
                config['optimization']['freeze_a_head'] = False
                # 需要手动解冻A头相关参数
                for name, param in model.named_parameters():
                    if 'arousal_branch' in name:
                        param.requires_grad = True
                logger.info("解冻Arousal头部参数")
        
        # 训练一个轮次
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, config, 
            scheduler if isinstance(scheduler, optim.lr_scheduler.OneCycleLR) else None,
            scaler
        )
        
        # 在验证集上评估
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, config
        )
        
        # 打印指标
        logger.info(f"结果: 训练损失={train_loss:.4f} | 验证损失={val_loss:.4f} | RMSE={val_metrics['overall']['rmse']:.4f} | V-RMSE={val_metrics['V']['rmse']:.4f} | A-RMSE={val_metrics['A']['rmse']:.4f}")
        
        # 记录CCC指标，作为参考
        logger.info(f"参考CCC: V-CCC={val_metrics['V']['ccc']:.4f} | A-CCC={val_metrics['A']['ccc']:.4f}")
        
        # 更新学习率调度器
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # 使用选定的指标进行调度
                metric_name = config['evaluation']['best_metric']
                metric_value = val_metrics['avg_ccc'] if metric_name == 'ccc' else val_metrics['overall'][metric_name]
                scheduler.step(metric_value)
            else:
                scheduler.step()
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/val_overall_rmse', val_metrics['overall']['rmse'], epoch)
            writer.add_scalar('Metrics/val_v_rmse', val_metrics['V']['rmse'], epoch)
            writer.add_scalar('Metrics/val_a_rmse', val_metrics['A']['rmse'], epoch)
            # CCC作为次要指标
            writer.add_scalar('Metrics/val_v_ccc', val_metrics['V']['ccc'], epoch)
            writer.add_scalar('Metrics/val_a_ccc', val_metrics['A']['ccc'], epoch)
            writer.add_scalar('Metrics/val_avg_ccc', val_metrics['avg_ccc'], epoch)
            
            # 记录当前学习率
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/learning_rate', lr, epoch)
        
        # 检查是否为最佳模型
        metric_name = config['evaluation']['best_metric']
        if metric_name == 'ccc':
            current_metric = val_metrics['avg_ccc']
        else:
            current_metric = val_metrics['overall'][metric_name]
        
        is_better = False
        if config['evaluation']['higher_better']:
            is_better = current_metric > best_metric
        else:
            is_better = current_metric < best_metric
        
        # 保存最佳模型
        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(save_dir, 'model_best.pt'))
            
            logger.info(f"保存最佳模型: {metric_name}={best_metric:.4f}")
        else:
            patience_counter += 1
            logger.info(f"模型未改进: {patience_counter}/{patience}")
            
            # 如果达到耐心值，提前停止
            if patience_counter >= patience:
                logger.info(f"达到耐心值{patience}，提前停止训练")
                break
        
        # 定期保存检查点
        if config['logging'].get('save_checkpoints', False) and epoch % config['logging'].get('checkpoints_interval', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))
        
        # 保存最后一个轮次的模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': config
        }, os.path.join(save_dir, 'model_last.pt'))
    
    # 训练完成后的总结
    logger.info(f"训练完成! 最佳轮次: {best_epoch}, {metric_name}={best_metric:.4f}")
    
    # 关闭TensorBoard写入器
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main() 