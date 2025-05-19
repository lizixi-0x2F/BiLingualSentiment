#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 训练脚本
适用于中英双语情感价效度回归任务
简化版本: 移除高级功能
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
except AttributeError:
    pass

# 情感标签一致性检查函数
def check_emotion_label_consistency(df, v_col, a_col, text_col=None):
    """
    检查情感标签的一致性
    
    参数:
        df: 数据框
        v_col: 价值列名
        a_col: 效度列名
        text_col: 文本列名(可选，用于示例)
    
    返回:
        问题数据统计字典
    """
    stats = {
        "total_samples": len(df),
        "v_range": [df[v_col].min(), df[v_col].max()],
        "a_range": [df[a_col].min(), df[a_col].max()],
        "v_mean": df[v_col].mean(),
        "a_mean": df[a_col].mean(),
        "extreme_pos_v": sum(df[v_col] > 0.8),
        "extreme_neg_v": sum(df[v_col] < -0.8),
        "extreme_pos_a": sum(df[a_col] > 0.8),
        "extreme_neg_a": sum(df[a_col] < -0.8),
        "neutral_samples": sum((abs(df[v_col]) < 0.2) & (abs(df[a_col]) < 0.2))
    }
    
    # 象限分布统计
    stats["q1_count"] = sum((df[v_col] > 0) & (df[a_col] > 0))  # 喜悦/兴奋
    stats["q2_count"] = sum((df[v_col] > 0) & (df[a_col] < 0))  # 满足/平静
    stats["q3_count"] = sum((df[v_col] < 0) & (df[a_col] > 0))  # 愤怒/焦虑
    stats["q4_count"] = sum((df[v_col] < 0) & (df[a_col] < 0))  # 悲伤/抑郁
    
    # 检查不一致样本 (示例：带有"开心"的文本但V值为负)
    if text_col is not None:
        positive_keywords = ["开心", "高兴", "喜欢", "happy", "joy", "喜悦", "满意", "满足"]
        negative_keywords = ["难过", "伤心", "生气", "愤怒", "悲伤", "讨厌", "失望", "焦虑", "sad", "angry", "fear"]
        
        potential_issues = []
        
        # 检查带正面词但V为负的样本
        for keyword in positive_keywords:
            mask = df[text_col].str.contains(keyword, na=False) & (df[v_col] < -0.3)
            if sum(mask) > 0:
                samples = df[mask].sample(min(3, sum(mask)))
                for _, row in samples.iterrows():
                    potential_issues.append({
                        "text": row[text_col],
                        "v": row[v_col],
                        "a": row[a_col],
                        "issue": f"包含正面词'{keyword}'但V值为负"
                    })
        
        # 检查带负面词但V为正的样本
        for keyword in negative_keywords:
            mask = df[text_col].str.contains(keyword, na=False) & (df[v_col] > 0.3)
            if sum(mask) > 0:
                samples = df[mask].sample(min(3, sum(mask)))
                for _, row in samples.iterrows():
                    potential_issues.append({
                        "text": row[text_col],
                        "v": row[v_col],
                        "a": row[a_col],
                        "issue": f"包含负面词'{keyword}'但V值为正"
                    })
        
        stats["potential_issues"] = potential_issues
    
    return stats

# 检查添加的额外元特征是否可用
def add_sentence_count(df, text_col="text"):
    """添加句子数量特征"""
    # 简单计算句号、问号、感叹号的数量作为句子数的近似
    df["sentence_count"] = df[text_col].apply(
        lambda x: len([c for c in str(x) if c in ['.', '!', '?', '。', '！', '？']])
    )
    return df

# 标准MSE损失函数，移除了标签平滑

class MSE_CCC_Loss(nn.Module):
    """结合MSE和CCC的损失函数"""
    def __init__(self, mse_weight=1.0, ccc_weight=0.0):  # 改为只使用MSE
        super(MSE_CCC_Loss, self).__init__()
        self.mse_weight = mse_weight
        self.ccc_weight = ccc_weight
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # 处理四象限分类头的输出元组
        if isinstance(predictions, tuple) and len(predictions) == 2:
            # 如果是(va_pred, quadrant_logits)元组，只取va_pred部分
            predictions = predictions[0]
        
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

class EmotionDirectionLoss(nn.Module):
    """
    情感方向感知损失函数
    增加对错误方向(VA象限错误)的惩罚
    可选择重点关注valence维度
    """
    def __init__(self, base_loss=nn.MSELoss(), direction_weight=0.5, valence_weight=1.0):
        super(EmotionDirectionLoss, self).__init__()
        self.base_loss = base_loss
        self.direction_weight = direction_weight
        self.valence_weight = valence_weight
    
    def forward(self, predictions, targets):
        # 处理四象限分类头的输出元组
        if isinstance(predictions, tuple) and len(predictions) == 2:
            # 如果是(va_pred, quadrant_logits)元组，只取va_pred部分用于基础损失计算
            va_pred, _ = predictions
            base_loss_val = self.base_loss(predictions, targets)  # 传递整个元组，让基础损失函数处理
            # 但对于方向损失，我们只使用va_pred部分
            predictions = va_pred
        else:
            # 基础损失(如MSE)
            base_loss_val = self.base_loss(predictions, targets)
        
        # 计算方向损失 - 惩罚象限错误
        # 提取V和A维度
        pred_v, pred_a = predictions[:, 0], predictions[:, 1]
        target_v, target_a = targets[:, 0], targets[:, 1]
        
        # 计算符号是否一致(方向一致)
        v_sign_match = ((pred_v * target_v) >= 0)
        a_sign_match = ((pred_a * target_a) >= 0)
        
        # 方向损失 - 不同象限的样本会有较大惩罚
        # 使用带权重的组合 - 增加对V维度的关注
        v_loss = torch.mean(1.0 - v_sign_match.float()) * self.valence_weight
        a_loss = torch.mean(1.0 - a_sign_match.float())
        direction_loss = (v_loss + a_loss) / (1.0 + self.valence_weight)
        
        # 计算V和A各自的MSE损失
        v_mse = torch.mean((pred_v - target_v) ** 2) * self.valence_weight
        a_mse = torch.mean((pred_a - target_a) ** 2)
        
        # 总损失 - 基础损失加上方向损失
        total_loss = base_loss_val + self.direction_weight * direction_loss
        
        # 如果启用了valence_weight，增加valence的权重
        if self.valence_weight > 1.0:
            # 调整基础损失中的valence权重
            adjusted_loss = (v_mse + a_mse) / (1.0 + self.valence_weight)
            total_loss = adjusted_loss + self.direction_weight * direction_loss
        
        return total_loss


# QuadrantClassificationLoss类已移除（简化版无需多任务学习头）
# 该类用于处理四象限分类辅助任务与VA回归的联合损失函数
# 简化版只使用基础损失函数（MSE或方向感知损失）


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
        
        # 新增：词性特征（用于增强Valence判断）
        self.use_pos_features = config.get('use_pos_features', False)
        if self.use_pos_features:
            try:
                from src.core.pos_features import extract_combined_pos_features, preprocess_batch_texts
                logger.info("启用词性特征，增强Valence判断")
                self.extract_pos_features = extract_combined_pos_features
                
                # 预处理词性特征（批量处理以加速）
                if not is_training:  # 只为验证/测试集预处理，训练集在运行时处理
                    logger.info("预处理词性特征...")
                    texts = self.df[self.text_col].tolist()
                    self.pos_features_cache = preprocess_batch_texts(texts)
                    logger.info(f"预处理完成，shape={self.pos_features_cache.shape}")
                else:
                    self.pos_features_cache = None
            except ImportError:
                logger.warning("未找到词性特征模块，禁用词性特征")
                self.use_pos_features = False
        
        # 对Valence极端值样本进行上采样（仅在训练集中进行）
        if is_training and config.get('oversample_extreme_valence', False):
            self._oversample_extreme_valence()
            
        # 对四象限样本进行上采样（仅在训练集中进行）
        if is_training and config.get('oversample_quadrants', False):
            self._oversample_quadrants()
    
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
            for feature in self.config['meta_features']:
                if feature in self.df.columns and self.df[feature].isna().sum() > 0:
                    # 根据特征类型选择填充策略
                    if self.df[feature].dtype in [np.int64, np.int32, np.float64, np.float32]:
                        # 数值型特征用均值填充
                        self.df[feature] = self.df[feature].fillna(self.df[feature].mean())
                    else:
                        # 其他类型用众数填充
                        self.df[feature] = self.df[feature].fillna(self.df[feature].mode()[0])
    
    def _get_quadrant_info(self):
        """计算四象限样本信息，用于数据分析和采样策略"""
        # 计算每个象限的样本数
        q1_mask = (self.df[self.v_col] >= 0) & (self.df[self.a_col] >= 0)  # 喜悦/兴奋
        q2_mask = (self.df[self.v_col] >= 0) & (self.df[self.a_col] < 0)   # 满足/平静
        q3_mask = (self.df[self.v_col] < 0) & (self.df[self.a_col] >= 0)   # 愤怒/焦虑
        q4_mask = (self.df[self.v_col] < 0) & (self.df[self.a_col] < 0)    # 悲伤/抑郁
        
        q1_count = q1_mask.sum()
        q2_count = q2_mask.sum()
        q3_count = q3_mask.sum()
        q4_count = q4_mask.sum()
        
        # 计算象限索引，用于均衡采样
        quadrant_indices = {
            0: list(self.df[q1_mask].index),  # 第一象限索引
            1: list(self.df[q2_mask].index),  # 第二象限索引
            2: list(self.df[q3_mask].index),  # 第三象限索引
            3: list(self.df[q4_mask].index)   # 第四象限索引
        }
        
        # 记录各象限样本数及比例
        total = len(self.df)
        quadrant_counts = {
            0: q1_count,
            1: q2_count,
            2: q3_count,
            3: q4_count
        }
        
        quadrant_ratios = {
            0: q1_count / total,
            1: q2_count / total,
            2: q3_count / total,
            3: q4_count / total
        }
        
        return {
            'counts': quadrant_counts,
            'ratios': quadrant_ratios,
            'indices': quadrant_indices,
            'masks': {
                0: q1_mask,
                1: q2_mask,
                2: q3_mask,
                3: q4_mask
            }
        }
    
    def _create_sampling_weights(self):
        """创建样本权重，用于加权随机采样"""
        import numpy as np
        
        # 获取四象限信息
        quad_info = self._get_quadrant_info()
        
        # 创建样本权重数组
        sample_weights = np.ones(len(self.df))
        
        # 计算象限权重
        counts = np.array([quad_info['counts'][i] for i in range(4)])
        weights = 1.0 / np.sqrt(counts)
        # 归一化权重，使其和为4
        weights = weights * (4.0 / weights.sum())
        
        # 为每个样本分配权重
        for quadrant in range(4):
            mask = quad_info['masks'][quadrant]
            sample_weights[mask] = weights[quadrant]
        
        # 为极端VA值样本增加额外权重
        extreme_v_mask = (abs(self.df[self.v_col]) > 0.7)
        extreme_a_mask = (abs(self.df[self.a_col]) > 0.7)
        
        # 极端V值样本权重提升20%
        sample_weights[extreme_v_mask] *= 1.2
        # 极端A值样本权重提升10%
        sample_weights[extreme_a_mask] *= 1.1
        # 悲伤/抑郁象限额外增强20%
        sample_weights[quad_info['masks'][3]] *= 1.2
        
        # 记录象限样本统计信息
        logger.info(f"四象限样本分布:")
        logger.info(f"  Q1(喜悦/兴奋): {quad_info['counts'][0]} 样本, 权重: {weights[0]:.4f}")
        logger.info(f"  Q2(满足/平静): {quad_info['counts'][1]} 样本, 权重: {weights[1]:.4f}")
        logger.info(f"  Q3(愤怒/焦虑): {quad_info['counts'][2]} 样本, 权重: {weights[2]:.4f}")
        logger.info(f"  Q4(悲伤/抑郁): {quad_info['counts'][3]} 样本, 权重: {weights[3]:.4f}*1.2")
        
        return sample_weights
    
    def get_weighted_sampler(self):
        """创建WeightedRandomSampler用于训练数据加载器"""
        from torch.utils.data import WeightedRandomSampler
        import numpy as np
        
        # 获取样本权重
        sample_weights = torch.DoubleTensor(self._create_sampling_weights())
        
        # 创建加权采样器，样本数量等于原始数据集大小乘以一个系数
        sample_count = int(len(self.df) * 1.5)  # 增加50%的采样量
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=sample_count,
            replacement=True
        )
        
        return sampler
    
    def _oversample_extreme_valence(self):
        """对极端价值样本进行上采样，以平衡数据集（传统方法，不推荐使用）"""
        # 输出警告信息
        logger.warning("使用传统上采样方法，推荐改用WeightedRandomSampler")
        
        # 原始数据框
        original_df = self.df.copy()
        
        # 定义极端值范围
        extreme_pos = self.df[self.v_col] > 0.7
        extreme_neg = self.df[self.v_col] < -0.7
        
        # 计算上采样系数 (增加约50%的样本)
        pos_samples = self.df[extreme_pos]
        neg_samples = self.df[extreme_neg]
        
        # 连接原始数据框和上采样样本
        self.df = pd.concat([original_df, pos_samples, neg_samples], ignore_index=True)
        
        logger.info(f"Valence极端值上采样后，数据集大小从 {len(original_df)} 增加到 {len(self.df)}")
    
    def _oversample_quadrants(self):
        """对四象限样本进行上采样，平衡各象限的样本数量（传统方法，不推荐使用）"""
        # 输出警告信息
        logger.warning("使用传统上采样方法，推荐改用WeightedRandomSampler")
        
        # 原始数据框
        original_df = self.df.copy()
        
        # 获取四象限信息
        quad_info = self._get_quadrant_info()
        
        # 找出样本最多的象限
        max_count = max(quad_info['counts'].values())
        
        # 为每个象限创建上采样样本
        upsampled_dfs = [original_df]  # 从原始数据开始
        
        for quadrant in range(4):
            count = quad_info['counts'][quadrant]
            if count < max_count:
                # 获取该象限的样本
                mask = quad_info['masks'][quadrant]
                quadrant_samples = self.df[mask]
                
                # 采样补足差额
                upsample = quadrant_samples.sample(n=max_count - count, replace=True)
                upsampled_dfs.append(upsample)
        
        # 为第四象限额外增加样本（因为从混淆矩阵看这个象限样本最少且常被误分类）
        extra_q4_factor = 1.2  # 额外增加20%
        extra_q4_count = int(quad_info['counts'][3] * extra_q4_factor)
        extra_q4_samples = self.df[quad_info['masks'][3]].sample(n=extra_q4_count, replace=True)
        upsampled_dfs.append(extra_q4_samples)
        
        # 连接所有数据框
        self.df = pd.concat(upsampled_dfs, ignore_index=True)
        
        # 记录各象限原始和上采样后的数量
        new_quad_info = self._get_quadrant_info()
        
        logger.info(f"四象限上采样结果:")
        for quadrant in range(4):
            old_count = quad_info['counts'][quadrant]
            new_count = new_quad_info['counts'][quadrant]
            logger.info(f"  Q{quadrant+1}: {old_count} → {new_count} (+{new_count-old_count})")
        
        logger.info(f"上采样后，数据集总大小从 {len(original_df)} 增加到 {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 获取文本和情感标签
        text = str(row[self.text_col])
        valence = float(row[self.v_col])
        arousal = float(row[self.a_col])
        
        # 编码文本为token IDs
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 创建VA值张量
        va_values = torch.tensor([valence, arousal], dtype=torch.float32)
        
        # 构建字典结果
        result = {
            'tokens': tokens,
            'targets': va_values,
            'length': len(text)
        }
        
        # 元特征列表
        all_meta_features = []
        
        # 如果启用元特征，提取元特征值
        if self.use_meta_features:
            meta_features = []
            for feature in self.meta_features:
                value = row[feature]
                # 确保值是数值型的
                if isinstance(value, (int, float)):
                    meta_features.append(float(value))
                else:
                    try:
                        meta_features.append(float(value))
                    except (ValueError, TypeError):
                        # 对于非数值型特征，可以尝试进行编码或使用默认值
                        meta_features.append(0.0)
            
            all_meta_features.extend(meta_features)
        
        # 如果启用词性特征，提取词性特征
        if self.use_pos_features:
            if self.pos_features_cache is not None:
                # 使用预处理的特征
                pos_features = self.pos_features_cache[idx]
            else:
                # 实时处理特征
                pos_features = self.extract_pos_features(text)
            
            all_meta_features.extend(pos_features)
        
        # 如果有元特征或词性特征，添加到结果
        if all_meta_features:
            result['meta_features'] = torch.tensor(all_meta_features, dtype=torch.float32)
        
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
    """准备训练、验证和测试数据"""
    data_config = config['data']
    train_path = data_config['train_path']
    val_path = data_config.get('val_path', None)
    test_path = data_config.get('test_path', None)
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")
    if val_path and not os.path.exists(val_path):
        raise FileNotFoundError(f"验证数据文件不存在: {val_path}")
    if test_path and not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_path}")
    
    # 加载数据集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path else None
    
    # 检查训练集标签一致性
    logger.info("检查训练集标签一致性...")
    train_stats = check_emotion_label_consistency(
        train_df, 
        config['data']['v_col'], 
        config['data']['a_col'],
        config['data']['text_col']
    )
    
    # 输出统计信息
    logger.info(f"训练集大小: {train_stats['total_samples']}")
    logger.info(f"价值范围: {train_stats['v_range']}, 平均值: {train_stats['v_mean']:.3f}")
    logger.info(f"效度范围: {train_stats['a_range']}, 平均值: {train_stats['a_mean']:.3f}")
    logger.info(f"象限分布: Q1(喜悦)={train_stats['q1_count']}, Q2(满足)={train_stats['q2_count']}, Q3(愤怒)={train_stats['q3_count']}, Q4(悲伤)={train_stats['q4_count']}")
    
    # 检查并报告潜在问题样本
    if 'potential_issues' in train_stats and len(train_stats['potential_issues']) > 0:
        logger.warning(f"发现 {len(train_stats['potential_issues'])} 个潜在标注问题样本")
        for i, issue in enumerate(train_stats['potential_issues'][:5]):  # 只显示前5个
            logger.warning(f"问题样本 {i+1}: {issue['text']} | V={issue['v']:.2f}, A={issue['a']:.2f} | {issue['issue']}")
        
        # 保存问题样本到文件
        issue_df = pd.DataFrame(train_stats['potential_issues'])
        issue_path = os.path.join(config['output_dir'], 'potential_label_issues.csv')
        issue_df.to_csv(issue_path, index=False)
        logger.warning(f"问题样本已保存到: {issue_path}")
        
        # 添加交互确认
        if config.get('data', {}).get('confirm_training', False):
            confirm = input("检测到潜在标签问题，是否继续训练? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("用户选择退出训练")
                sys.exit(0)
        else:
            logger.info("已禁用训练确认，继续训练...")
    
    # 构建分词器
    max_vocab_size = data_config.get('vocab_size', 10000)
    tokenizer = SimpleTokenizer(vocab_size=max_vocab_size)
    
    # 构建词汇表（如果需要）
    if not hasattr(tokenizer, 'word2idx') or len(tokenizer.word2idx) <= 2:  # 只有PAD和UNK
        tokenizer.build_vocab([str(text) for text in train_df[data_config['text_col']]])
        
    # 创建数据集
    max_length = data_config.get('max_length', 100)
    
    train_dataset = EmotionDataset(
        train_path, 
        tokenizer, 
        data_config, 
        max_length=max_length,
        is_training=True
    )
    
    val_dataset = None
    if val_path:
        val_dataset = EmotionDataset(
            val_path, 
            tokenizer, 
            data_config, 
            max_length=max_length,
            is_training=False
        )
    
    # 创建数据加载器
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('hardware', {}).get('num_workers', 4)
    
    # 确定是否使用加权采样器
    use_weighted_sampler = config['data'].get('use_weighted_sampler', False)
    
    if use_weighted_sampler:
        # 使用加权随机采样器代替传统的上采样
        logger.info("使用加权随机采样器进行四象限均衡...")
        train_sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # 使用加权采样器
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info("已启用加权随机采样，样本数量: {}".format(len(train_sampler)))
    else:
        # 使用传统的随机打乱
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 修改：返回字典而不是元组
    return {
        'train_loader': train_loader, 
        'val_loader': val_loader, 
        'vocab_size': tokenizer.vocab_size_actual
    }


def create_model(config, vocab_size):
    """创建模型"""
    model_config = config['model']
    
    # 使用Transformer相关设置
    use_transformer = model_config.get('use_transformer', False)
    
    # 获取反转标志
    invert_valence = model_config.get('invert_valence', False)
    invert_arousal = model_config.get('invert_arousal', False)
    
    # 记录创建细节
    logger.info(f"创建模型: LTC-NCP-RNN | 词汇表大小: {vocab_size}")
    logger.info(f"集成方法: {model_config['integration_method']} | 隐藏层大小: {model_config['hidden_size']}")
    
    if invert_valence:
        logger.info("启用价值(V)预测反转")
    if invert_arousal:
        logger.info("启用效度(A)预测反转")
    
    if use_transformer:
        transformer_config = model_config.get('transformer', {})
        num_layers = transformer_config.get('num_layers', 4)
        nhead = transformer_config.get('num_heads', 4)
        dim_feedforward = transformer_config.get('dim_feedforward', model_config['hidden_size']*2)
        transformer_dropout = transformer_config.get('dropout', model_config['dropout']/2)
        
        logger.info(f"启用Transformer架构: {num_layers}层, {nhead}头, 前馈维度: {dim_feedforward}")
    
    try:
        # 准备额外参数
        extra_args = {}
        
        # 检查是否使用词性特征
        use_pos_features = config['data'].get('use_pos_features', False)
        if use_pos_features:
            pos_features_count = config['data'].get('pos_features_count', 15)
            extra_args['pos_features_count'] = pos_features_count
            logger.info(f"启用词性特征增强Valence判断，特征维度: {pos_features_count}")
        
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
            use_transformer=use_transformer,
            invert_valence=invert_valence,     # 添加价值反转参数
            invert_arousal=invert_arousal,     # 添加效度反转参数
            enhance_valence=model_config.get('enhance_valence', False),  # 添加价值增强参数
            valence_layers=model_config.get('valence_layers', 2),        # 添加价值层数参数
            use_quadrant_head=model_config.get('use_quadrant_head', False),  # 添加四象限分类头参数
            quadrant_weight=model_config.get('quadrant_weight', 0.3),        # 添加四象限分类权重参数
            **extra_args  # 添加额外参数
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
    # 确保输入是torch张量
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    
    y_true_var = torch.var(y_true, unbiased=False)
    y_pred_var = torch.var(y_pred, unbiased=False)
    
    covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc.item()  # 返回标量值而不是张量


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算每个维度的指标
    metrics_v = {}
    metrics_a = {}
    
    # 计算MSE（添加这两行）
    mse_v = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_a = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    
    # 计算RMSE
    rmse_v = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_a = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    
    # 计算MAE
    mae_v = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_a = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    
    # 计算相关系数
    corr_v, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
    corr_a, _ = pearsonr(y_true[:, 1], y_pred[:, 1])
    
    # 计算Spearman相关系数
    spearman_v, _ = spearmanr(y_true[:, 0], y_pred[:, 0])
    spearman_a, _ = spearmanr(y_true[:, 1], y_pred[:, 1])
    
    # 计算CCC
    ccc_v = concordance_correlation_coefficient(y_true[:, 0], y_pred[:, 0])
    ccc_a = concordance_correlation_coefficient(y_true[:, 1], y_pred[:, 1])
    
    # 汇总指标
    metrics_v = {
        "mse": mse_v,  # 添加MSE指标
        "rmse": rmse_v,
        "mae": mae_v,
        "corr": corr_v,
        "spearman": spearman_v,
        "ccc": ccc_v
    }
    
    metrics_a = {
        "mse": mse_a,  # 添加MSE指标
        "rmse": rmse_a,
        "mae": mae_a,
        "corr": corr_a,
        "spearman": spearman_a,
        "ccc": ccc_a
    }
    
    # 计算平均/整体指标
    avg_mse = (mse_v + mse_a) / 2  # 添加平均MSE
    avg_rmse = (rmse_v + rmse_a) / 2
    avg_mae = (mae_v + mae_a) / 2
    avg_corr = (corr_v + corr_a) / 2
    avg_spearman = (spearman_v + spearman_a) / 2
    avg_ccc = (ccc_v + ccc_a) / 2
    
    # 新增：计算象限准确率
    true_v, true_a = y_true[:, 0], y_true[:, 1]
    pred_v, pred_a = y_pred[:, 0], y_pred[:, 1]
    
    # 计算象限
    true_quadrant = np.zeros(len(true_v), dtype=int)
    true_quadrant[(true_v >= 0) & (true_a >= 0)] = 1  # 喜悦/兴奋
    true_quadrant[(true_v >= 0) & (true_a < 0)] = 2   # 满足/平静
    true_quadrant[(true_v < 0) & (true_a >= 0)] = 3   # 愤怒/焦虑
    true_quadrant[(true_v < 0) & (true_a < 0)] = 4    # 悲伤/抑郁
    
    pred_quadrant = np.zeros(len(pred_v), dtype=int)
    pred_quadrant[(pred_v >= 0) & (pred_a >= 0)] = 1
    pred_quadrant[(pred_v >= 0) & (pred_a < 0)] = 2
    pred_quadrant[(pred_v < 0) & (pred_a >= 0)] = 3
    pred_quadrant[(pred_v < 0) & (pred_a < 0)] = 4
    
    # 计算象限准确率
    quadrant_accuracy = np.mean(true_quadrant == pred_quadrant)
    
    # 计算象限F1分数
    from sklearn.metrics import f1_score
    try:
        quadrant_f1_macro = f1_score(true_quadrant, pred_quadrant, average='macro')
        quadrant_f1_weighted = f1_score(true_quadrant, pred_quadrant, average='weighted')
    except:
        quadrant_f1_macro = 0.0
        quadrant_f1_weighted = 0.0
    
    # 整合所有指标
    overall_metrics = {
        "mse": avg_mse,  # 添加MSE指标
        "rmse": avg_rmse,
        "mae": avg_mae,
        "corr": avg_corr,
        "spearman": avg_spearman,
        "ccc": avg_ccc
    }
    
    results = {
        "V": metrics_v,
        "A": metrics_a,
        "overall": overall_metrics,  # 确保包含overall键
        "avg_mse": avg_mse,  # 添加平均MSE
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_corr": avg_corr,
        "avg_spearman": avg_spearman,
        "avg_ccc": avg_ccc,
        "quadrant_accuracy": quadrant_accuracy,
        "quadrant_f1_macro": quadrant_f1_macro,
        "quadrant_f1_weighted": quadrant_f1_weighted
    }
    
    return results


def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler=None, scaler=None):
    """训练一个轮次"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    valid_batches = 0
    
    # 获取训练配置
    grad_accum_steps = config.get('optimization', {}).get('gradient_accumulation_steps', 1)
    mixup_alpha = config.get('optimization', {}).get('mixup_alpha', 0)
    freeze_a_head = config.get('optimization', {}).get('freeze_a_head_epochs', 0) > config.get('current_epoch', 0)
    
    # 硬件配置
    use_amp = config.get('hardware', {}).get('precision', '') == 'mixed'
      # 简化版: 移除FGM对抗训练
    use_adversarial = False  # 在简化版中不使用对抗训练
    fgm = None  # 保留变量以避免后续代码出错
      # 简化版: 移除边界样本权重计算器
    use_boundary_weights = False  # 在简化版中不使用边界样本权重
    boundary_weighter = None
    
    # 创建进度条
    pbar = tqdm(train_loader, desc="训练")
    
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
        
        # 新增: 计算并应用边界样本权重，如果启用
        sample_weights = None
        if boundary_weighter is not None:
            v_values = targets[:, 0]
            a_values = targets[:, 1]
            sample_weights = boundary_weighter.compute_sample_weights(v_values, a_values)
        
        # 使用混合精度（如果启用）- 修复弃用警告
        with autocast('cuda') if use_amp else nullcontext():
            # 前向传播 - 判断是否启用了四象限分类头
            model_output = model(tokens, lengths, meta_features)
            
            # 检查输出中是否包含NaN 
            # 如果model_output是元组(va_outputs, quadrant_logits)，则检查两者
            if isinstance(model_output, tuple):
                va_outputs, quadrant_logits = model_output
                if torch.isnan(va_outputs).any() or torch.isnan(quadrant_logits).any():
                    logger.debug(f"警告: 批次{i}包含NaN输出，跳过此批次...")
                    # 更新进度条显示
                    pbar.set_postfix({'loss': 'NaN-跳过'})
                    continue
            else:
                # 与原来相同的检查
                if torch.isnan(model_output).any():
                    logger.debug(f"警告: 批次{i}包含NaN输出，跳过此批次...")
                    # 更新进度条显示
                    pbar.set_postfix({'loss': 'NaN-跳过'})
                    continue
            
            # 计算损失 - 根据输出类型使用不同的方式
            # 检查model_output是否是带有四象限预测的元组
            if isinstance(model_output, tuple) and len(model_output) == 2:
                # 如果是带有四象限预测的元组，直接传递整个输出到损失函数
                loss = criterion(model_output, targets)
            else:
                # 否则，按照常规方式计算损失
                loss = criterion(model_output, targets)
            
            # 应用样本权重（如果已计算）
            if sample_weights is not None:
                # 对每个样本的损失应用权重后重新计算平均损失
                # 注意：criterion中的reduction应为'none'才能正确应用权重
                # 如果reduction不是'none'，这里简单地将整体损失乘以权重均值
                loss = loss * torch.mean(sample_weights)
            
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
          # 简化版: 移除对抗训练部分
        
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
        # 如果输出是(va_pred, quadrant_logits)的元组，只保存va_pred部分
        if isinstance(model_output, tuple):
            va_outputs, _ = model_output
            all_preds.append(va_outputs.detach())
        else:
            all_preds.append(model_output.detach())
        all_targets.append(targets.detach())
        
        # 更新进度条
        pbar.set_postfix({'loss': total_loss / max(1, valid_batches)})
        
        # 打印日志 - 减少记录频率
        if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
            logger.info(f"TRAIN: 批次 {i+1}/{len(train_loader)} | 损失: {loss.item() * grad_accum_steps:.4f}")
    
    # 更新边界样本权重器的轮次信息
    if boundary_weighter is not None:
        avg_loss = total_loss / max(1, valid_batches)
        boundary_weighter.update_epoch(config.get('current_epoch', 0), avg_loss)
        
        # 记录边界样本统计信息
        stats = boundary_weighter.get_stats()
        logger.info(f"边界样本统计: 比例={stats['boundary_sample_ratio']:.4f}, 权重倍增={stats['weight_multiplier']:.2f}")
    
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
                model_outputs = model(tokens, lengths, meta_features)
                
                # 计算损失 - 模型输出可能是VA预测或(VA预测,四象限logits)元组
                if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
                    # 如果是带有四象限预测的元组，直接传递整个输出到损失函数
                    loss = criterion(model_outputs, targets)
                else:
                    # 否则，按照常规方式计算损失
                    loss = criterion(model_outputs, targets)
            
            # 更新统计信息
            total_loss += loss.item()
            
            # 收集预测和目标用于指标计算
            # 如果输出是元组(VA预测,四象限logits)，只保存VA预测部分
            if isinstance(model_outputs, tuple):
                va_outputs, _ = model_outputs
                all_preds.append(va_outputs)
            else:
                all_preds.append(model_outputs)
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
    
    # 计算并记录象限混淆矩阵
    all_preds_np = all_preds.cpu().numpy()
    all_targets_np = all_targets.cpu().numpy()
    
    true_v, true_a = all_targets_np[:, 0], all_targets_np[:, 1]
    pred_v, pred_a = all_preds_np[:, 0], all_preds_np[:, 1]
    
    # 计算象限
    true_quadrant = np.zeros(len(true_v), dtype=int)
    true_quadrant[(true_v >= 0) & (true_a >= 0)] = 1  # 喜悦/兴奋
    true_quadrant[(true_v >= 0) & (true_a < 0)] = 2   # 满足/平静
    true_quadrant[(true_v < 0) & (true_a >= 0)] = 3   # 愤怒/焦虑
    true_quadrant[(true_v < 0) & (true_a < 0)] = 4    # 悲伤/抑郁
    
    pred_quadrant = np.zeros(len(pred_v), dtype=int)
    pred_quadrant[(pred_v >= 0) & (pred_a >= 0)] = 1
    pred_quadrant[(pred_v >= 0) & (pred_a < 0)] = 2
    pred_quadrant[(pred_v < 0) & (pred_a >= 0)] = 3
    pred_quadrant[(pred_v < 0) & (pred_a < 0)] = 4
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_quadrant, pred_quadrant, labels=[1, 2, 3, 4])
    
    # 添加到结果
    metrics["confusion_matrix"] = cm.tolist()
    
    # 输出混淆矩阵
    quadrant_names = ["喜悦/兴奋", "满足/平静", "愤怒/焦虑", "悲伤/抑郁"]
    logger.info("情感象限混淆矩阵:")
    cm_str = ""
    cm_str += "预测 →\n实际 ↓ | " + " | ".join(f"{name}" for name in quadrant_names) + "\n"
    cm_str += "-" * 60 + "\n"
    for i, name in enumerate(quadrant_names):
        cm_str += f"{name} | " + " | ".join(f"{cm[i, j]:5d}" for j in range(4)) + "\n"
    logger.info("\n" + cm_str)
    
    # 如果有明显的象限反转，输出警告
    if cm[0, 3] > cm[0, 0] or cm[3, 0] > cm[3, 3]:
        logger.warning("⚠️ 检测到可能的情感象限反转！正面情感被预测为负面，或负面情感被预测为正面")
    
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
    
    # 设置随机种子
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)
    
    # 初始化设备
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 准备数据
    data = prepare_data(config)
    
    # 创建模型
    vocab_size = data['vocab_size']
    model = create_model(config, vocab_size)
    model = model.to(device)
    
    # 创建优化器
    optimizer_type = config.get('training', {}).get('optimizer', 'adam').lower()
    lr = config.get('training', {}).get('learning_rate', 1e-3)
    weight_decay = config.get('training', {}).get('weight_decay', 0)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('training', {}).get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config, data['train_loader'])
    
    # 创建损失函数
    base_loss = None
    if config.get('loss', {}).get('use_mse_ccc_loss', False):
        mse_weight = config.get('loss', {}).get('mse_weight', 1.0)
        ccc_weight = config.get('loss', {}).get('ccc_weight', 0.0)
        base_loss = MSE_CCC_Loss(mse_weight=mse_weight, ccc_weight=ccc_weight)    else:
        # 简化版本使用MSE损失
        base_loss = nn.MSELoss()
      # 检查是否使用方向感知损失
    if config.get('loss', {}).get('use_direction_loss', False):
        direction_weight = config.get('loss', {}).get('direction_weight', 0.5)
        valence_weight = config.get('loss', {}).get('valence_weight', 1.0)
        criterion = EmotionDirectionLoss(base_loss=base_loss, direction_weight=direction_weight, valence_weight=valence_weight)
    else:
        # 简化版本只使用基础损失函数
        criterion = base_loss
    
    # 是否启用混合精度
    use_amp = config.get('hardware', {}).get('precision', '') == 'mixed'
    scaler = GradScaler() if use_amp else None
    
    # 初始化日志器
    log_dir = config.get('logging', {}).get('log_dir', 'runs/default')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练设置
    epochs = args.epochs or config.get('training', {}).get('epochs', 10)
    # 记录轮次到配置中，用于其他组件访问
    config['current_epoch'] = 0
    
    # 初始化最佳模型跟踪
    best_metric_value = float('inf')
    best_metric_name = config.get('evaluation', {}).get('best_metric', 'rmse')
    higher_better = config.get('evaluation', {}).get('higher_better', False)
    patience = config.get('evaluation', {}).get('patience', 5)
    
    if higher_better:
        best_metric_value = float('-inf')
    
    epochs_without_improvement = 0
    
    # 训练循环
    for epoch in range(epochs):
        # 更新当前轮次
        config['current_epoch'] = epoch
        
        # 检查是否应该解冻A头
        if epoch >= config.get('optimization', {}).get('freeze_a_head_epochs', 0):
            # 只有当之前冻结过才需要解冻
            if epoch == config.get('optimization', {}).get('freeze_a_head_epochs', 0):
                for name, param in model.named_parameters():
                    if 'arousal_branch' in name:
                        param.requires_grad = True
                logger.info("已解冻Arousal头部参数")
        
        # 训练一个轮次
        train_loss, train_metrics = train_epoch(
            model=model,
            train_loader=data['train_loader'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            scheduler=scheduler if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
            scaler=scaler
        )
        
        # 验证
        val_loss, val_metrics = validate(model, data['val_loader'], criterion, device, config)
        
        # 更新学习率（如果使用轮次级调度器）
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 记录指标
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # 记录V和A的单独指标
        for dim, dim_name in [(0, 'V'), (1, 'A')]:
            for metric in ['mse', 'rmse', 'mae', 'ccc', 'spearman', 'pearson']:
                if metric in val_metrics[dim_name]:
                    writer.add_scalar(f'Metrics/{dim_name}_{metric}/val', val_metrics[dim_name][metric], epoch)
                    writer.add_scalar(f'Metrics/{dim_name}_{metric}/train', train_metrics[dim_name][metric], epoch)
        
        # 记录四象限分类准确率，如果有
        if 'quadrant_accuracy' in val_metrics:
            writer.add_scalar('Metrics/quadrant_accuracy/val', val_metrics['quadrant_accuracy'], epoch)
            writer.add_scalar('Metrics/quadrant_accuracy/train', train_metrics['quadrant_accuracy'], epoch)
        
        # 记录平均指标
        if 'avg_ccc' in val_metrics:
            writer.add_scalar('Metrics/avg_ccc/val', val_metrics['avg_ccc'], epoch)
        
        # 打印进度
        logger.info(f"轮次 {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 学习率: {current_lr:.6f}")
        
        # 更新最佳模型
        current_metric = None
        if best_metric_name == 'loss':
            current_metric = val_loss
        elif best_metric_name in val_metrics['overall']:
            current_metric = val_metrics['overall'][best_metric_name]
        elif best_metric_name == 'avg_ccc' and 'avg_ccc' in val_metrics:
            current_metric = val_metrics['avg_ccc']
        else:
            # 默认使用总体RMSE
            current_metric = val_metrics['overall']['rmse']
            logger.warning(f"配置的最佳指标 '{best_metric_name}' 不可用，退回到使用RMSE")
        
        # 输出详细的验证指标
        logger.info(f"验证指标 - V: MSE={val_metrics['V']['mse']:.4f}, RMSE={val_metrics['V']['rmse']:.4f}, CCC={val_metrics['V']['ccc']:.4f}")
        logger.info(f"验证指标 - A: MSE={val_metrics['A']['mse']:.4f}, RMSE={val_metrics['A']['rmse']:.4f}, CCC={val_metrics['A']['ccc']:.4f}")
        if 'quadrant_accuracy' in val_metrics:
            logger.info(f"验证指标 - 四象限准确率: {val_metrics['quadrant_accuracy']:.4f}")
        
        if (higher_better and current_metric > best_metric_value) or (not higher_better and current_metric < best_metric_value):
            # 更新最佳指标
            best_metric_value = current_metric
            epochs_without_improvement = 0
            
            # 保存最佳模型
            if config.get('logging', {}).get('save_model', False):
                save_dir = config.get('logging', {}).get('save_dir', 'runs/default')
                os.makedirs(save_dir, exist_ok=True)
                
                model_info = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'best_metric': best_metric_value,
                    'best_metric_name': best_metric_name,
                    'val_metrics': val_metrics,
                    'config': config
                }
                
                torch.save(model_info, os.path.join(save_dir, 'best_model.pt'))
                logger.info(f"保存最佳模型 @ 轮次 {epoch+1}，{best_metric_name}={best_metric_value:.4f}")
        else:
            epochs_without_improvement += 1
        
        # 是否保存检查点
        if config.get('logging', {}).get('save_checkpoints', False):
            if (epoch + 1) % config.get('logging', {}).get('checkpoints_interval', 5) == 0:
                save_dir = config.get('logging', {}).get('save_dir', 'runs/default')
                os.makedirs(save_dir, exist_ok=True)
                
                checkpoint_info = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'best_metric': best_metric_value,
                    'best_metric_name': best_metric_name,
                    'val_metrics': val_metrics,
                    'config': config
                }
                
                torch.save(checkpoint_info, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
                logger.info(f"保存检查点 @ 轮次 {epoch+1}")
        
        # 早停
        if epochs_without_improvement >= patience:
            logger.info(f"早停：{patience}轮次内未改善 {best_metric_name}")
            break
    
    # 记录训练完成
    logger.info(f"训练完成! 最佳{best_metric_name}: {best_metric_value:.4f}")
    
    # 关闭SummaryWriter
    writer.close()

if __name__ == "__main__":
    main() 