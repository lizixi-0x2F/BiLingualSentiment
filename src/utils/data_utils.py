import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class BilingualSentimentDataset(Dataset):
    """双语情感数据集，适用于预训练模型处理"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签数组 (valence, arousal)
            tokenizer: 预训练模型的tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])  # 确保文本是字符串
        label = self.labels[idx]
        
        # 使用预训练模型的tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 去除batch维度
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
        
        # 某些模型需要token_type_ids
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
            
        return item

def prepare_pretrained_dataloaders(config, tokenizer, batch_size=None):
    """
    准备用于预训练模型的数据加载器
    
    Args:
        config: 配置对象
        tokenizer: 预训练模型的tokenizer
        batch_size: 批次大小（可选，默认使用config.BATCH_SIZE）
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    # 加载中文数据集
    logger.info("加载中文情感数据集...")
    chinese_df = pd.read_csv(config.CHINESE_DATASET_PATH)
    chinese_texts = chinese_df['text'].tolist()
    
    # 提取标签并转换为NumPy数组
    chinese_labels = chinese_df[['valence', 'arousal']].values
    
    # 加载英文数据集
    logger.info("加载英文情感数据集...")
    english_df = pd.read_csv(config.EMOBANK_DATASET_PATH)
    english_texts = english_df['text'].tolist()
    english_labels = english_df[['valence', 'arousal']].values
    
    # 合并数据集
    all_texts = chinese_texts + english_texts
    all_labels = np.vstack((chinese_labels, english_labels))
    
    # 标准化标签值到[-1, 1]范围
    logger.info("将标签归一化到[-1, 1]范围...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    all_labels = scaler.fit_transform(all_labels)
    
    # 划分数据集
    logger.info("划分数据集为训练/验证/测试集...")
    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO
    test_ratio = config.TEST_RATIO
    
    # 首先划分出训练集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels, test_size=(1 - train_ratio), random_state=42
    )
    
    # 从剩余数据中划分验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=(1 - val_size), random_state=42
    )
    
    # 检查划分结果
    logger.info(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}, 测试集大小: {len(test_texts)}")
    
    # 创建数据集
    train_dataset = BilingualSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = BilingualSentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = BilingualSentimentDataset(test_texts, test_labels, tokenizer)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_dataloader, val_dataloader, test_dataloader
