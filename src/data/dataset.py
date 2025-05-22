import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer
import numpy as np

class VADataset(Dataset):
    """双语情感价效和唤起数据集"""
    
    def __init__(self, texts, valence_labels, arousal_labels, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            texts (List[str]): 文本列表
            valence_labels (List[float]): 价效标签
            arousal_labels (List[float]): 唤起标签
            tokenizer: 分词器
            max_length (int): 最大序列长度
        """
        self.texts = texts
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        valence = float(self.valence_labels[idx])
        arousal = float(self.arousal_labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor([valence, arousal], dtype=torch.float)
        }

def load_and_prepare_data(chinese_data_path, english_data_path, tokenizer, max_length=128, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    加载和准备中英文数据集
    
    Args:
        chinese_data_path (str): 中文数据集路径
        english_data_path (str): 英文数据集路径
        tokenizer: 分词器
        max_length (int): 最大序列长度
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
        
    Returns:
        dict: 包含训练集、验证集和测试集的字典
    """
    # 加载中文数据集
    chinese_df = pd.read_csv(chinese_data_path)
    chinese_texts = chinese_df['text'].tolist()
    chinese_valence = chinese_df['valence'].tolist()
    chinese_arousal = chinese_df['arousal'].tolist()
    
    # 加载英文数据集
    english_df = pd.read_csv(english_data_path)
    english_texts = english_df['text'].tolist()
    english_valence = english_df['valence'].tolist()
    english_arousal = english_df['arousal'].tolist()
    
    # 合并数据
    all_texts = chinese_texts + english_texts
    all_valence = chinese_valence + english_valence
    all_arousal = chinese_arousal + english_arousal
    
    # 分割数据集
    train_texts, temp_texts, train_valence, temp_valence, train_arousal, temp_arousal = train_test_split(
        all_texts, all_valence, all_arousal, test_size=val_ratio+test_ratio, random_state=seed
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_texts, test_texts, val_valence, test_valence, val_arousal, test_arousal = train_test_split(
        temp_texts, temp_valence, temp_arousal, test_size=1-val_ratio_adjusted, random_state=seed
    )
    
    # 创建数据集
    train_dataset = VADataset(train_texts, train_valence, train_arousal, tokenizer, max_length)
    val_dataset = VADataset(val_texts, val_valence, val_arousal, tokenizer, max_length)
    test_dataset = VADataset(test_texts, test_valence, test_arousal, tokenizer, max_length)
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

def create_dataloaders(datasets, batch_size=32, num_workers=4):
    """
    创建数据加载器
    
    Args:
        datasets (dict): 包含训练集、验证集和测试集的字典
        batch_size (int): 批量大小
        num_workers (int): 数据加载的工作线程数
        
    Returns:
        dict: 包含训练集、验证集和测试集数据加载器的字典
    """
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders 