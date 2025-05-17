import pandas as pd
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..models.text_encoder import SimpleTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """情感数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表 (valence, arousal)
            tokenizer: 分词器 (SimpleTokenizer)
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 数据统计
        labels_array = np.array(labels)
        logger.info(f"标签统计 - 均值: {np.mean(labels_array, axis=0)}, 标准差: {np.std(labels_array, axis=0)}")
        logger.info(f"标签范围 - 最小值: {np.min(labels_array, axis=0)}, 最大值: {np.max(labels_array, axis=0)}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用分词器处理文本
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = encoded
        
        # 创建注意力掩码
        attention_mask = torch.zeros_like(encoded)
        attention_mask[encoded != self.tokenizer.word2idx['[PAD]']] = 1
        
        # 转换标签为tensor
        label = torch.tensor(label, dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


def load_chinese_data(file_path):
    """
    加载中文VA数据集
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        texts: 文本列表
        labels: 标签列表 (valence, arousal)
    """
    df = pd.read_csv(file_path)
    
    # 打印列名，帮助调试
    logger.info(f"Chinese dataset columns: {df.columns.tolist()}")
    
    # 使用正确的小写列名
    texts = df['text'].tolist()
    
    # 检查数据质量
    logger.info(f"数据集大小: {len(texts)}行")
    logger.info(f"检查缺失值: text={df['text'].isna().sum()}, valence={df['valence'].isna().sum()}, arousal={df['arousal'].isna().sum()}")
    
    # 使用小写列名
    valence = df['valence'].tolist()
    arousal = df['arousal'].tolist()
    
    # 检查数值范围
    logger.info(f"Valence 范围: [{min(valence)}, {max(valence)}], 均值: {sum(valence)/len(valence)}")
    logger.info(f"Arousal 范围: [{min(arousal)}, {max(arousal)}], 均值: {sum(arousal)/len(arousal)}")
    
    # 组合成标签对
    labels = list(zip(valence, arousal))
    
    return texts, labels


def load_emobank_data(file_path):
    """
    加载EmoBank数据集
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        texts: 文本列表
        labels: 标签列表 (valence, arousal)
    """
    df = pd.read_csv(file_path)
    
    # 打印列名，帮助调试
    logger.info(f"Emobank dataset columns: {df.columns.tolist()}")
    
    # 使用正确的列名
    texts = df['text'].tolist()
    
    # 检查数据质量
    logger.info(f"数据集大小: {len(texts)}行")
    logger.info(f"检查缺失值: text={df['text'].isna().sum()}, valence={df['valence'].isna().sum()}, arousal={df['arousal'].isna().sum()}")
    
    # 使用列名 valence 和 arousal
    valence = df['valence'].tolist()
    arousal = df['arousal'].tolist()
    
    # 检查数值范围
    logger.info(f"Valence 范围: [{min(valence)}, {max(valence)}], 均值: {sum(valence)/len(valence)}")
    logger.info(f"Arousal 范围: [{min(arousal)}, {max(arousal)}], 均值: {sum(arousal)/len(arousal)}")
    
    # 组合成标签对
    labels = list(zip(valence, arousal))
    
    return texts, labels


def standardize_labels(train_labels, val_labels, test_labels):
    """
    标准化标签值，将所有标签缩放到[-1, 1]范围内
    
    Args:
        train_labels: 训练集标签
        val_labels: 验证集标签
        test_labels: 测试集标签
        
    Returns:
        standardized_train_labels: 标准化后的训练集标签
        standardized_val_labels: 标准化后的验证集标签
        standardized_test_labels: 标准化后的测试集标签
        scaler: 标准化器
    """
    # 将标签转换为数组
    train_labels_array = np.array(train_labels)
    
    # 创建并拟合标准化器 - 使用MinMaxScaler将值缩放到[-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    standardized_train_array = scaler.fit_transform(train_labels_array)
    
    # 应用于验证集和测试集
    standardized_val_array = scaler.transform(np.array(val_labels))
    standardized_test_array = scaler.transform(np.array(test_labels))
    
    # 转换回列表
    standardized_train_labels = [tuple(x) for x in standardized_train_array]
    standardized_val_labels = [tuple(x) for x in standardized_val_array]
    standardized_test_labels = [tuple(x) for x in standardized_test_array]
    
    # 打印标准化前后的统计信息
    logger.info(f"标准化前 - 训练集均值: {np.mean(train_labels_array, axis=0)}")
    logger.info(f"标准化后 - 训练集均值: {np.mean(standardized_train_array, axis=0)}")
    logger.info(f"标准化后 - 训练集范围: 最小值={np.min(standardized_train_array, axis=0)}, 最大值={np.max(standardized_train_array, axis=0)}")
    
    return standardized_train_labels, standardized_val_labels, standardized_test_labels, scaler


def prepare_dataloaders(config):
    """
    准备数据加载器
    
    Args:
        config: 配置对象
        
    Returns:
        train_dataloader: 训练集数据加载器
        val_dataloader: 验证集数据加载器
        test_dataloader: 测试集数据加载器
        tokenizer: 分词器
    """
    # 同时加载中英文数据集
    logger.info(f"加载中文数据集: {config.CHINESE_DATASET_PATH}")
    chinese_texts, chinese_labels = load_chinese_data(config.CHINESE_DATASET_PATH)
    
    logger.info(f"加载英文数据集: {config.EMOBANK_DATASET_PATH}")
    english_texts, english_labels = load_emobank_data(config.EMOBANK_DATASET_PATH)
    
    # 合并数据集
    all_texts = chinese_texts + english_texts
    all_labels = chinese_labels + english_labels
    
    logger.info(f"合并后的数据集大小: 中文={len(chinese_texts)}，英文={len(english_texts)}，总计={len(all_texts)}")
    
    # 数据分割
    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO / (1 - train_ratio)  # 调整验证集比例
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels, train_size=train_ratio, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, train_size=val_ratio, random_state=42
    )
    
    # 标准化标签
    train_labels, val_labels, test_labels, _ = standardize_labels(
        train_labels, val_labels, test_labels
    )
    
    # 创建自定义分词器
    tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
    
    # 构建词汇表
    if len(train_texts) > 1000:
        # 只使用部分数据构建词汇表，以加快速度
        sample_size = min(20000, len(train_texts))
        logger.info(f"使用 {sample_size} 个文本样本构建词汇表...")
        tokenizer.build_vocab(train_texts[:sample_size])
    else:
        # 使用所有训练数据构建词汇表
        logger.info(f"使用全部 {len(train_texts)} 个训练文本构建词汇表...")
        tokenizer.build_vocab(train_texts)
    
    # 创建数据集
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE
    )
    
    return train_dataloader, val_dataloader, test_dataloader, tokenizer