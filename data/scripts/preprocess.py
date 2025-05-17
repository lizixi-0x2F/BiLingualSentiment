#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理脚本：清洗、规范化和增强数据集
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_emobank(file_path):
    """加载并预处理英文EmoBank数据集"""
    df = pd.read_csv(file_path)
    print(f"加载了 {len(df)} 条英文情感样本")
    
    # 将 'valence' 和 'arousal' 重命名为 'V' 和 'A'
    df = df.rename(columns={'valence': 'V', 'arousal': 'A'})
    
    return df

def load_chinese_va(file_path):
    """加载并预处理中文VA数据集"""
    df = pd.read_csv(file_path)
    print(f"加载了 {len(df)} 条中文情感样本")
    
    # 将 'valence' 和 'arousal' 重命名为 'V' 和 'A'
    df = df.rename(columns={'valence': 'V', 'arousal': 'A'})
    
    return df

def extract_features(text_series):
    """提取元特征：句长和标点密度"""
    # 处理可能的NaN值
    text_series = text_series.fillna("").astype(str)
    
    # 句长特征
    length = text_series.str.len()
    
    # 标点密度特征（标点数量/句长）
    punctuation = ',.?!;:，。？！；：'
    punct_counts = text_series.apply(lambda x: sum(1 for c in x if c in punctuation))
    
    # 避免除以零
    length_safe = length.apply(lambda x: max(x, 1))
    punct_density = punct_counts / length_safe
    
    return pd.DataFrame({
        'text_length': length,
        'punct_density': punct_density
    })

def split_and_save(df, output_dir, prefix, text_col='text', test_size=0.2, val_size=0.1):
    """分割并保存训练/验证/测试集"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 先分离测试集
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    
    # 从剩余数据中分离验证集
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    
    # 提取元特征
    train_meta = extract_features(train[text_col])
    val_meta = extract_features(val[text_col])
    test_meta = extract_features(test[text_col])
    
    # 合并元特征
    train = pd.concat([train, train_meta], axis=1)
    val = pd.concat([val, val_meta], axis=1)
    test = pd.concat([test, test_meta], axis=1)
    
    # 保存
    train.to_csv(os.path.join(output_dir, f"{prefix}_train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, f"{prefix}_val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, f"{prefix}_test.csv"), index=False)
    
    print(f"保存数据分割：训练集 {len(train)}，验证集 {len(val)}，测试集 {len(test)}")
    return train, val, test

def main():
    # 定义输入和输出路径
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(data_dir, "processed")
    
    # 创建处理后的数据目录
    os.makedirs(processed_dir, exist_ok=True)
    
    # 加载并处理英文数据集
    eng_path = os.path.join(data_dir, "emobank_va_normalized.csv")
    eng_df = load_emobank(eng_path)
    
    # 英文数据集列名
    eng_text_col = 'text'  # 文本列
    
    split_and_save(eng_df, processed_dir, 'eng', text_col=eng_text_col)
    
    # 加载并处理中文数据集
    chn_path = os.path.join(data_dir, "Chinese_VA_dataset_gaussNoise.csv")
    chn_df = load_chinese_va(chn_path)
    
    # 中文数据集列名
    chn_text_col = 'text'  # 文本列
    
    split_and_save(chn_df, processed_dir, 'chn', text_col=chn_text_col)
    
    print("所有数据预处理完成！")

if __name__ == "__main__":
    main() 