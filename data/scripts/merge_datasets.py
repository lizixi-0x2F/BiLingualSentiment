#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集合并脚本：将中英文数据集合并并保存到processed_clean目录
"""

import pandas as pd
import os
import numpy as np

def count_sentences(text):
    """计算句子数量，同时处理NaN值"""
    if pd.isna(text):
        return 0
    return sum(1 for i in str(text).split(r'[.!?。！？]') if i.strip())

def main():
    # 定义路径
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(data_dir, "processed")
    clean_dir = os.path.join(data_dir, "processed_clean")
    
    # 确保输出目录存在
    os.makedirs(clean_dir, exist_ok=True)
    
    # 加载中英文训练集
    eng_train = pd.read_csv(os.path.join(processed_dir, "eng_train.csv"))
    chn_train = pd.read_csv(os.path.join(processed_dir, "chn_train.csv"))
    
    # 加载中英文验证集
    eng_val = pd.read_csv(os.path.join(processed_dir, "eng_val.csv"))
    chn_val = pd.read_csv(os.path.join(processed_dir, "chn_val.csv"))
    
    # 加载中英文测试集
    eng_test = pd.read_csv(os.path.join(processed_dir, "eng_test.csv"))
    chn_test = pd.read_csv(os.path.join(processed_dir, "chn_test.csv"))
    
    # 合并训练集
    merged_train = pd.concat([eng_train, chn_train], ignore_index=True)
    
    # 合并验证集
    merged_val = pd.concat([eng_val, chn_val], ignore_index=True)
    
    # 合并测试集
    merged_test = pd.concat([eng_test, chn_test], ignore_index=True)
    
    # 确保文本列不含NaN值
    merged_train['text'] = merged_train['text'].fillna("")
    merged_val['text'] = merged_val['text'].fillna("")
    merged_test['text'] = merged_test['text'].fillna("")
    
    # 添加句子数量特征
    merged_train['sentence_count'] = merged_train['text'].apply(lambda x: len([s for s in str(x).split(r'[.!?。！？]') if s.strip()]))
    merged_val['sentence_count'] = merged_val['text'].apply(lambda x: len([s for s in str(x).split(r'[.!?。！？]') if s.strip()]))
    merged_test['sentence_count'] = merged_test['text'].apply(lambda x: len([s for s in str(x).split(r'[.!?。！？]') if s.strip()]))
    
    # 保存合并后的数据集
    merged_train.to_csv(os.path.join(clean_dir, "merged_train_clean.csv"), index=False)
    merged_val.to_csv(os.path.join(clean_dir, "merged_val_clean.csv"), index=False)
    merged_test.to_csv(os.path.join(clean_dir, "merged_test_clean.csv"), index=False)
    
    print(f"合并后的训练集：{len(merged_train)} 条样本")
    print(f"合并后的验证集：{len(merged_val)} 条样本")
    print(f"合并后的测试集：{len(merged_test)} 条样本")
    print("数据集合并完成！")

if __name__ == "__main__":
    main() 