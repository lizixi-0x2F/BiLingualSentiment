#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
词性特征提取模块
用于增强情感Valence判断能力
"""

import jieba
import jieba.posseg as pseg
import numpy as np
from collections import Counter
import logging

# 初始化jieba
jieba.setLogLevel(logging.INFO)

# 词性映射到情感倾向的权重
# 形容词、副词、动词和名词在情感表达中往往更重要
POS_SENTIMENT_WEIGHTS = {
    'a': 1.5,    # 形容词：高权重
    'ad': 1.3,   # 副形词
    'an': 1.2,   # 名形词
    'd': 1.3,    # 副词：高权重
    'v': 1.2,    # 动词：中高权重
    'vd': 1.3,   # 动副词 
    'vn': 1.1,   # 名动词
    'n': 0.8,    # 名词：中等权重
    'nr': 0.5,   # 人名
    'ns': 0.5,   # 地名
    'nt': 0.7,   # 机构团体
    'nz': 0.8,   # 其他专名
    'i': 1.4,    # 习用语（包括成语）：高权重，常带有强烈情感
    'l': 1.0,    # 习用语
    'c': 0.3,    # 连词
    'p': 0.3,    # 介词
    'r': 0.6,    # 代词
    'm': 0.5,    # 数词
    'q': 0.5,    # 量词
    'u': 0.2,    # 助词
    'xc': 0.3,   # 其他虚词
    'w': 0.1,    # 标点符号
    't': 0.6,    # 时间词
    'f': 0.5,    # 方位词
    'e': 1.2,    # 叹词：高权重，常带情感
    'y': 1.0,    # 语气词：中高权重，可能带情感
    'o': 0.5,    # 拟声词
    'h': 0.3,    # 前缀
    'k': 0.3,    # 后缀
    'x': 0.1,    # 字符串
    'z': 0.6,    # 状态词
    'zg': 0.8,   # 状态词
}

# 情感词性列表（对情感表达最重要的词性）
SENTIMENT_POS_TAGS = ['a', 'ad', 'an', 'd', 'v', 'vd', 'i', 'l', 'z', 'e', 'y']

def extract_pos_features(text):
    """
    提取词性特征并转换为向量
    
    参数:
        text: 输入文本
        
    返回:
        词性特征向量 (12维)
    """
    # 使用jieba进行词性标注
    words_pos = pseg.cut(text)
    
    # 统计各词性数量
    pos_counts = Counter(pos for _, pos in words_pos)
    
    # 文本长度（词数）
    total_words = sum(pos_counts.values())
    if total_words == 0:
        return np.zeros(12)  # 空文本返回零向量
    
    # 提取特征
    features = []
    
    # 1-3. 形容词、副词、动词的比例
    adj_ratio = sum(pos_counts[pos] for pos in ['a', 'ad', 'an']) / total_words
    adv_ratio = pos_counts['d'] / total_words if 'd' in pos_counts else 0
    verb_ratio = sum(pos_counts[pos] for pos in ['v', 'vd', 'vn']) / total_words
    features.extend([adj_ratio, adv_ratio, verb_ratio])
    
    # 4. 情感相关词性占比
    sentiment_pos_ratio = sum(pos_counts[pos] for pos in SENTIMENT_POS_TAGS if pos in pos_counts) / total_words
    features.append(sentiment_pos_ratio)
    
    # 5. 加权情感词性分数
    weighted_pos_score = sum(POS_SENTIMENT_WEIGHTS.get(pos, 0.5) * count 
                            for pos, count in pos_counts.items()) / total_words
    features.append(weighted_pos_score)
    
    # 6. 名词与动词比率（越高越客观，越低越主观）
    noun_verb_ratio = sum(pos_counts[pos] for pos in ['n', 'nr', 'ns', 'nt', 'nz']) / max(1, sum(pos_counts[pos] for pos in ['v', 'vd', 'vn']))
    features.append(min(noun_verb_ratio, 5.0))  # 截断过大值
    
    # 7. 情感词密度
    sentiment_word_density = sum(POS_SENTIMENT_WEIGHTS.get(pos, 0) * count 
                                for pos, count in pos_counts.items() 
                                if pos in SENTIMENT_POS_TAGS) / total_words
    features.append(sentiment_word_density)
    
    # 8. 标点符号比例（表达强度）
    punct_ratio = pos_counts.get('w', 0) / total_words
    features.append(punct_ratio)
    
    # 9. 感叹词和语气词比例
    exclamation_ratio = sum(pos_counts[pos] for pos in ['e', 'y']) / total_words
    features.append(exclamation_ratio)
    
    # 10. 形容词和副词比例之和（描述性强度）
    descriptive_ratio = adj_ratio + adv_ratio
    features.append(descriptive_ratio)
    
    # 11. 代词比例（个人化程度）
    pronoun_ratio = pos_counts.get('r', 0) / total_words
    features.append(pronoun_ratio)
    
    # 12. 虚词比例（语气和语调）
    function_word_ratio = sum(pos_counts[pos] for pos in ['c', 'p', 'u', 'xc']) / total_words
    features.append(function_word_ratio)
    
    return np.array(features, dtype=np.float32)

def get_valence_polarity_indicators(text):
    """
    提取词性表示的Valence极性指标
    
    参数:
        text: 输入文本
        
    返回:
        极性指标 (3维): [正极性分数, 负极性分数, 情感强度分数]
    """
    # 加载情感词典（简化版）- 实际应用中应该加载完整词典
    # 这里使用一些常见的情感词示例
    positive_words = set(['好', '喜欢', '开心', '高兴', '快乐', '愉快', '满意', '优秀', '棒', '赞', 
                         '美好', '感谢', '谢谢', '幸福', '欣赏', '厉害', '漂亮', '酷', '可爱'])
    negative_words = set(['坏', '讨厌', '生气', '难过', '失望', '悲伤', '不满', '糟糕', '差', '烂', 
                         '可怕', '恐怖', '恨', '焦虑', '害怕', '忧伤', '丑', '恶心', '可恶'])
    
    # 使用jieba分词
    words = jieba.lcut(text)
    words_pos = pseg.cut(text)
    
    # 计数器
    positive_count = 0
    negative_count = 0
    intensity_score = 0
    
    # 统计极性词数量和强度
    for word, pos in words_pos:
        weight = POS_SENTIMENT_WEIGHTS.get(pos, 0.5)
        
        if word in positive_words:
            positive_count += weight
        elif word in negative_words:
            negative_count += weight
            
        # 对形容词和副词等情感词给予更高强度
        if pos in SENTIMENT_POS_TAGS:
            intensity_score += weight
    
    total_words = len(words)
    if total_words == 0:
        return np.zeros(3, dtype=np.float32)
    
    # 归一化
    positive_score = positive_count / total_words
    negative_score = negative_count / total_words
    intensity_score = min(intensity_score / total_words, 1.0)
    
    return np.array([positive_score, negative_score, intensity_score], dtype=np.float32)

def extract_combined_pos_features(text):
    """
    组合词性特征和极性指标
    
    参数:
        text: 输入文本
        
    返回:
        组合特征向量 (15维)
    """
    pos_features = extract_pos_features(text)
    polarity_indicators = get_valence_polarity_indicators(text)
    
    return np.concatenate([pos_features, polarity_indicators])

def preprocess_batch_texts(texts, feature_type='combined'):
    """
    批量处理文本并提取词性特征
    
    参数:
        texts: 文本列表
        feature_type: 特征类型，'pos'，'polarity'或'combined'
        
    返回:
        特征矩阵
    """
    if feature_type == 'pos':
        return np.array([extract_pos_features(text) for text in texts])
    elif feature_type == 'polarity':
        return np.array([get_valence_polarity_indicators(text) for text in texts])
    else:  # combined
        return np.array([extract_combined_pos_features(text) for text in texts])

# 测试代码
if __name__ == "__main__":
    test_texts = [
        "我今天很开心，因为天气真的太好了！",
        "这部电影真是太糟糕了，浪费了我的时间和金钱。",
        "周末我们去了公园，孩子们玩得很开心。",
        "这个产品质量不错，价格也合理，我很满意。"
    ]
    
    for text in test_texts:
        print(f"\n文本: {text}")
        print("词性特征:", extract_pos_features(text))
        print("极性指标:", get_valence_polarity_indicators(text))
        print("组合特征:", extract_combined_pos_features(text)) 