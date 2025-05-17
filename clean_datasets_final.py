import pandas as pd
import os
import re

def clean_emotion_dataset(input_file, output_file):
    """清洗情感数据集，移除真正的标注不一致样本"""
    print(f"开始清洗数据集：{input_file}")
    
    # 读取数据集
    df = pd.read_csv(input_file)
    original_size = len(df)
    
    # 定义情感词汇表
    positive_keywords = [
        "开心", "高兴", "喜欢", "happy", "joy", "喜悦", "满意", "满足", 
        "愉快", "快乐", "舒适", "感激", "享受", "赞赏", "欣赏", "感谢"
    ]
    
    negative_keywords = [
        "难过", "伤心", "生气", "愤怒", "悲伤", "讨厌", "失望", "焦虑", 
        "sad", "angry", "fear", "担心", "害怕", "恐惧", "恨", "烦恼", 
        "痛苦", "忧虑", "沮丧", "郁闷", "抑郁", "哀伤"
    ]
    
    # 否定词列表
    negation_words = ["不", "没", "无", "别", "莫", "非", "未", "勿", "毫不", "绝不"]
    
    # 找出不一致的样本
    inconsistent_rows = []
    
    # 1. 找出真正的异常数据：简单积极词汇但V值为负的样本
    positive_single_words = ["开心", "高兴", "快乐", "喜欢", "满意", "满足", "happy", "joy"]
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        # 如果文本仅为简单积极词汇，且V值为负
        if text in positive_single_words and row['V'] < 0:
            inconsistent_rows.append(idx)
            print(f"删除 - 基本积极词汇但V值为负: {text} | V={row['V']:.2f}")
    
    # 2. 找出真正的异常数据：简单消极词汇但V值为正的样本
    negative_single_words = ["难过", "伤心", "生气", "愤怒", "悲伤", "讨厌", "焦虑", "sad", "angry"]
    for idx, row in df.iterrows():
        if idx in inconsistent_rows:
            continue
            
        text = str(row['text']).strip()
        # 如果文本仅为简单消极词汇，且V值为正
        if text in negative_single_words and row['V'] > 0:
            inconsistent_rows.append(idx)
            print(f"删除 - 基本消极词汇但V值为正: {text} | V={row['V']:.2f}")
    
    # 3. 处理极端异常值
    extreme_rows = df[(df['V'] < -0.95) | (df['V'] > 0.95) | (df['A'] < -0.95) | (df['A'] > 0.95)].index
    if len(extreme_rows) > 0:
        extreme_rows_not_in_inconsistent = [idx for idx in extreme_rows if idx not in inconsistent_rows]
        if extreme_rows_not_in_inconsistent:
            print(f"删除 {len(extreme_rows_not_in_inconsistent)} 个极端异常值样本")
            inconsistent_rows.extend(extreme_rows_not_in_inconsistent)
    
    # 4. 检查一些明显的矛盾文本，如包含"我很开心"但V值为负的样本
    # 注意：要区分"不开心"和"开心"的情况
    for idx, row in df.iterrows():
        if idx in inconsistent_rows:
            continue
            
        text = str(row['text']).lower()
        v_value = row['V']
        
        # 先检查是否包含否定词+积极词的组合
        has_negated_positive = any(neg + pos.lower() in text for neg in negation_words for pos in positive_keywords)
        
        # 再检查是否包含积极词但不是被否定的情况
        for pos_word in positive_keywords:
            if pos_word.lower() in text:
                # 确认这个积极词没有被否定词修饰
                is_negated = any(text.find(neg + pos_word.lower()) != -1 for neg in negation_words)
                
                # 如果积极词没有被否定，但V值为负，则可能是标注问题
                if not is_negated and not has_negated_positive and v_value < -0.3:
                    # 额外检查是否是特殊情况，如"想念过去快乐的时光"等
                    special_contexts = ["过去", "曾经", "以前", "再也", "不再", "失去"]
                    has_special_context = any(context in text for context in special_contexts)
                    
                    if not has_special_context:
                        inconsistent_rows.append(idx)
                        print(f"删除 - 未被否定的积极词但V值为负: {text[:40]}... | V={v_value:.2f}")
                        break
    
    # 5. 检查包含明确消极词但V值为正的样本
    for idx, row in df.iterrows():
        if idx in inconsistent_rows:
            continue
            
        text = str(row['text']).lower()
        v_value = row['V']
        
        # 先检查是否包含否定词+消极词的组合
        has_negated_negative = any(neg + neg_word.lower() in text for neg in negation_words for neg_word in negative_keywords)
        
        # 如果包含不被否定的消极词，但V值为正
        for neg_word in negative_keywords:
            if neg_word.lower() in text:
                # 确认这个消极词没有被否定词修饰
                is_negated = any(text.find(neg + neg_word.lower()) != -1 for neg in negation_words)
                
                # 检查是否包含特殊安慰词汇（不要担心，别难过等）
                comfort_words = ["不要", "别", "无需", "不必", "无须", "无需", "不需", "不用"]
                has_comfort = any(comfort + neg_word.lower() in text for comfort in comfort_words)
                
                # 如果消极词没有被否定，也不是安慰语境，但V值为正
                if not is_negated and not has_negated_negative and not has_comfort and v_value > 0.3:
                    # 额外排除一些特殊情况
                    exclude_words = ["解决", "克服", "战胜", "帮助", "安慰", "支持"]
                    has_exclude = any(ex_word in text for ex_word in exclude_words)
                    
                    if not has_exclude:
                        inconsistent_rows.append(idx)
                        print(f"删除 - 未被否定的消极词但V值为正: {text[:40]}... | V={v_value:.2f}")
                        break
    
    # 删除识别出的不一致样本
    inconsistent_rows = list(set(inconsistent_rows))  # 去重
    df_clean = df.drop(inconsistent_rows)
    
    # 保存清洗后的数据集
    df_clean.to_csv(output_file, index=False)
    
    # 输出统计信息
    removed_count = len(inconsistent_rows)
    print(f"已删除 {removed_count} 个异常样本 ({removed_count/original_size*100:.2f}%)")
    print(f"清洗后数据集大小: {len(df_clean)}")
    print(f"清洗后数据集已保存到: {output_file}")
    
    return df_clean

# 清洗训练集和验证集
if __name__ == "__main__":
    # 创建清洗后数据的目录
    output_dir = "data/processed_clean"
    os.makedirs(output_dir, exist_ok=True)
    
    # 清洗训练集
    train_clean = clean_emotion_dataset(
        "data/processed/merged_train.csv", 
        f"{output_dir}/merged_train_clean.csv"
    )
    
    # 清洗验证集
    val_clean = clean_emotion_dataset(
        "data/processed/merged_val.csv", 
        f"{output_dir}/merged_val_clean.csv"
    )
    
    print("\n数据清洗完成！")
    print(f"清洗前：训练集 {len(pd.read_csv('data/processed/merged_train.csv'))} 样本，验证集 {len(pd.read_csv('data/processed/merged_val.csv'))} 样本")
    print(f"清洗后：训练集 {len(train_clean)} 样本，验证集 {len(val_clean)} 样本") 