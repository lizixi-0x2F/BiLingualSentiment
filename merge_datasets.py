import pandas as pd
import os

# 读取中英文训练集和验证集
chn_train = pd.read_csv('data/processed/chn_train.csv')
eng_train = pd.read_csv('data/processed/eng_train.csv')
chn_val = pd.read_csv('data/processed/chn_val.csv')
eng_val = pd.read_csv('data/processed/eng_val.csv')

# 合并训练集和验证集
merged_train = pd.concat([chn_train, eng_train], ignore_index=True)
merged_val = pd.concat([chn_val, eng_val], ignore_index=True)

# 确保sentence_count列存在
if 'sentence_count' not in merged_train.columns:
    merged_train['sentence_count'] = merged_train['text'].apply(
        lambda x: len([c for c in str(x) if c in ['.', '!', '?', '。', '！', '？']])
    )
    merged_val['sentence_count'] = merged_val['text'].apply(
        lambda x: len([c for c in str(x) if c in ['.', '!', '?', '。', '！', '？']])
    )

# 打乱数据
merged_train = merged_train.sample(frac=1, random_state=42).reset_index(drop=True)
merged_val = merged_val.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存合并后的数据集
os.makedirs('data/processed', exist_ok=True)
merged_train.to_csv('data/processed/merged_train.csv', index=False)
merged_val.to_csv('data/processed/merged_val.csv', index=False)

print(f'合并后的训练集大小: {len(merged_train)}')
print(f'合并后的验证集大小: {len(merged_val)}') 