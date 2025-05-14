import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, concatenate_datasets
import json

# 设置设备 - 添加MPS支持（Apple Silicon芯片加速）
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"使用设备: {device}")

# 加载英文数据
def load_english_data(use_normalized=True, max_samples=10000):
    """
    加载英文数据
    
    参数:
        use_normalized: 是否使用归一化的英文数据
        max_samples: 最大样本数量
    """
    if use_normalized and os.path.exists('emobank_va_normalized.csv'):
        print("加载归一化英文数据(emobank_va_normalized.csv)...")
        df = pd.read_csv('emobank_va_normalized.csv')
        # 数据清洗 - 确保text列是字符串类型
        df['text'] = df['text'].astype(str)
        
        # 检查文件格式
        columns = df.columns.tolist()
        print(f"英文数据列: {columns}")
        
        # 检查英文数据的值范围
        val_min, val_max = df['valence'].min(), df['valence'].max()
        aro_min, aro_max = df['arousal'].min(), df['arousal'].max()
        print(f"英文数据值范围: valence ({val_min:.3f} 到 {val_max:.3f})")
        print(f"英文数据值范围: arousal ({aro_min:.3f} 到 {aro_max:.3f})")
        
        # 确保数据值在[-1, 1]范围内
        if val_min < -1 or val_max > 1 or aro_min < -1 or aro_max > 1:
            print("标准化英文数据到[-1, 1]范围...")
            # 标准化valence
            val_range = val_max - val_min
            df['valence'] = 2 * (df['valence'] - val_min) / val_range - 1
            
            # 标准化arousal
            aro_range = aro_max - aro_min
            df['arousal'] = 2 * (df['arousal'] - aro_min) / aro_range - 1
            
            print(f"标准化后: valence ({df['valence'].min():.3f} 到 {df['valence'].max():.3f})")
            print(f"标准化后: arousal ({df['arousal'].min():.3f} 到 {df['arousal'].max():.3f})")
        
        # 限制样本数量以减少内存使用和加快训练
        if len(df) > max_samples:
            print(f"限制英文样本数量为 {max_samples}（原始数量: {len(df)}）...")
            df = df.sample(n=max_samples, random_state=42)
        
        # 添加语言标识
        df['language'] = 'english'
        print(f"英文数据集大小: {len(df)} 条记录")
        return df
    else:
        print("加载原始英文数据(emobank_va.csv)...")
        df = pd.read_csv('emobank_va.csv')
        # 数据清洗 - 确保text列是字符串类型
        df['text'] = df['text'].astype(str)
        
        # 限制样本数量以减少内存使用和加快训练
        if len(df) > max_samples:
            print(f"限制英文样本数量为 {max_samples}（原始数量: {len(df)}）...")
            df = df.sample(n=max_samples, random_state=42)
        
        # 添加语言标识
        df['language'] = 'english'
        print(f"英文数据集大小: {len(df)} 条记录")
        return df

# 加载中文诗词数据
def load_chinese_data(max_samples=10000):
    print("加载中文诗词数据...")
    # 从text_valence_arousal_poetry_noisy.csv加载数据
    if os.path.exists('text_valence_arousal_poetry_noisy.csv'):
        print("从text_valence_arousal_poetry_noisy.csv加载中文诗词数据...")
        df = pd.read_csv('text_valence_arousal_poetry_noisy.csv')
        # 数据清洗 - 确保text列是字符串类型
        df['text'] = df['text'].astype(str)
        
        # 限制样本数量以减少内存使用和加快训练
        if len(df) > max_samples:
            print(f"限制中文样本数量为 {max_samples}（原始数量: {len(df)}）...")
            df = df.sample(n=max_samples, random_state=42)
        
        # 检查中文数据的值范围
        val_min, val_max = df['valence'].min(), df['valence'].max()
        aro_min, aro_max = df['arousal'].min(), df['arousal'].max()
        print(f"中文数据原始范围: valence ({val_min:.3f} 到 {val_max:.3f})")
        print(f"中文数据原始范围: arousal ({aro_min:.3f} 到 {aro_max:.3f})")
        
        # 将值标准化到[-1, 1]范围，与英文数据保持一致
        if val_min < -1 or val_max > 1 or aro_min < -1 or aro_max > 1:
            print("对中文数据进行标准化，映射到[-1, 1]范围...")
            
            # valence标准化
            val_range = val_max - val_min
            df['valence'] = 2 * (df['valence'] - val_min) / val_range - 1
            
            # arousal标准化
            aro_range = aro_max - aro_min
            df['arousal'] = 2 * (df['arousal'] - aro_min) / aro_range - 1
            
            print(f"标准化后的valence范围: {df['valence'].min():.3f} 到 {df['valence'].max():.3f}")
            print(f"标准化后的arousal范围: {df['arousal'].min():.3f} 到 {df['arousal'].max():.3f}")
        
        # 添加语言标识
        df['language'] = 'chinese'
        print(f"中文数据集大小: {len(df)} 条记录")
        return df
    else:
        print("错误: 中文数据集文件 text_valence_arousal_poetry_noisy.csv 不存在")
        print("创建简单的中文诗词数据...")
        # 如果文件不存在，创建一个小的样本数据集
        poetry_data = {
            'text': [
                "春眠不觉晓，处处闻啼鸟。",
                "床前明月光，疑是地上霜。",
                "两个黄鹂鸣翠柳，一行白鹭上青天。",
                "欲穷千里目，更上一层楼。",
                "飞流直下三千尺，疑是银河落九天。",
                "孤帆远影碧空尽，唯见长江天际流。",
                "泉眼无声惜细流，树阴照水爱晴柔。",
                "千山鸟飞绝，万径人踪灭。",
                "寒雨连江夜入吴，平明送客楚山孤。",
                "人闲桂花落，夜静春山空。",
                "会当凌绝顶，一览众山小。",
                "枯藤老树昏鸦，小桥流水人家。",
                "小时不识月，呼作白玉盘。",
                "停车坐爱枫林晚，霜叶红于二月花。",
                "采菊东篱下，悠然见南山。"
            ],
            'valence': [-0.5, -0.3, 0.6, 0.5, 0.4, 0.2, 0.4, -0.1, 0.0, 0.3, 0.6, 0.2, 0.4, 0.5, 0.6],
            'arousal': [-0.2, -0.3, 0.0, 0.2, 0.3, -0.2, -0.4, 0.1, -0.1, -0.4, 0.2, -0.2, -0.4, 0.0, -0.1]
        }
        
        poetry_df = pd.DataFrame(poetry_data)
        # 添加语言标识
        poetry_df['language'] = 'chinese' 
        print(f"创建的中文样本数据集大小: {len(poetry_df)} 条记录")
        return poetry_df

# 将数据转换为模型可用的格式
def prepare_data(dataframe, tokenizer):
    # 确保文本列是字符串类型
    dataframe['text'] = dataframe['text'].astype(str)
    
    # 创建Dataset对象
    dataset = Dataset.from_pandas(dataframe)
    
    # 将valence和arousal作为标签
    dataset = dataset.map(lambda x: {
        'labels': [float(x['valence']), float(x['arousal'])]
    })
    
    # 添加分词处理
    def tokenize_function(examples):
        texts = [str(text) for text in examples["text"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    
    # 分词处理
    dataset = dataset.map(tokenize_function, batched=True, batch_size=64)
    
    return dataset

# 训练模型
def train_model():
    # 使用MacBERT作为同时支持中英文的模型
    model_name = "hfl/chinese-macbert-base"
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载数据 - 使用更大的数据集
    max_samples = 10000  # 增加到10000（之前是5000）
    english_df = load_english_data(max_samples=max_samples)
    chinese_df = load_chinese_data(max_samples=max_samples)
    
    # 合并数据集
    print("合并中英文数据...")
    combined_df = pd.concat([english_df, chinese_df], ignore_index=True)
    print(f"合并后数据集大小: {len(combined_df)} 条记录")
    print(f"- 英文: {len(english_df)} 条")
    print(f"- 中文: {len(chinese_df)} 条")
    
    # 划分训练集和验证集
    train_df, eval_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['language'])
    
    # 准备数据集
    print("处理训练数据...")
    train_dataset = prepare_data(train_df, tokenizer)
    print("处理评估数据...")
    eval_dataset = prepare_data(eval_df, tokenizer)
    
    # 加载MacBERT模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="regression",
        ignore_mismatched_sizes=True
    ).to(device)
    
    # 训练参数 - 使用指定设备
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,  # 增加学习率从2e-5到3e-5
        per_device_train_batch_size=8,  # 保持批次大小为8
        per_device_eval_batch_size=8,  # 保持评估批次大小为8
        num_train_epochs=5,  # 增加训练轮次从3改为5
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        # 添加梯度累积以减少内存使用
        gradient_accumulation_steps=2,
        # 添加fp16训练以提高速度（如果设备支持）
        fp16=torch.cuda.is_available(),  # 仅在CUDA可用时使用fp16
        # 添加学习率预热和线性调度
        warmup_ratio=0.1,  # 添加10%的预热步骤
        lr_scheduler_type="linear",  # 使用线性学习率调度
        # 添加更好的权重初始化
        dataloader_num_workers=4 if not torch.backends.mps.is_available() else 0,  # MPS不支持多进程数据加载
        # 添加梯度裁剪以提高稳定性
        max_grad_norm=1.0,
        # 定期保存checkpoint
        save_total_limit=3,  # 只保留最好的3个checkpoint
        # 添加早停回调而不是参数
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 开始训练
    print("开始微调MacBERT模型（中英文同时支持）...")
    trainer.train()
    
    # 创建模型目录
    os.makedirs("model_dir", exist_ok=True)
    
    # 保存微调后的模型到本地目录
    model.save_pretrained("model_dir")
    tokenizer.save_pretrained("model_dir")
    
    print("模型微调完成，已保存到model_dir目录")
    
    return model, tokenizer

# 主函数
if __name__ == "__main__":
    # 训练单一模型支持中英文
    model, tokenizer = train_model()
    
    print("训练完成!")
    print("- 中英文模型保存在: model_dir") 