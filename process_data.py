#!/usr/bin/env python3
import pandas as pd

def process_emobank_normalized():
    """
    处理emobank_va_normalized.csv文件
    """
    # 输入和输出文件
    input_file = 'emobank_va_normalized.csv'
    output_file = 'emobank_va_normalized_clean.csv'
    
    print(f"读取文件: {input_file}")
    # 读取原始CSV文件
    df = pd.read_csv(input_file)
    
    # 显示原始列
    print(f"原始列: {df.columns.tolist()}")
    
    # 检查所需列是否存在
    columns = df.columns.tolist()
    if 'valence_norm' in columns and 'arousal_norm' in columns:
        print("文件已包含valence_norm和arousal_norm列")
        # 重命名列名以符合训练脚本的预期
        if 'valence' not in columns or 'arousal' not in columns:
            df = df.rename(columns={'valence_norm': 'valence', 'arousal_norm': 'arousal'})
            print("已将valence_norm和arousal_norm列重命名为valence和arousal")
    else:
        print("文件中没有valence_norm和arousal_norm列，将保持原样")
    
    # 显示数据形状
    print(f"数据形状: {df.shape}")
    
    # 显示前5行
    print("\n数据前5行:")
    print(df.head())
    
    # 保存处理后的文件
    df.to_csv(output_file, index=False)
    print(f"\n已保存处理后的文件: {output_file}")
    
    # 覆盖原始文件
    df.to_csv(input_file, index=False)
    print(f"已覆盖原始文件: {input_file}")

if __name__ == "__main__":
    process_emobank_normalized() 