#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def normalize_va(input_path: str, output_path: str):
    """
    读取 CSV，将 valence 和 arousal 从 [0,1] 映射到 (-1,1)，
    并写入新文件。
    """
    # 1) 读取原始数据
    df = pd.read_csv(input_path)

    # 2) 检查必需列是否存在
    for col in ('valence', 'arousal'):
        if col not in df.columns:
            raise KeyError(f"输入文件中缺少列: {col}")

    # 3) 归一化：x_norm = x * 2 - 1
    df['valence_norm'] = df['valence'] * 2 - 1
    df['arousal_norm'] = df['arousal'] * 2 - 1

    # 4) 保存到新文件（默认使用 UTF-8 BOM，以兼容 Excel 打开）
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 归一化完成，已保存到：{output_path}")

if __name__ == "__main__":
    # 如果脚本在本文件夹运行，确保 emobank_va.csv 已放在同目录
    input_csv = "emobank_va.csv"
    output_csv = "emobank_va_normalized.csv"

    # 可根据需要修改路径：
    if not os.path.isfile(input_csv):
        print(f"❌ 找不到输入文件: {input_csv}")
    else:
        normalize_va(input_csv, output_csv)
