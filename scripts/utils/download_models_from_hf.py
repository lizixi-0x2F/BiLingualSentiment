#!/usr/bin/env python
"""
从Hugging Face下载中英文情感分析模型的脚本

该脚本将从Hugging Face下载DistilBERT和XLM-RoBERTa情感分析模型
及其相关文件到本地目录。

用法:
python download_models_from_hf.py --output_dir downloaded_models
"""

import os
import argparse
from huggingface_hub import hf_hub_download

def download_models(repo_id="YourUsername/bilingual-sentiment-va", save_dir="downloaded_models"):
    """从Hugging Face Hub下载DistilBERT和XLM-RoBERTa模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义要下载的文件
    files_to_download = [
        "distilbert/best_model.pth",
        "xlm-roberta/best_model.pth",
        "models/roberta_model.py",
        "config.py",
        "inference.py"
    ]
    
    # 下载每个文件
    for file_path in files_to_download:
        print(f"下载 {file_path}...")
        local_dir = os.path.join(save_dir, os.path.dirname(file_path))
        os.makedirs(local_dir, exist_ok=True)
        
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
    
    print(f"所有模型文件已下载到 {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='从Hugging Face下载情感分析模型')
    parser.add_argument('--output_dir', type=str, default="downloaded_models",
                        help='下载文件的输出目录')
    parser.add_argument('--repo_id', type=str, default="YourUsername/bilingual-sentiment-va",
                        help='Hugging Face仓库ID')
    
    args = parser.parse_args()
    download_models(repo_id=args.repo_id, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
