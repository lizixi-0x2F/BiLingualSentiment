#!/usr/bin/env python
"""
验证Hugging Face模型工作流的脚本

这个脚本可以帮助用户验证从Hugging Face下载的模型是否正常工作。
它会下载模型并测试不同语言的文本示例。

用法:
python verify_workflow.py --repo_id YourUsername/bilingual-sentiment-va
"""

import os
import argparse
import subprocess
import sys
import time

def run_command(cmd):
    """运行命令并显示输出"""
    print(f"\n执行命令: {cmd}\n")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
        sys.stdout.flush()
    
    process.wait()
    return process.returncode

def verify_workflow(repo_id, output_dir="downloaded_models"):
    """验证整个工作流"""
    print("="*80)
    print("开始验证Hugging Face模型工作流")
    print("="*80)

    # 1. 检查依赖项
    print("\n[1/4] 检查依赖项...")
    try:
        import torch
        import huggingface_hub
        import transformers
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Hugging Face Hub版本: {huggingface_hub.__version__}")
        print(f"Transformers版本: {transformers.__version__}")
        print("✅ 依赖项已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖项: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

    # 2. 下载模型
    print("\n[2/4] 从Hugging Face下载模型...")
    cmd = f"python download_models_from_hf.py --repo_id {repo_id} --output_dir {output_dir}"
    if run_command(cmd) != 0:
        print("❌ 下载模型失败")
        return False
    print("✅ 模型下载成功")

    # 3. 测试DistilBERT模型
    print("\n[3/4] 测试DistilBERT模型...")
    test_texts = [
        "今天天气真好，我很开心！",  # 中文积极
        "这个产品质量很差，我非常失望。",  # 中文消极
        "I'm feeling great today!",  # 英文积极
        "This is the worst experience ever."  # 英文消极
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n测试 {i+1}/{len(test_texts)}: {text}")
        cmd = f'python test_downloaded_models.py --model_dir {output_dir} --model_type distilbert --text "{text}"'
        if run_command(cmd) != 0:
            print(f"❌ DistilBERT模型测试失败: {text}")
            return False
        # 短暂暂停，避免输出混淆
        time.sleep(1)
    
    # 4. 测试XLM-RoBERTa模型
    print("\n[4/4] 测试XLM-RoBERTa模型...")
    for i, text in enumerate(test_texts):
        print(f"\n测试 {i+1}/{len(test_texts)}: {text}")
        cmd = f'python test_downloaded_models.py --model_dir {output_dir} --model_type xlm-roberta --text "{text}"'
        if run_command(cmd) != 0:
            print(f"❌ XLM-RoBERTa模型测试失败: {text}")
            return False
        # 短暂暂停，避免输出混淆
        time.sleep(1)

    # 验证完成
    print("\n"+"="*80)
    print("✅ 验证成功！Hugging Face模型工作流正常工作")
    print("="*80)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='验证Hugging Face模型工作流')
    parser.add_argument('--repo_id', type=str, default="YourUsername/bilingual-sentiment-va",
                        help='Hugging Face仓库ID')
    parser.add_argument('--output_dir', type=str, default="downloaded_models",
                        help='下载文件的输出目录')
    
    args = parser.parse_args()
    verify_workflow(args.repo_id, args.output_dir)

if __name__ == "__main__":
    main()
