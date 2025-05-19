#!/usr/bin/env python
"""
简单测试中英文情感分析模型的脚本

这个脚本可以快速测试从Hugging Face下载的情感分析模型。

用法:
python test_downloaded_models.py --model_dir downloaded_models --model_type distilbert --text "我今天很开心！"
"""

import os
import sys
import argparse
import torch
import numpy as np
import importlib.util

def import_from_file(module_name, file_path):
    """从文件导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_model(model_dir, model_type, text, device="cpu"):
    """测试情感分析模型"""
    # 导入必要模块
    config_module = import_from_file("config", os.path.join(model_dir, "config.py"))
    Config = config_module.Config
    
    models_module = import_from_file("roberta_model", os.path.join(model_dir, "models/roberta_model.py"))
    MultilingualDistilBERTModel = models_module.MultilingualDistilBERTModel
    XLMRobertaDistilledModel = models_module.XLMRobertaDistilledModel
    
    # 初始化配置
    config = Config()
    config.DEVICE = device
    
    # 选择模型
    if model_type.lower() == "distilbert":
        config.MULTILINGUAL_MODEL_NAME = "distilbert-base-multilingual-cased"
        model_class = MultilingualDistilBERTModel
        model_path = os.path.join(model_dir, "distilbert/best_model.pth")
    elif model_type.lower() == "xlm-roberta":
        config.MULTILINGUAL_MODEL_NAME = "xlm-roberta-base"
        model_class = XLMRobertaDistilledModel
        model_path = os.path.join(model_dir, "xlm-roberta/best_model.pth")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 初始化模型
    print(f"加载模型: {model_type}...")
    model = model_class(config)
    
    # 检查设备
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 获取tokenizer
    tokenizer = model.get_tokenizer()
    
    # 准备输入
    print(f"分析文本: \"{text}\"")
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # 运行推理
    with torch.no_grad():
        outputs = model(**encoding)
    
    # 获取结果
    valence = outputs[0, 0].item()
    arousal = outputs[0, 1].item()
    
    print("\n情感分析结果:")
    print(f"效价(Valence): {valence:.4f} [-1=消极, 1=积极]")
    print(f"唤醒度(Arousal): {arousal:.4f} [-1=平静, 1=激动]")
    
    # 分析情感象限
    quadrant = ""
    if valence >= 0 and arousal >= 0:
        quadrant = "快乐/兴奋"
    elif valence < 0 and arousal >= 0:
        quadrant = "愤怒/焦虑"
    elif valence < 0 and arousal < 0:
        quadrant = "悲伤/抑郁"
    else:  # valence >= 0 and arousal < 0
        quadrant = "满足/平静"
    
    print(f"情感象限: {quadrant}")
    
    return valence, arousal, quadrant

def main():
    parser = argparse.ArgumentParser(description='测试情感分析模型')
    parser.add_argument('--model_dir', type=str, default="downloaded_models",
                        help='模型目录')
    parser.add_argument('--model_type', type=str, default="distilbert",
                        choices=["distilbert", "xlm-roberta"],
                        help='模型类型 (distilbert 或 xlm-roberta)')
    parser.add_argument('--text', type=str, required=True,
                        help='要分析的文本')
    parser.add_argument('--device', type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help='使用的设备 (cpu 或 cuda)')
    
    args = parser.parse_args()
    test_model(args.model_dir, args.model_type, args.text, args.device)

if __name__ == "__main__":
    main()
