#!/usr/bin/env python
"""
交互式情感分析测试脚本

这个脚本提供了一个交互式界面，让用户可以:
1. 选择使用DistilBERT或XLM-RoBERTa模型
2. 输入自定义文本（中文、英文或双语）
3. 查看情感分析结果和可视化输出
4. 连续测试多个输入

用法:
python interactive_test.py
"""

import os
import sys
import torch
import logging
import webbrowser
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
from src.utils.visualization import get_html_visualization

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路径
MODEL_PATHS = {
    "distilbert": "outputs/pretrained_distilbert_local/best_model.pth",
    "xlm-roberta": "outputs/pretrained_xlm_roberta_local/best_model.pth"
}

def load_model(model_type, model_path, device="cpu"):
    """
    加载模型
    
    Args:
        model_type: 模型类型 (distilbert 或 xlm-roberta)
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        model: 加载的模型
        tokenizer: 对应的tokenizer
    """
    config = Config()
    config.DEVICE = device
    
    if model_type == "distilbert":
        config.MULTILINGUAL_MODEL_NAME = "distilbert-base-multilingual-cased"
        model_class = MultilingualDistilBERTModel
    elif model_type == "xlm-roberta":
        config.MULTILINGUAL_MODEL_NAME = "xlm-roberta-base"
        model_class = XLMRobertaDistilledModel
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 初始化模型
    model = model_class(config)
    tokenizer = model.get_tokenizer()
    
    # 检查设备
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    elif torch.backends.mps.is_available() and device == "mps":
        device = "mps"
    else:
        device = "cpu"
    
    # 加载模型权重
    logger.info(f"从 {model_path} 加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"模型加载成功，训练轮次: {checkpoint.get('epoch', 'unknown')}")
    if "val_metrics" in checkpoint:
        metrics = checkpoint["val_metrics"]
        logger.info(f"验证集指标: RMSE={metrics.get('rmse', 'N/A'):.4f}, R2={metrics.get('r2', 'N/A'):.4f}")
    
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """
    预测文本的情感
    
    Args:
        text: 输入文本
        model: 模型
        tokenizer: tokenizer
        device: 计算设备
    
    Returns:
        valence: 效价值 [-1, 1]
        arousal: 唤醒度值 [-1, 1]
    """
    # 编码文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(**encoding)
    
    # 获取结果
    valence = outputs[0, 0].item()
    arousal = outputs[0, 1].item()
    
    return valence, arousal

def save_visualization(valence, arousal, text, output_dir="temp_viz"):
    """
    保存情感可视化HTML
    
    Args:
        valence: 效价值
        arousal: 唤醒度值
        text: 文本
        output_dir: 输出目录
    
    Returns:
        html_path: HTML文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成HTML内容
    html_content = get_html_visualization(
        predictions=np.array([valence, arousal]),
        text=text,
        remove_special_tokens=True
    )
    
    # 保存HTML
    html_path = os.path.join(output_dir, "sentiment_visualization.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path

def print_sentiment_analysis(valence, arousal):
    """打印情感分析结果"""
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

def interactive_test():
    """交互式测试主函数"""
    print("\n" + "="*50)
    print("欢迎使用交互式情感分析测试")
    print("="*50)
    
    # 选择设备
    device = "cpu"
    if torch.cuda.is_available():
        device_input = input("\n检测到CUDA，是否使用GPU？(y/n，默认y): ").strip().lower()
        if device_input != "n":
            device = "cuda"
            print("使用GPU进行推理")
        else:
            print("使用CPU进行推理")
    else:
        print("未检测到CUDA，使用CPU进行推理")
    
    # 选择模型
    print("\n可用的模型:")
    print("1. DistilBERT多语言模型 (更快，较小)")
    print("2. XLM-RoBERTa模型 (更准确，较大)")
    
    model_choice = ""
    while model_choice not in ["1", "2"]:
        model_choice = input("\n请选择模型 [1/2，默认1]: ").strip()
        if model_choice == "":
            model_choice = "1"
    
    model_type = "distilbert" if model_choice == "1" else "xlm-roberta"
    model_path = MODEL_PATHS[model_type]
    
    # 加载模型
    try:
        model, tokenizer, device = load_model(model_type, model_path, device)
        print(f"\n成功加载 {model_type} 模型！")
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        return
    
    # 创建临时目录
    temp_dir = os.path.join(os.getcwd(), "temp_visualizations")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 主交互循环
    while True:
        print("\n" + "-"*50)
        print("输入文本进行情感分析 (输入'q'退出):")
        text = input("> ")
        
        if text.lower() in ["q", "quit", "exit"]:
            break
        
        if not text.strip():
            print("请输入有效文本")
            continue
        
        try:
            # 执行预测
            valence, arousal = predict_sentiment(text, model, tokenizer, device)
            
            # 显示结果
            print_sentiment_analysis(valence, arousal)
            
            # 保存并显示可视化
            html_path = save_visualization(valence, arousal, text, temp_dir)
            print(f"\n生成的可视化报告位于: {html_path}")
            
            # 打开浏览器显示结果
            show_viz = input("是否打开可视化报告？ (y/n，默认y): ").strip().lower()
            if show_viz != "n":
                webbrowser.open('file://' + os.path.abspath(html_path))
                
        except Exception as e:
            print(f"分析过程中出错: {e}")
    
    print("\n感谢使用交互式情感分析测试！")
    
if __name__ == "__main__":
    interactive_test()
