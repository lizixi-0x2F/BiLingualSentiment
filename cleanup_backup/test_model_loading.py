"""
测试本地模型加载功能的脚本
"""
import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel

def test_models():
    """测试从本地加载模型"""
    print("开始测试模型加载...")
    
    # 设置本地模型路径
    pretrained_models_dir = Path("pretrained_models")
    xlm_path = pretrained_models_dir / "xlm-roberta-base"
    distil_path = pretrained_models_dir / "distilbert-multilingual"
    
    # 检查模型路径是否存在
    if not xlm_path.exists():
        print(f"错误: XLM-RoBERTa 模型路径不存在: {xlm_path}")
    else:
        print(f"找到 XLM-RoBERTa 模型路径: {xlm_path}")
    
    if not distil_path.exists():
        print(f"错误: DistilBERT 模型路径不存在: {distil_path}")
    else:
        print(f"找到 DistilBERT 模型路径: {distil_path}")
        
    # 创建配置对象
    config = Config()
    config.OUTPUT_DIM = 2  # Valence 和 Arousal
    config.DROPOUT = 0.2
    
    # 测试 DistilBERT 模型加载
    try:
        print("\n测试 DistilBERT 模型加载...")
        config.MULTILINGUAL_MODEL_NAME = 'distilbert-base-multilingual-cased'
        model = MultilingualDistilBERTModel(config)
        
        # 查看对象方法
        print(f"模型类型: {type(model)}")
        print(f"可用方法: {[m for m in dir(model) if not m.startswith('_')]}")
        
        # 尝试使用本地模型
        if distil_path.exists():
            model.load_from_pretrained(str(distil_path))
            print("成功从本地路径加载 DistilBERT 模型!")
            
            # 简单测试前向传播
            dummy_input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
            dummy_attention_mask = torch.ones_like(dummy_input_ids)
            
            with torch.no_grad():
                outputs = model(dummy_input_ids, dummy_attention_mask)
                print(f"模型输出形状: {outputs.shape}")
                
    except Exception as e:
        print(f"DistilBERT 模型加载失败: {e}")
        import traceback
        print(traceback.format_exc())
    
    # 测试 XLM-RoBERTa 模型加载
    try:
        print("\n测试 XLM-RoBERTa 模型加载...")
        config.MULTILINGUAL_MODEL_NAME = 'xlm-roberta-base'
        model = XLMRobertaDistilledModel(config)
        
        # 查看对象方法
        print(f"模型类型: {type(model)}")
        print(f"可用方法: {[m for m in dir(model) if not m.startswith('_')]}")
        
        # 尝试使用本地模型
        if xlm_path.exists():
            model.load_from_pretrained(str(xlm_path))
            print("成功从本地路径加载 XLM-RoBERTa 模型!")
            
            # 简单测试前向传播
            dummy_input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
            dummy_attention_mask = torch.ones_like(dummy_input_ids)
            
            with torch.no_grad():
                outputs = model(dummy_input_ids, dummy_attention_mask)
                print(f"模型输出形状: {outputs.shape}")
                
    except Exception as e:
        print(f"XLM-RoBERTa 模型加载失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_models()
