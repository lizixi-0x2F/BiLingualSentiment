#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的转换脚本，用于在macOS环境中将PyTorch模型转换为Core ML格式
使用方法: python mac_convert.py
"""

import os
import torch
import coremltools as ct
import argparse
import numpy as np
from src.models.student_model import StudentModel

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为Core ML格式 (macOS专用)")
    parser.add_argument("--model_path", type=str, default="checkpoints/student_large_s/best_model.pt", help="PyTorch模型路径")
    parser.add_argument("--output_path", type=str, default="models/student_large_s.mlmodel", help="输出Core ML模型路径")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"加载模型: {args.model_path}")
    # 加载模型
    state = torch.load(args.model_path, map_location="cpu")
    config = state["config"]
    model_config = config["model"]
    
    # 创建模型实例
    model = StudentModel(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        intermediate_size=model_config["intermediate_size"],
        hidden_dropout_prob=model_config["hidden_dropout_prob"],
        attention_probs_dropout_prob=model_config["attention_probs_dropout_prob"],
        ltc_hidden_size=model_config["ltc_hidden_size"],
        ltc_num_layers=model_config["ltc_num_layers"],
        ltc_dropout=model_config["ltc_dropout"],
        output_dim=model_config["output_dim"]
    )
    model.load_state_dict(state["model"])
    model.eval()
    
    print(f"模型配置: hidden_size={model_config['hidden_size']}, layers={model_config['num_hidden_layers']}")
    
    # 创建模型包装类
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)
    
    wrapped_model = ModelWrapper(model)
    
    # 准备样本输入
    batch_size = 1
    max_seq_length = args.max_seq_length
    input_ids = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    
    print("跟踪模型...")
    # 跟踪模型
    traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
    
    print("转换为Core ML...")
    # 转换为Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(batch_size, max_seq_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(batch_size, max_seq_length), dtype=np.int32)
        ],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )
    
    # 确定模型类型
    model_type = "unknown"
    if model_config["hidden_size"] <= 384:
        model_type = "student_micro"
    elif model_config["hidden_size"] <= 512:
        model_type = "student_small"
    elif model_config["hidden_size"] <= 640:
        model_type = "student_medium"
    elif model_config["hidden_size"] <= 768:
        model_type = "student_large_s"
    
    # 添加元数据
    mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.model"] = model_type
    mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.version"] = "1.0"
    mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.hidden_size"] = str(model_config["hidden_size"])
    mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.num_layers"] = str(model_config["num_hidden_layers"])
    
    # 保存模型
    print(f"保存模型到: {args.output_path}")
    mlmodel.save(args.output_path)
    
    print("转换完成!")
    print(f"模型大小: {os.path.getsize(args.output_path) / (1024*1024):.2f} MB")
    
    # 验证模型
    print("验证模型...")
    loaded_model = ct.models.MLModel(args.output_path)
    print(f"输入特征: {loaded_model.input_description}")
    print(f"输出特征: {loaded_model.output_description}")
    print(f"元数据: {loaded_model.user_defined_metadata}")
    
    print("\n转换成功! 现在可以在iOS应用中使用此模型。")

if __name__ == "__main__":
    main() 