#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感价效度预测脚本
使用LTC-NCP-VA模型进行中英文文本的情感分析
"""

import os
import sys
import torch
import argparse
import yaml
import numpy as np
from load_safetensors_model_fixed import load_model_from_safetensors
import logging
from src.core import LTC_NCP_RNN

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 简易分词器实现
class SimpleTokenizer:
    """简单分词器，适用于英文和中文"""
    
    def __init__(self, vocab_file=None, vocab_size=10000):
        """初始化分词器"""
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.vocab_size_actual = 2  # 初始词汇表大小
        
        # 如果提供了词汇表文件，加载它
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        """从文件加载词汇表"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and word not in self.word2idx:
                        self.word2idx[word] = self.vocab_size_actual
                        self.idx2word[self.vocab_size_actual] = word
                        self.vocab_size_actual += 1
            logger.info(f"从文件加载词汇表，大小: {self.vocab_size_actual}")
        except Exception as e:
            logger.error(f"加载词汇表出错: {str(e)}")
    
    def encode(self, text, max_length=120, truncation=True, padding='max_length', return_tensors=None):
        """将文本编码为ID序列"""
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = str(text)
                
        # 将文本转换为token IDs
        token_ids = []
        for char in text:
            if char in self.word2idx:
                token_ids.append(self.word2idx[char])
            else:
                token_ids.append(self.unk_token_id)
        
        # 截断
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # 填充
        if padding == 'max_length':
            padded_tokens = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
            token_ids = padded_tokens[:max_length]
        
        # 转换为张量
        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            return token_ids

def generate_meta_features(text):
    """生成文本的元特征"""
    # 文本长度
    text_length = len(text)
    
    # 计算标点符号密度
    punct_chars = set("，。！？；：""''「」『』（）【】《》，。、：；！？·~!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
    punct_count = sum(1 for char in text if char in punct_chars)
    punct_density = punct_count / max(text_length, 1)
    
    # 统计句子数量 (通过句子终止符号)
    sentence_terminators = set("。！？.!?")
    sentence_count = sum(1 for char in text if char in sentence_terminators)
    sentence_count = max(1, sentence_count)  # 至少有一个句子
    
    # 返回元特征张量
    meta_features = torch.tensor([text_length, punct_density, sentence_count], dtype=torch.float32)
    return meta_features

def calibrate_emotion_value(value):
    """
    校准情感值，增强区分度
    使用S型函数增强对极值的敏感度
    """
    # 应用非线性变换，增强极值效果
    calibration_factor = 1.5  # 校准系数，调整曲线陡度
    if abs(value) < 0.3:
        # 小值区域轻微压缩
        return value * 0.8
    else:
        # 大值区域适度放大，保持符号
        sign = 1 if value > 0 else -1
        scaled = abs(value) * calibration_factor
        # 确保不超过1.0
        scaled = min(1.0, scaled)
        return sign * scaled

def predict_emotion(model, tokenizer, text, device='cpu', calibrate=True):
    """使用模型预测文本的情感价效度"""
    # 准备输入
    tokens = tokenizer.encode(
        text, 
        max_length=120,  # 使用与训练时相同的最大长度
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).unsqueeze(0).to(device)  # 添加批次维度
    
    # 准备长度信息
    text_length = min(len(text), 120)
    lengths = torch.tensor([text_length], dtype=torch.long).to(device)
    
    # 准备元特征
    meta_features = generate_meta_features(text).unsqueeze(0).to(device)  # 添加批次维度
    
    # 设置模型为评估模式
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        try:
            # 预测情感值
            predictions = model(tokens, lengths, meta_features)
            
            # 提取价效度值
            valence, arousal = predictions[0].tolist()
            
            # 应用额外校准，增强情感判断效果
            if calibrate:
                # 非线性校准，增强对极值的反应，压缩中间值
                valence = calibrate_emotion_value(valence)
                arousal = calibrate_emotion_value(arousal)
            
            # 确保值在[-1, 1]范围内
            valence = max(-1.0, min(1.0, valence))
            arousal = max(-1.0, min(1.0, arousal))
            
            return valence, arousal
        except Exception as e:
            logger.error(f"预测出错: {str(e)}")
            return 0.0, 0.0  # 出错时返回中性情感

def emotion_to_text(valence, arousal):
    """将价效度值转换为可读的情感描述"""
    # 重新调整情感判断阈值
    v_threshold = 0.15  # 价值阈值调低，提高敏感度
    a_threshold = 0.15  # 效度阈值调低，提高敏感度
    
    # 根据象限确定基础情感类型
    if valence >= v_threshold and arousal >= a_threshold:
        quadrant = "喜悦/兴奋"
    elif valence >= v_threshold and arousal < -a_threshold:
        quadrant = "满足/平静"
    elif valence < -v_threshold and arousal >= a_threshold:
        quadrant = "愤怒/焦虑"
    elif valence < -v_threshold and arousal < -a_threshold:
        quadrant = "悲伤/抑郁"
    elif abs(valence) < v_threshold and abs(arousal) < a_threshold:
        # 当价和效都接近0时，视为中性情感
        return "中性情感"
    elif abs(valence) < v_threshold:
        # 价接近0，以效度为主导情感
        if arousal >= a_threshold:
            quadrant = "激动/紧张"
        else:
            quadrant = "冷静/放松"
    else:  # abs(arousal) < a_threshold
        # 效接近0，以价值为主导情感
        if valence >= v_threshold:
            quadrant = "愉悦/满意"
        else:
            quadrant = "不满/失望"
    
    # 计算情感强度
    # 使用非线性映射增强对中等强度的感知
    intensity = (abs(valence) + abs(arousal)) / 2
    intensity = min(1.0, intensity * 1.2)  # 稍微放大强度感知
    
    if intensity < 0.25:
        return f"轻微的{quadrant}"
    elif intensity < 0.5:
        return f"中等的{quadrant}"
    else:
        if "满足" in quadrant or "悲伤" in quadrant:
            return f"深度的{quadrant}"
        else:
            return f"强烈的{quadrant}"

def main():
    parser = argparse.ArgumentParser(description='使用LTC-NCP-VA模型进行情感分析')
    parser.add_argument('--model', type=str, default=None,
                      help='模型文件路径(默认使用最新的safetensors模型)')
    parser.add_argument('--config', type=str, default='configs/optimized_performance.yaml',
                      help='模型配置文件路径')
    parser.add_argument('--vocab', type=str, default='model_exports/vocab.txt',
                      help='词汇表文件路径')
    parser.add_argument('--text', type=str, default=None,
                      help='要分析的文本(如未指定则进入交互模式)')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='运行设备，可选cpu或cuda (默认: cpu)')
    parser.add_argument('--no-calibrate', action='store_true',
                      help='不对模型输出进行校准处理(禁用非线性校准)')
    
    args = parser.parse_args()
        
    # 确定是否进行校准
    calibrate = not args.no_calibrate
    if calibrate:
        logger.info("已启用情感校准，用于增强情感表达")
    
    # 检查是否存在model_exports目录
    if not os.path.isdir('model_exports'):
        logger.error("未找到model_exports目录，请先运行提取脚本")
        sys.exit(1)
    
    # 检查词汇表文件
    if not os.path.exists(args.vocab):
        logger.error(f"词汇表文件不存在: {args.vocab}")
        logger.info("请先运行extract_vocab.py生成词汇表")
        sys.exit(1)
    
    # 查找模型文件
    model_path = args.model
    if not model_path:
        # 查找最新的safetensors模型
        from glob import glob
        model_files = glob('model_exports/*.safetensors')
        if not model_files:
            logger.error("未找到safetensors模型文件，请先运行提取脚本")
            sys.exit(1)
        model_path = max(model_files, key=os.path.getmtime)
        logger.info(f"使用最新的模型文件: {model_path}")
    
    # 创建分词器
    tokenizer = SimpleTokenizer(vocab_file=args.vocab, vocab_size=10000)
    
    # 加载设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # 加载模型
    logger.info(f"正在加载模型...")
    model = load_model_from_safetensors(model_path, args.config, device)
    
    if not model:
        logger.error("模型加载失败")
        sys.exit(1)
    
    logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 交互模式或单次分析
    if args.text:
        # 单次分析模式
        valence, arousal = predict_emotion(model, tokenizer, args.text, device, calibrate)
        emotion_text = emotion_to_text(valence, arousal)
        
        print("\n" + "=" * 50)
        print(f"文本: {args.text}")
        print(f"价效度 (V-A): [{valence:.4f}, {arousal:.4f}]")
        print(f"情感: {emotion_text}")
        print("=" * 50)
    else:
        # 交互模式
        print("\n" + "=" * 50)
        print("LTC-NCP-VA 情感分析系统 - 交互模式")
        print("输入文本进行情感分析，输入'退出'或'exit'退出")
        if calibrate:
            print("注: 已启用情感校准增强")
        print("=" * 50 + "\n")
        
        while True:
            try:
                text = input("请输入文本 > ")
                if text.lower() in ['exit', 'quit', '退出']:
                    break
                    
                if not text.strip():
                    continue
                
                valence, arousal = predict_emotion(model, tokenizer, text, device, calibrate)
                emotion_text = emotion_to_text(valence, arousal)
                
                print(f"价效度 (V-A): [{valence:.4f}, {arousal:.4f}]")
                print(f"情感: {emotion_text}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {str(e)}\n")
        
        print("\n感谢使用LTC-NCP-VA情感分析系统！")

if __name__ == "__main__":
    main() 