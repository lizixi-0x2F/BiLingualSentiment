import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 设置设备 - 添加MPS支持（Apple Silicon芯片加速）
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"使用设备: {device}")

def load_model(model_path="model_dir"):
    """
    加载模型，支持中英文
    
    参数:
        model_path: 模型路径，默认为model_dir
    """
    # 确保使用本地路径
    model_path = os.path.abspath(model_path)
    
    # 检查模型目录是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型目录 {model_path} 不存在")
        
    # 加载tokenizer和模型
    print(f"从本地目录加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    
    return tokenizer, model

def predict(text, tokenizer, model):
    """预测文本的情感值"""
    try:
        # 将文本转换为模型输入格式
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # 将输入张量移动到相同设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 设置模型为评估模式
        model.eval()
        
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()
        
        # 如果只有一个样本，确保predictions是数组
        if not isinstance(predictions, np.ndarray) or predictions.ndim == 0:
            predictions = np.array([predictions])
        
        # 确保predictions是二维的 [batch_size, num_labels]
        if predictions.ndim == 1 and len(predictions) == 2:
            # 单个样本，形状是 [num_labels]
            valence = float(predictions[0])
            arousal = float(predictions[1])
        else:
            # 可能是批量预测，取第一个样本的结果
            valence = float(predictions[0][0] if predictions.ndim > 1 else predictions[0])
            arousal = float(predictions[0][1] if predictions.ndim > 1 else predictions[1])
        
        # 将结果限制在[-1, 1]范围内，与训练数据保持一致
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))
        
        return {
            "text": text,
            "valence": valence,
            "arousal": arousal
        }
    except Exception as e:
        print(f"预测文本时出错: {e}")
        print(f"问题文本: {text}")
        # 返回默认值
        return {
            "text": text,
            "valence": 0.0,  # 默认中性值修改为0（范围中间值）
            "arousal": 0.0   # 默认中性值修改为0（范围中间值）
        }

def batch_predict(texts, model_path="model_dir", batch_size=8):
    """批量预测多个文本的情感值"""
    # 加载模型
    tokenizer, model = load_model(model_path)
    
    # 对文本进行分批处理，减少内存使用
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        for text in batch_texts:
            result = predict(text, tokenizer, model)
            results.append(result)
            
        # 清理GPU缓存，如果使用GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    # 英文示例文本
    english_texts = [
        "I feel very happy and excited!",
        "This is really depressing news.",
        "I am very satisfied with this result.",
        "I'm feeling quite anxious about the upcoming exam.",
        "The sunset was breathtakingly beautiful.",
        "The horror movie was terrifying and made me jump.",
        "I'm feeling rather neutral about the whole situation."
    ]
    
    # 中文示例文本（包括诗词）
    chinese_texts = [
        "春眠不觉晓，处处闻啼鸟。",
        "床前明月光，疑是地上霜。",
        "飞流直下三千尺，疑是银河落九天。",
        "我今天感到非常开心！",
        "这个消息真让人沮丧。",
        "北堂天未晚，游子归来早。堂前一夜风，开遍宜男草。",
        "岁穷浩长叹，心事如波澜。高卧老将至，相思天正寒。浮云纷霰雪，陋巷独瓢箪。里饭非无意，悠悠行路难。",
        "精舍初成近惠泉，几回辰至起僧眠。新正屈指今三日，旧约惊心又一年。慈母板舆应偶尔，佳宾草饭乃居然。凭君莫问山中事，万里青云彩鹢前。"
    ]
    
    # 所有测试文本
    all_texts = english_texts + chinese_texts
    
    try:
        # 进行预测
        print("\n===== 中英文双语情感预测 =====")
        results = batch_predict(all_texts)
        
        # 打印结果
        for i, result in enumerate(results):
            # 标识语言
            lang = "英文" if i < len(english_texts) else "中文"
            print(f"语言: {lang}")
            print(f"文本: {result['text']}")
            print(f"价效度 (Valence): {result['valence']:.3f}")
            print(f"唤起度 (Arousal): {result['arousal']:.3f}")
            print("-" * 50)
            
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        
        # 检查模型文件是否存在
        if not os.path.exists("model_dir"):
            print(f"警告: 模型目录 model_dir 不存在。请先运行训练脚本。") 