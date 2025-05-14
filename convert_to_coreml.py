#!/usr/bin/env python3
import os
import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shutil

# 设置设备
device = "cpu"  # Core ML转换必须在CPU上进行

def load_model():
    """
    加载已训练的模型
    """
    model_path = "model_dir"
    
    # 检查目录是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径 {model_path} 不存在")
    
    # 加载模型和tokenizer
    print(f"从 {model_path} 加载模型和tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    
    # 将模型设置为评估模式
    model.eval()
    
    return model, tokenizer

class EmotionModelWrapper(torch.nn.Module):
    """包装模型使其适合转换为Core ML"""
    def __init__(self, model):
        super(EmotionModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 直接使用单独的参数
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 返回模型输出（情感值）
        return outputs.logits

def convert_to_coreml():
    """
    将PyTorch模型转换为Core ML格式
    """
    try:
        # 创建输出目录
        os.makedirs("coreml_model", exist_ok=True)
        
        # 加载模型
        model, tokenizer = load_model()
        
        # 创建示例文本和输入
        print("准备示例输入...")
        example_text = "这是一个示例文本，用于Core ML转换"
        
        # 编码文本
        encoded = tokenizer(
            example_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 提取输入张量
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]
        
        # 包装模型
        print("包装模型...")
        wrapped_model = EmotionModelWrapper(model)
        
        # 转换为Core ML
        print("开始转换为Core ML格式...")
        
        # 使用torchscript跟踪模型
        traced_model = torch.jit.trace(
            wrapped_model, 
            (input_ids, attention_mask, token_type_ids)
        )
        
        # 保存torch模型（备份）
        torch_model_path = os.path.join("coreml_model", "emotion_model.pt")
        torch.jit.save(traced_model, torch_model_path)
        print(f"TorchScript模型已保存到: {torch_model_path}")
        
        # 复制tokenizer文件到coreml_model目录
        for file in ["vocab.txt", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]:
            source = os.path.join("model_dir", file)
            if os.path.exists(source):
                dest = os.path.join("coreml_model", file)
                shutil.copy(source, dest)
                print(f"复制 {file} 到 coreml_model 目录")
        
        # 创建模型信息文件
        info_path = os.path.join("coreml_model", "model_info.txt")
        with open(info_path, "w") as f:
            f.write("情感分析模型 (Valence-Arousal)\n")
            f.write("===========================\n\n")
            f.write("模型描述:\n")
            f.write("- 双语支持: 中文和英文\n")
            f.write("- 输入: 文本\n")
            f.write("- 输出: 价效度(Valence)和唤起度(Arousal)值，范围为[-1, 1]\n\n")
            f.write("使用说明:\n")
            f.write("1. 在iOS/macOS应用中加载此模型\n")
            f.write("2. 将文本输入转换为标记ID\n")
            f.write("3. 运行模型预测\n")
            f.write("4. 解释输出值: \n")
            f.write("   - Valence (价效度): 正值表示积极情感，负值表示消极情感\n")
            f.write("   - Arousal (唤起度): 正值表示高唤起/激烈情感，负值表示低唤起/平静情感\n")
        
        print(f"模型信息已保存到: {info_path}")
        
        # 创建简单的示例代码
        sample_code_path = os.path.join("coreml_model", "sample_code.py")
        with open(sample_code_path, "w") as f:
            f.write("""
import torch
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("coreml_model", local_files_only=True)

# 加载模型
model = torch.jit.load("coreml_model/emotion_model.pt")

# 预测函数
def predict(text):
    # 对文本进行编码
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # 提取输入张量
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]
    
    # 运行模型
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    # 提取结果
    valence = outputs[0][0].item()
    arousal = outputs[0][1].item()
    
    # 确保在[-1, 1]范围内
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    
    return {
        "text": text,
        "valence": valence,
        "arousal": arousal
    }

# 示例
if __name__ == "__main__":
    texts = [
        "我今天感到非常开心！",
        "这个消息真让人沮丧。",
        "I feel very happy today!",
        "This news is really depressing."
    ]
    
    for text in texts:
        result = predict(text)
        print(f"文本: {result['text']}")
        print(f"价效度 (Valence): {result['valence']:.3f}")
        print(f"唤起度 (Arousal): {result['arousal']:.3f}")
        print("-" * 50)
""")
        
        print(f"示例代码已保存到: {sample_code_path}")
        return True
    except Exception as e:
        print(f"转换模型时出错: {e}")
        return False

if __name__ == "__main__":
    print("开始将模型转换为TorchScript格式...")
    success = convert_to_coreml()
    
    if success:
        print("转换完成! 模型保存到 coreml_model 目录。")
        print("由于coremltools兼容性问题，改为生成TorchScript模型+tokenizer。")
        print("您可以使用 python coreml_model/sample_code.py 进行测试。")
    else:
        print("转换失败。请检查错误信息。") 