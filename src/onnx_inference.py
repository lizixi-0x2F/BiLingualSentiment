import argparse
import numpy as np
import onnxruntime as ort
from transformers import XLMRobertaTokenizer

def load_tokenizer():
    """加载XLM-RoBERTa分词器"""
    return XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_text(tokenizer, text, max_length=128):
    """使用XLM-RoBERTa分词器对文本进行分词"""
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }

def run_onnx_inference(model_path, text, max_length=128):
    """使用ONNX模型进行推理"""
    # 加载分词器
    tokenizer = load_tokenizer()
    
    # 分词
    inputs = tokenize_text(tokenizer, text, max_length)
    
    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # 获取输入名称
    input_names = [input.name for input in session.get_inputs()]
    
    # 准备输入
    ort_inputs = {}
    for name in input_names:
        if name in inputs:
            ort_inputs[name] = inputs[name]
    
    # 运行推理
    print(f"进行推理: '{text}'")
    ort_outputs = session.run(None, ort_inputs)
    
    # 解析结果
    output = ort_outputs[0][0]
    valence, arousal = output[0], output[1]
    
    result = {
        "text": text,
        "valence": float(valence),
        "arousal": float(arousal)
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="使用ONNX模型进行情感分析推理")
    parser.add_argument("--model_path", type=str, required=True, help="ONNX模型路径")
    parser.add_argument("--text", type=str, required=True, help="要分析的文本")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    
    args = parser.parse_args()
    
    result = run_onnx_inference(args.model_path, args.text, args.max_length)
    
    print("\n结果:")
    print(f"文本: {result['text']}")
    print(f"价效值: {result['valence']:.4f}")
    print(f"唤起值: {result['arousal']:.4f}")

if __name__ == "__main__":
    main() 