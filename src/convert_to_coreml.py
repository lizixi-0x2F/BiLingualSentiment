import os
import torch
import argparse
import numpy as np
import coremltools as ct
from src.models.student_model import StudentModel

def load_student_model(model_path):
    """加载训练好的学生模型"""
    print(f"加载模型权重: {model_path}")
    
    # 加载模型状态
    state = torch.load(model_path, map_location="cpu", weights_only=False)
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
    
    # 加载权重
    model.load_state_dict(state["model"])
    model.eval()
    
    return model, config

def get_model_metadata(config):
    """获取模型的元数据"""
    model_config = config["model"]
    metadata = {
        "com.lizixi.BiLingualSentiment.model": "student_large",
        "com.lizixi.BiLingualSentiment.version": "1.0",
        "com.lizixi.BiLingualSentiment.hidden_size": str(model_config["hidden_size"]),
        "com.lizixi.BiLingualSentiment.num_layers": str(model_config["num_hidden_layers"]),
        "com.lizixi.BiLingualSentiment.max_length": str(config["data"]["max_length"]),
    }
    return metadata

def create_model_wrapper(model, max_length):
    """创建模型包装类，用于转换为Core ML"""
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)
    
    return ModelWrapper(model)

def convert_to_coreml(model_path, output_path=None, max_seq_length=128):
    """将PyTorch模型转换为Core ML格式"""
    # 加载模型
    model, config = load_student_model(model_path)
    
    # 如果未指定输出路径，则从模型路径生成
    if output_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}.mlmodel")
    
    # 创建模型包装
    wrapped_model = create_model_wrapper(model, max_seq_length)
    
    # 准备样本输入
    batch_size = 1
    input_ids = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    
    # 跟踪模型
    print("将PyTorch模型转换为ONNX格式...")
    traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
    
    # 定义输入和输出
    inputs = [
        ct.TensorType(name="input_ids", shape=(batch_size, max_seq_length), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(batch_size, max_seq_length), dtype=np.int32)
    ]
    
    # 将PyTorch模型转换为Core ML
    print("转换为Core ML格式...")
    metadata = get_model_metadata(config)
    
    try:
        # 尝试新版API
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
            metadata_props=metadata
        )
    except TypeError:
        # 回退到旧版API
        print("回退到旧版API...")
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15
        )
        
        # 手动添加元数据
        for key, value in metadata.items():
            mlmodel.user_defined_metadata[key] = value
    
    # 保存模型
    print(f"保存Core ML模型到: {output_path}")
    mlmodel.save(output_path)
    
    print("转换完成!")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为Core ML格式")
    parser.add_argument("--model_path", type=str, default="checkpoints/student_large_s/best_model.pt", help="PyTorch模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出Core ML模型路径")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    
    args = parser.parse_args()
    
    convert_to_coreml(args.model_path, args.output_path, args.max_seq_length)

if __name__ == "__main__":
    main() 