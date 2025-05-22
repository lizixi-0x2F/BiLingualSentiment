import os
import torch
import argparse
import numpy as np
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

def create_model_wrapper(model):
    """创建模型包装类，用于转换为ONNX"""
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)
    
    return ModelWrapper(model)

def convert_to_onnx(model_path, output_path=None, max_seq_length=128):
    """将PyTorch模型转换为ONNX格式"""
    # 加载模型
    model, config = load_student_model(model_path)
    
    # 如果未指定输出路径，则从模型路径生成
    if output_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}.onnx")
    
    # 创建模型包装
    wrapped_model = create_model_wrapper(model)
    
    # 准备样本输入
    batch_size = 1
    input_ids = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    
    # 导出为ONNX格式
    print(f"将PyTorch模型转换为ONNX格式: {output_path}")
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # 添加元数据（在ONNX中以字符串形式存储）
    import onnx
    model_proto = onnx.load(output_path)
    
    # 添加模型信息
    model_config = config["model"]
    metadata = {
        "model_type": "student_" + os.path.basename(output_path).split('_')[1].split('.')[0],
        "version": "1.0",
        "hidden_size": str(model_config["hidden_size"]),
        "num_layers": str(model_config["num_hidden_layers"]),
        "max_length": str(config["data"]["max_length"]),
    }
    
    # 添加元数据到ONNX模型
    for key, value in metadata.items():
        meta = model_proto.metadata_props.add()
        meta.key = key
        meta.value = value
    
    # 保存带有元数据的模型
    onnx.save(model_proto, output_path)
    
    print(f"ONNX模型已保存到: {output_path}")
    
    # 验证ONNX模型
    try:
        import onnxruntime as ort
        print("验证ONNX模型...")
        ort_session = ort.InferenceSession(output_path)
        
        # 准备输入
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        
        # 进行推理
        ort_outputs = ort_session.run(None, ort_inputs)
        print("ONNX模型验证成功!")
        
        # 比较PyTorch和ONNX输出
        with torch.no_grad():
            torch_output = wrapped_model(input_ids, attention_mask).numpy()
        
        # 检查结果是否匹配
        np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-3, atol=1e-3)
        print("PyTorch和ONNX输出匹配!")
    except ImportError:
        print("没有找到onnxruntime，跳过验证步骤。")
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为ONNX格式")
    parser.add_argument("--model_path", type=str, default="checkpoints/student_large_s/best_model.pt", help="PyTorch模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出ONNX模型路径")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model_path, args.output_path, args.max_seq_length)

if __name__ == "__main__":
    main() 