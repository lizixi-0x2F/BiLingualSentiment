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

def create_model_wrapper(model):
    """创建模型包装类，用于转换为Core ML"""
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)
    
    return ModelWrapper(model)

def convert_to_coreml(model_path, output_path=None, max_seq_length=128, min_deployment_target=None):
    """将PyTorch模型转换为Core ML格式"""
    # 加载模型
    model, config = load_student_model(model_path)
    
    # 如果未指定输出路径，则从模型路径生成
    if output_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}.mlmodel")
    
    # 创建模型包装
    wrapped_model = create_model_wrapper(model)
    
    # 准备样本输入
    batch_size = 1
    input_ids = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
    
    # 跟踪模型
    print("跟踪PyTorch模型...")
    traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
    
    # 定义输入
    print("准备转换为Core ML格式...")
    inputs = [
        ct.TensorType(name="input_ids", shape=(batch_size, max_seq_length), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(batch_size, max_seq_length), dtype=np.int32)
    ]
    
    # 添加模型信息
    model_config = config["model"]
    
    # 确定模型类型
    model_type = "unknown"
    if "micro" in model_path or model_config["hidden_size"] <= 384:
        model_type = "student_micro"
    elif "small" in model_path or model_config["hidden_size"] <= 512:
        model_type = "student_small"
    elif "medium" in model_path or model_config["hidden_size"] <= 640:
        model_type = "student_medium"
    elif "large" in model_path or model_config["hidden_size"] <= 768:
        model_type = "student_large_s"
    
    # 转换模型
    print("转换中...")
    try:
        # 尝试使用TorchScript直接转换
        print("尝试使用TorchScript路径转换...")
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            minimum_deployment_target=min_deployment_target,
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT
        )
        
        # 添加元数据
        mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.model"] = model_type
        mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.version"] = "1.0"
        mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.hidden_size"] = str(model_config["hidden_size"])
        mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.num_layers"] = str(model_config["num_hidden_layers"])
        
        # 保存模型
        print(f"保存Core ML模型到: {output_path}")
        mlmodel.save(output_path)
        
        print("转换完成!")
        return output_path
    
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        print("尝试使用替代方法...")
        
        try:
            # 先保存为ONNX，再从ONNX转换为Core ML
            print("1. 导出到临时ONNX文件...")
            temp_onnx_path = os.path.join(os.path.dirname(output_path), "temp_model.onnx")
            
            # 导出为ONNX格式
            torch.onnx.export(
                wrapped_model,
                (input_ids, attention_mask),
                temp_onnx_path,
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
            
            print(f"2. 从ONNX转换为Core ML...")
            mlmodel = ct.convert(
                temp_onnx_path,
                minimum_deployment_target=min_deployment_target,
                source="auto"
            )
            
            # 添加元数据
            mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.model"] = model_type
            mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.version"] = "1.0"
            mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.hidden_size"] = str(model_config["hidden_size"])
            mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.num_layers"] = str(model_config["num_hidden_layers"])
            
            # 保存模型
            print(f"保存Core ML模型到: {output_path}")
            mlmodel.save(output_path)
            
            # 清理临时文件
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)
                
            print("转换完成!")
            return output_path
            
        except Exception as second_e:
            print(f"替代方法也失败: {second_e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为Core ML格式")
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出Core ML模型路径")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--ios_version", type=str, default="15", help="最低iOS部署目标版本")
    
    args = parser.parse_args()
    
    # 设置最低部署目标
    min_deployment_target = None
    if args.ios_version == "15":
        min_deployment_target = ct.target.iOS15
    elif args.ios_version == "16":
        min_deployment_target = ct.target.iOS16
    elif args.ios_version == "17":
        min_deployment_target = ct.target.iOS17
    
    convert_to_coreml(args.model_path, args.output_path, args.max_seq_length, min_deployment_target)

if __name__ == "__main__":
    main() 