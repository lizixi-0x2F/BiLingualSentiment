import argparse
import os
import coremltools as ct

def convert_onnx_to_coreml(onnx_path, output_path=None, min_deployment_target=None):
    """将ONNX模型转换为Core ML格式"""
    print(f"加载ONNX模型: {onnx_path}")
    
    # 如果未指定输出路径，则从ONNX路径生成
    if output_path is None:
        model_dir = os.path.dirname(onnx_path)
        model_name = os.path.basename(onnx_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}.mlmodel")
    
    # 加载ONNX模型
    print("使用coremltools自动检测功能转换模型...")
    model = ct.convert(
        onnx_path,
        minimum_deployment_target=min_deployment_target,
        source="auto"
    )
    
    # 添加模型元数据
    model_type = "unknown"
    if "micro" in onnx_path:
        model_type = "student_micro"
    elif "small" in onnx_path:
        model_type = "student_small"
    elif "medium" in onnx_path:
        model_type = "student_medium"
    elif "large" in onnx_path:
        model_type = "student_large_s"
    
    # 添加元数据
    model.user_defined_metadata["com.lizixi.BiLingualSentiment.model"] = model_type
    model.user_defined_metadata["com.lizixi.BiLingualSentiment.version"] = "1.0"
    
    # 保存模型
    print(f"保存Core ML模型到: {output_path}")
    model.save(output_path)
    
    print("转换完成!")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="将ONNX模型转换为Core ML格式")
    parser.add_argument("--onnx_path", type=str, required=True, help="ONNX模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出Core ML模型路径")
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
    
    convert_onnx_to_coreml(args.onnx_path, args.output_path, min_deployment_target)

if __name__ == "__main__":
    main() 