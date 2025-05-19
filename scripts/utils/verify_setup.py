#!/usr/bin/env python
"""
项目设置验证脚本

此脚本用于验证项目环境是否正确设置，包括：
- 验证依赖项安装
- 检查数据文件
- 检查目录结构
- 验证模型文件路径
- 如果可用，简单运行推理测试

用法：
    python scripts/utils/verify_setup.py
"""

import os
import sys
import importlib
import platform
from pathlib import Path

def green(text):
    """返回绿色文本"""
    return f"\033[92m{text}\033[0m"

def red(text):
    """返回红色文本"""
    return f"\033[91m{text}\033[0m"

def yellow(text):
    """返回黄色文本"""
    return f"\033[93m{text}\033[0m"

def check_dependencies():
    """检查必要依赖项是否已安装"""
    print("检查依赖项...")
    required_packages = ['torch', 'transformers', 'numpy', 'pandas', 'matplotlib', 'scikit-learn']
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  - {package}: {green('已安装')}")
        except ImportError:
            print(f"  - {package}: {red('未安装')}")
            missing.append(package)
    
    if missing:
        print(f"\n{yellow('缺少以下依赖项，建议运行:')} pip install {' '.join(missing)}")
    else:
        print(f"\n{green('所有基础依赖项已安装')}")
    
    # 检查CUDA可用性
    if importlib.util.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA: {green('可用')} (设备: {torch.cuda.get_device_name(0)})")
        else:
            print(f"CUDA: {yellow('不可用')} - 模型将在CPU上运行，这可能会很慢")
    
    return len(missing) == 0

def check_directory_structure():
    """检查项目目录结构"""
    print("\n检查目录结构...")
    root_path = Path(__file__).parent.parent.parent
    
    required_dirs = [
        'data',
        'scripts/inference',
        'scripts/training',
        'scripts/utils',
        'src/models',
        'src/utils',
        'docs',
        'outputs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = root_path / dir_path
        if full_path.exists():
            print(f"  - {dir_path}: {green('存在')}")
        else:
            print(f"  - {dir_path}: {red('不存在')}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n{yellow('检测到缺少的目录结构，请运行项目重组脚本')}")
    else:
        print(f"\n{green('目录结构正确')}")
    
    return len(missing_dirs) == 0

def check_data_files():
    """检查必要的数据文件"""
    print("\n检查数据文件...")
    root_path = Path(__file__).parent.parent.parent
    
    required_files = [
        'data/Chinese_VA_dataset_gaussNoise.csv',
        'data/emobank_va_normalized.csv',
        'data/example_texts.txt',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = root_path / file_path
        if full_path.exists():
            print(f"  - {file_path}: {green('存在')}")
        else:
            print(f"  - {file_path}: {red('不存在')}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n{yellow('部分数据文件缺失，你可能需要下载它们')}")
    else:
        print(f"\n{green('所有必要的数据文件都存在')}")
    
    return len(missing_files) == 0

def check_model_files():
    """检查模型文件是否存在，或是否可以通过Hugging Face下载"""
    print("\n检查模型文件...")
    root_path = Path(__file__).parent.parent.parent
    
    model_files = [
        'outputs/pretrained_distilbert_local/best_model.pth',
        'outputs/pretrained_xlm_roberta_local/best_model.pth'
    ]
    
    missing_models = []
    for model_file in model_files:
        full_path = root_path / model_file
        if full_path.exists():
            print(f"  - {model_file}: {green('已存在')}")
        else:
            print(f"  - {model_file}: {yellow('未找到')} (可使用Hugging Face下载)")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n{yellow('模型文件缺失，你可以使用以下命令下载：')}")
        print(f"  python scripts/utils/download_models_from_hf.py")
    else:
        print(f"\n{green('所有模型文件都已存在')}")
    
    # 检查网络连接到Hugging Face
    try:
        import requests
        response = requests.get('https://huggingface.co/', timeout=5)
        if response.status_code == 200:
            print(f"Hugging Face API: {green('可访问')}")
        else:
            print(f"Hugging Face API: {yellow('可能无法访问')} (HTTP状态码: {response.status_code})")
    except Exception as e:
        print(f"Hugging Face API: {red('无法访问')} - {e}")
    
    return True  # 即使缺少模型也返回True，因为可以下载

def run_basic_test():
    """运行简单的测试以确保项目能够工作"""
    print("\n尝试运行基本测试...")
    root_path = Path(__file__).parent.parent.parent
    
    # 检查模型文件
    model_path = root_path / 'outputs/pretrained_distilbert_local/best_model.pth'
    if not model_path.exists():
        print(f"{yellow('跳过测试')} - 模型文件不存在，请先下载模型")
        return True
    
    # 导入必要的模块
    try:
        sys.path.append(str(root_path))
        from src.config import Config
        from src.models.roberta_model import MultilingualDistilBERTModel
        import torch
        
        # 创建配置
        config = Config()
        
        # 初始化模型
        model = MultilingualDistilBERTModel(config)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试是否可以获取tokenizer
        tokenizer = model.get_tokenizer()
        test_text = "这是一个测试文本。"
        
        # 对文本进行编码
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # 运行推理
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
        
        print(f"模型测试: {green('成功')}")
        print(f"  - 输入文本: \"{test_text}\"")
        print(f"  - 效价(Valence): {outputs[0, 0].item():.4f}")
        print(f"  - 唤醒度(Arousal): {outputs[0, 1].item():.4f}")
        
        return True
    
    except Exception as e:
        print(f"模型测试: {red('失败')} - {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print(f"中英文情感分析模型 - 项目验证")
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    checks = [
        check_dependencies(),
        check_directory_structure(),
        check_data_files(),
        check_model_files()
    ]
    
    if all(checks):
        print("\n" + "=" * 60)
        print(f"{green('项目基本设置已验证')} - 尝试运行模型测试")
        run_basic_test()
    else:
        print("\n" + "=" * 60)
        print(f"{yellow('项目设置需要调整')} - 请修复上面的问题")
    
    print("\n" + "=" * 60)
    print("如需更多帮助，请参阅文档或提交Issue")
    print("=" * 60)

if __name__ == "__main__":
    main()
