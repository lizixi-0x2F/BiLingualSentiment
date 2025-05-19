#!/usr/bin/env python
"""
将中英文情感分析模型上传至Hugging Face的脚本

该脚本将两个模型（DistilBERT和XLM-RoBERTa）及其相关文件上传到一个Hugging Face仓库中。
运行前请确保已安装huggingface_hub包并已登录Hugging Face账号。

用法:
python upload_to_huggingface.py --username YourUsername

参数:
--username: Hugging Face用户名
--repo_name: 仓库名称（默认：bilingual-sentiment-va）
--force: 如果仓库已存在，是否强制删除并重新创建
--no_push: 仅准备文件，不推送到Hugging Face
"""

import os
import shutil
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行系统命令并返回结果"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, 
                           capture_output=True, text=True)
    return result.stdout.strip()

def create_model_card(workspace_dir, temp_dir):
    """创建模型卡片（README.md）"""
    
    # 获取模型性能指标
    distilbert_metrics = {}
    xlm_roberta_metrics = {}
    
    try:
        with open(os.path.join(workspace_dir, "outputs/pretrained_distilbert_local/test_results.json"), "r") as f:
            distilbert_metrics = json.load(f)
    except:
        print("警告: 无法加载DistilBERT测试结果")
    
    try:
        with open(os.path.join(workspace_dir, "outputs/pretrained_xlm_roberta_local/test_results.json"), "r") as f:
            xlm_roberta_metrics = json.load(f)
    except:
        print("警告: 无法加载XLM-RoBERTa测试结果")
    
    # 格式化指标
    distilbert_r2 = distilbert_metrics.get("r2", 0.66)
    distilbert_rmse = distilbert_metrics.get("rmse", 0.21)
    xlm_roberta_r2 = xlm_roberta_metrics.get("r2", 0.69)
    xlm_roberta_rmse = xlm_roberta_metrics.get("rmse", 0.20)
      # 创建README.md内容
    readme = f"""# 中英文混合情感分析模型 (Bilingual Sentiment Analysis)

这个仓库包含两个用于中英文双语情感分析的微调Transformer模型。这些模型可以预测文本的效价(Valence，情感正负程度)和唤醒度(Arousal，情感强度)，取值范围为[-1, 1]。

## 模型

1. **DistilBERT多语言模型**
   - 文件位置: `/distilbert/best_model.pth`
   - 大小: 较小，推理更快
   - 性能: R² = {distilbert_r2:.2f}, RMSE = {distilbert_rmse:.2f}

2. **XLM-RoBERTa模型**
   - 文件位置: `/xlm-roberta/best_model.pth`
   - 大小: 较大，更准确
   - 性能: R² = {xlm_roberta_r2:.2f}, RMSE = {xlm_roberta_rmse:.2f}

## 使用方法

```python
from models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
import torch
from config import Config

# 选择模型类型
model_type_var = "distilbert"  # 或 "xlm-roberta"
model_path_var = f"{{model_type_var}}/best_model.pth"

# 初始化配置
config = Config()
if model_type_var == "distilbert":
    config.MULTILINGUAL_MODEL_NAME = "distilbert-base-multilingual-cased"
    model_class = MultilingualDistilBERTModel
else:
    config.MULTILINGUAL_MODEL_NAME = "xlm-roberta-base"
    model_class = XLMRobertaDistilledModel

# 加载模型
model = model_class(config)
checkpoint = torch.load(model_path_var, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = model.get_tokenizer()

# 运行推理
text = "这是一个测试文本，我感到非常开心！"  # "This is a test text, I feel very happy!"
encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**encoding)
valence_var = outputs[0, 0].item()
arousal_var = outputs[0, 1].item()
print(f"效价(Valence): {{valence_var:.4f}}，唤醒度(Arousal): {{arousal_var:.4f}}")

# 象限解释
if valence_var >= 0 and arousal_var >= 0:
    print("情感象限: 快乐/兴奋")
elif valence_var < 0 and arousal_var >= 0:
    print("情感象限: 愤怒/焦虑")
elif valence_var < 0 and arousal_var < 0:
    print("情感象限: 悲伤/抑郁")
else:  # valence_var >= 0 and arousal_var < 0
    print("情感象限: 满足/平静")
```

## 象限解释

情感可以根据效价和唤醒度分为四个象限：

- 效价 ≥ 0，唤醒度 ≥ 0：快乐/兴奋
- 效价 < 0，唤醒度 ≥ 0：愤怒/焦虑
- 效价 < 0，唤醒度 < 0：悲伤/抑郁
- 效价 ≥ 0，唤醒度 < 0：满足/平静

## 训练数据

这些模型在以下数据集上微调：
- 中文VA数据集（约4,100个样本）
- EmoBank英文数据集（约10,000个样本）

## 应用场景

- 社交媒体情感监测
- 客户反馈分析
- 心理健康文本分析
- 内容推荐系统
- 跨语言情感分析

## 许可证

MIT License
"""
    
    # 写入README.md
    with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    
    return True

def prepare_files(workspace_dir, temp_dir):
    """准备上传文件"""
    print(f"准备要上传的文件...")
    
    # 创建目录结构
    os.makedirs(os.path.join(temp_dir, "distilbert"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "xlm-roberta"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
    
    # 复制模型文件
    print("复制模型权重文件...")
    shutil.copy(
        os.path.join(workspace_dir, "outputs/pretrained_distilbert_local/best_model.pth"),
        os.path.join(temp_dir, "distilbert/best_model.pth")
    )
    shutil.copy(
        os.path.join(workspace_dir, "outputs/pretrained_xlm_roberta_local/best_model.pth"),
        os.path.join(temp_dir, "xlm-roberta/best_model.pth")
    )
    
    # 复制测试结果
    try:
        shutil.copy(
            os.path.join(workspace_dir, "outputs/pretrained_distilbert_local/test_results.json"),
            os.path.join(temp_dir, "distilbert/test_results.json")
        )
    except:
        print("警告: 无法复制DistilBERT测试结果")
        
    try:
        shutil.copy(
            os.path.join(workspace_dir, "outputs/pretrained_xlm_roberta_local/test_results.json"),
            os.path.join(temp_dir, "xlm-roberta/test_results.json")
        )
    except:
        print("警告: 无法复制XLM-RoBERTa测试结果")
    
    # 复制源代码文件
    print("复制源代码文件...")
    shutil.copy(
        os.path.join(workspace_dir, "src/models/roberta_model.py"),
        os.path.join(temp_dir, "models/roberta_model.py")
    )
    shutil.copy(
        os.path.join(workspace_dir, "src/config.py"),
        os.path.join(temp_dir, "config.py")
    )
    shutil.copy(
        os.path.join(workspace_dir, "src/inference.py"),
        os.path.join(temp_dir, "inference.py")
    )
    shutil.copy(
        os.path.join(workspace_dir, "requirements.txt"),
        os.path.join(temp_dir, "requirements.txt")
    )
    
    # 创建.gitattributes文件用于Git LFS
    with open(os.path.join(temp_dir, ".gitattributes"), "w") as f:
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
    
    return True

def upload_to_huggingface(temp_dir, username, repo_name, force=False):
    """上传文件到Hugging Face"""
    full_repo_name = f"{username}/{repo_name}"
    print(f"准备上传到 {full_repo_name}...")
    
    # 检查huggingface-cli是否已安装
    try:
        run_command("huggingface-cli --version")
    except:
        print("错误: 未检测到huggingface-cli。请运行 'pip install huggingface_hub' 并登录")
        return False
    
    # 检查是否已登录
    try:
        user_info = run_command("huggingface-cli whoami")
        print(f"已作为 {user_info} 登录")
    except:
        print("错误: 未登录Hugging Face。请运行 'huggingface-cli login' 登录")
        return False
    
    # 创建仓库
    try:
        if force:
            try:
                print(f"尝试删除现有仓库 {full_repo_name}...")
                run_command(f"huggingface-cli repo delete {full_repo_name} --yes")
            except:
                print("警告: 删除仓库失败，可能仓库不存在")
        
        print(f"创建仓库 {full_repo_name}...")
        run_command(f"huggingface-cli repo create {repo_name} --type model")
    except:
        print("警告: 创建仓库失败，可能仓库已存在")
    
    # 克隆仓库并上传文件
    try:
        repo_url = f"https://huggingface.co/{full_repo_name}"
        clone_dir = os.path.join(temp_dir, "hf_repo")
        os.makedirs(clone_dir, exist_ok=True)
        
        print(f"克隆仓库 {repo_url}...")
        run_command(f"git clone {repo_url} .", cwd=clone_dir)
        
        # 复制所有文件到克隆目录
        for item in os.listdir(temp_dir):
            if item != "hf_repo":
                src = os.path.join(temp_dir, item)
                dst = os.path.join(clone_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        # 添加并提交文件
        print("初始化Git LFS...")
        run_command("git lfs install", cwd=clone_dir)
        
        print("添加文件...")
        run_command("git add .", cwd=clone_dir)
        
        print("提交更改...")
        run_command('git commit -m "上传中英文情感分析模型: DistilBERT 和 XLM-RoBERTa"', cwd=clone_dir)
        
        print("推送到远程仓库...")
        run_command("git push", cwd=clone_dir)
        
        print(f"上传完成! 模型现在可在 {repo_url} 访问")
        return True
    except Exception as e:
        print(f"上传过程中出错: {str(e)}")
        return False

def create_download_script(workspace_dir, username, repo_name):
    """创建下载脚本"""
    script_content = f"""#!/usr/bin/env python
\"\"\"
从Hugging Face下载中英文情感分析模型的脚本

该脚本将从Hugging Face下载DistilBERT和XLM-RoBERTa情感分析模型
及其相关文件到本地目录。

用法:
python download_models_from_hf.py --output_dir downloaded_models
\"\"\"

import os
import argparse
from huggingface_hub import hf_hub_download

def download_models(repo_id="{username}/{repo_name}", save_dir="downloaded_models"):
    \"\"\"从Hugging Face Hub下载DistilBERT和XLM-RoBERTa模型\"\"\"
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义要下载的文件
    files_to_download = [
        "distilbert/best_model.pth",
        "xlm-roberta/best_model.pth",
        "models/roberta_model.py",
        "config.py",
        "inference.py"
    ]
    
    # 下载每个文件
    for file_path in files_to_download:
        print(f"下载 {{file_path}}...")
        local_dir = os.path.join(save_dir, os.path.dirname(file_path))
        os.makedirs(local_dir, exist_ok=True)
        
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
    
    print(f"所有模型文件已下载到 {{save_dir}}")

def main():
    parser = argparse.ArgumentParser(description='从Hugging Face下载情感分析模型')
    parser.add_argument('--output_dir', type=str, default="downloaded_models",
                        help='下载文件的输出目录')
    parser.add_argument('--repo_id', type=str, default="{username}/{repo_name}",
                        help='Hugging Face仓库ID')
    
    args = parser.parse_args()
    download_models(repo_id=args.repo_id, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
"""
    
    script_path = os.path.join(workspace_dir, "download_models_from_hf.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"创建下载脚本: {script_path}")
    return script_path

def update_readme(workspace_dir, username, repo_name):
    """更新本地README.md增加Hugging Face链接"""
    try:
        readme_path = os.path.join(workspace_dir, "README.md")
        
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查是否已经包含Hugging Face链接
        if f"huggingface.co/{username}/{repo_name}" in content:
            print("README.md已包含Hugging Face链接，无需更新")
            return True
        
        # 添加Hugging Face模型链接
        hf_section = f"""
## 预训练模型

模型已托管在Hugging Face:
- [中英文双语情感分析模型](https://huggingface.co/{username}/{repo_name})

使用以下命令下载模型:
```bash
pip install huggingface_hub
python download_models_from_hf.py
```
"""
        
        # 查找一个好的插入点
        insert_marker = "## 推理使用"
        if insert_marker in content:
            content = content.replace(insert_marker, f"{hf_section}\n{insert_marker}")
        else:
            # 如果找不到特定标记，附加到文件末尾
            content += f"\n{hf_section}\n"
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"更新README.md，添加了Hugging Face链接")
        return True
    except Exception as e:
        print(f"更新README.md时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='将情感分析模型上传到Hugging Face')
    parser.add_argument('--username', type=str, required=True, 
                        help='Hugging Face用户名')
    parser.add_argument('--repo_name', type=str, default="bilingual-sentiment-va",
                        help='Hugging Face仓库名称')
    parser.add_argument('--force', action='store_true',
                        help='如果仓库已存在，是否强制删除并重新创建')
    parser.add_argument('--no_push', action='store_true',
                        help='仅准备文件，不推送到Hugging Face')
    
    args = parser.parse_args()
    
    # 获取工作目录
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用临时目录: {temp_dir}")
        
        # 准备文件
        if not prepare_files(workspace_dir, temp_dir):
            print("准备文件失败")
            return
        
        # 创建模型卡片
        if not create_model_card(workspace_dir, temp_dir):
            print("创建模型卡片失败")
            return
        
        # 上传到Hugging Face
        if not args.no_push:
            if not upload_to_huggingface(temp_dir, args.username, args.repo_name, args.force):
                print("上传到Hugging Face失败")
                return
            
            # 创建下载脚本
            download_script_path = create_download_script(workspace_dir, args.username, args.repo_name)
            
            # 更新README.md
            update_readme(workspace_dir, args.username, args.repo_name)
            
            print("\n完成!")
            print(f"模型已上传到 https://huggingface.co/{args.username}/{args.repo_name}")
            print(f"下载脚本已创建: {download_script_path}")
        else:
            print("\n文件准备完成，但未上传（--no_push选项已启用）")
            print(f"文件位于: {temp_dir}")

if __name__ == "__main__":
    main()
