# 在macOS环境中转换PyTorch模型到Core ML格式

由于Core ML是Apple的机器学习框架，在macOS环境中转换模型通常会更加可靠。本文档提供了在macOS环境中将PyTorch模型转换为Core ML格式的详细步骤。

## 准备工作

1. 确保你有一台运行macOS的计算机
2. 安装Python环境（推荐使用Homebrew）
3. 将以下文件传输到macOS计算机：
   - 训练好的模型文件 (`checkpoints/student_large_s/best_model.pt`)
   - 转换脚本和相关代码文件

## 详细步骤

### 1. 安装必要的依赖

在macOS终端中运行以下命令：

```bash
# 安装Homebrew（如果尚未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Python
brew install python

# 创建并激活虚拟环境
python -m venv coreml_env
source coreml_env/bin/activate

# 安装必要的包
pip install torch torchvision coremltools transformers
```

### 2. 准备目录结构

```bash
mkdir -p BiLingualSentiment/src/models
mkdir -p BiLingualSentiment/checkpoints/student_large_s
```

### 3. 传输必要文件

将以下文件从Linux环境传输到macOS：

- `checkpoints/student_large_s/best_model.pt` → `BiLingualSentiment/checkpoints/student_large_s/best_model.pt`
- `src/models/student_model.py` → `BiLingualSentiment/src/models/student_model.py`
- `src/pytorch_to_coreml.py` → `BiLingualSentiment/src/pytorch_to_coreml.py`
- 其他必要的依赖文件

可以使用scp命令或其他文件传输工具：

```bash
# 在Linux环境中执行
tar -czf transfer.tar.gz checkpoints/student_large_s/best_model.pt src/models/ src/pytorch_to_coreml.py

# 使用scp传输到Mac（需要替换username和mac_ip）
scp transfer.tar.gz username@mac_ip:~/

# 在Mac上解压
tar -xzf transfer.tar.gz -C BiLingualSentiment/
```

### 4. 修改转换脚本

在macOS中，打开`BiLingualSentiment/src/pytorch_to_coreml.py`文件，确保以下代码部分：

```python
# 转换模型
print("转换中...")
mlmodel = ct.convert(
    traced_model,
    inputs=inputs,
    minimum_deployment_target=min_deployment_target,
    convert_to="mlprogram"
)
```

### 5. 执行转换

```bash
cd BiLingualSentiment
mkdir -p models

# 运行转换脚本
python -m src.pytorch_to_coreml \
  --model_path checkpoints/student_large_s/best_model.pt \
  --output_path models/student_large_s.mlmodel \
  --max_seq_length 128 \
  --ios_version 15
```

### 6. 验证转换后的模型

```bash
# 检查模型大小
ls -lh models/student_large_s.mlmodel

# 使用coremltools查看模型信息
python -c "
import coremltools as ct
model = ct.models.MLModel('models/student_large_s.mlmodel')
print(model.get_spec())
"
```

### 7. 将转换后的模型传回Linux环境

```bash
# 在Mac上执行
scp models/student_large_s.mlmodel username@linux_ip:~/BiLingualSentiment/models/
```

## 简化脚本（可选）

你也可以创建一个简化的转换脚本，专门用于macOS环境：

```python
# mac_convert.py
import torch
import coremltools as ct
from src.models.student_model import StudentModel

# 加载模型
state = torch.load("checkpoints/student_large_s/best_model.pt", map_location="cpu")
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
model.load_state_dict(state["model"])
model.eval()

# 创建模型包装类
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

wrapped_model = ModelWrapper(model)

# 准备样本输入
batch_size = 1
max_seq_length = 128
input_ids = torch.ones((batch_size, max_seq_length), dtype=torch.int64)
attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.int64)

# 跟踪模型
traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))

# 转换为Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(batch_size, max_seq_length), dtype=ct.int32),
        ct.TensorType(name="attention_mask", shape=(batch_size, max_seq_length), dtype=ct.int32)
    ],
    minimum_deployment_target=ct.target.iOS15,
    convert_to="mlprogram"
)

# 添加元数据
mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.model"] = "student_large_s"
mlmodel.user_defined_metadata["com.lizixi.BiLingualSentiment.version"] = "1.0"

# 保存模型
mlmodel.save("models/student_large_s.mlmodel")
print("转换完成!")
```

## 注意事项

1. Core ML转换在macOS上通常更可靠，因为这是Apple自己的框架，工具在其原生环境中表现最佳。

2. 确保macOS环境中安装的PyTorch版本与训练模型时使用的版本兼容。

3. 如果在macOS上遇到内存问题，可以考虑使用具有更多RAM的机器，因为模型转换可能需要大量内存。

## 替代方案

如果无法访问macOS环境，建议使用ONNX格式作为替代方案，ONNX在iOS上也有很好的支持，通过ONNX Runtime Mobile可以高效运行。请参考`IOS_MODEL_USAGE.md`文档中关于使用ONNX模型的指南。 