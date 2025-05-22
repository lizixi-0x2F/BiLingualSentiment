# 配置文件说明

本目录包含各种模型配置文件，用于训练不同规模的模型。

## 教师模型配置

- `teacher_config.json` - 标准教师模型配置，使用XLM-R-Base + LTC_NCP架构

## 学生模型配置（不同档位）

- `student_config.json` - Micro档位 (384维隐藏层，6层Transformer，共25M参数)
- `student_config_small.json` - Small档位 (512维隐藏层，8层Transformer，共46M参数)
- `student_config_medium.json` - Medium档位 (640维隐藏层，12层Transformer，共79M参数)
- `student_config_large_s.json` - Large-S档位 (768维隐藏层，8层Transformer，共149M参数)

## 使用方法

### 训练教师模型

```bash
./run_train_teacher.sh
```

### 训练特定档位学生模型

```bash
./run_train_student.sh configs/student_config_small.json
```

### 运行演示程序

```bash
./run_demo.sh student small
```

或者使用教师模型：

```bash
./run_demo.sh teacher
```

### 运行完整流程

```bash
./run_all.sh small
```

## 高级配置选项

### 正则化选项

- `use_rdrop` - 是否使用R-Drop正则化 (2× forward pass + 双向KL损失)
  - `rdrop_alpha` - R-Drop损失权重系数
  
- `use_gradnorm` - 是否使用GradNorm动态平衡Valence和Arousal损失
  - 自动平衡多任务学习中的不同损失权重，使训练更加稳定
  
- `use_fgm` - 是否使用FGM对抗训练 
  - `fgm_epsilon` - 对抗扰动大小参数（默认1e-3）
  - 在嵌入层添加扰动提高模型鲁棒性，有效防止过拟合

### 液态神经网络参数

- `ltc_hidden_size` - 液态神经网络隐藏层大小
- `ltc_memory_size` - 液态神经网络记忆单元大小
- `ltc_num_layers` - 液态神经网络层数
- `ltc_dropout` - 液态神经网络Dropout比率

### 知识蒸馏参数（仅学生模型）

- `temperature` - 蒸馏温度参数，控制软标签的"软化"程度
- `soft_target_loss_weight` - 软目标损失权重
- `hard_target_loss_weight` - 硬目标损失权重

## 特殊网络设计

### 液态时间常数网络(LTC)

液态时间常数网络是一种特殊的循环神经网络变体，通过动态时间常数实现"液态"行为，特别适合处理序列数据。其核心特点：

- 每个神经元拥有独立的时间常数参数，自适应调整信息流
- 通过衰减系数控制新旧信息的融合比例
- 对长距离依赖有较强的建模能力

### 神经回路机制(NCP)

神经回路机制是LTC的增强版本，实现了更复杂的信息处理路径：

- 层间正向连接：允许信息从低层直接流向高层
- 跳跃连接：实现残差学习，缓解梯度消失问题
- 反馈连接：高层信息可以回流到低层，形成循环路径
- 层归一化：稳定训练过程，加速收敛 