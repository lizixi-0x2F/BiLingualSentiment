# 双语情感分析模型移动部署指南

本文档介绍如何将训练好的双语情感分析模型转换为移动设备（如iOS、Android）可用的格式。

## 概述

双语情感分析模型使用教师-学生知识蒸馏框架训练。我们提供了将学生模型转换为ONNX格式的工具，以便在移动设备上部署。

### 模型架构

- **教师模型**: XLM-R-Base + LTC_NCP (约270M参数)
- **学生模型**:
  - Micro: Mini Transformer + LTC_NCP (约5M参数)
  - Small: Mini Transformer + LTC_NCP (约10M参数)
  - Medium: Mini Transformer + LTC_NCP (约20M参数)
  - Large-S: Mini Transformer + LTC_NCP (约40M参数)

所有学生模型均使用知识蒸馏技术从教师模型中学习，在保持较高性能的同时大幅减小模型体积。

## 转换步骤

### 1. PyTorch → ONNX

我们使用ONNX(Open Neural Network Exchange)作为中间格式，它是一种开放的模型交换格式，支持多种深度学习框架。

```bash
# 转换命令
./convert_to_onnx.sh checkpoints/student_large_s/best_model.pt models/student_large_s.onnx
```

该脚本会：
- 加载PyTorch模型
- 创建模型包装类
- 导出为ONNX格式
- 添加元数据
- 验证转换后的模型

### 2. ONNX → 移动设备

ONNX模型可以直接在移动设备上使用，通过ONNX Runtime Mobile SDK：

- **iOS**: 使用`onnxruntime-objc`或`onnxruntime-swift`
- **Android**: 使用`org.onnxruntime:onnxruntime-android`

## 性能对比

| 模型 | 参数量 | 大小(PyTorch) | 大小(ONNX) | 精度 | 推理速度 |
|------|--------|--------------|------------|------|----------|
| Micro | ~5M | ~20MB | ~10MB | 基准的80% | 最快 |
| Small | ~10M | ~40MB | ~20MB | 基准的85% | 快 |
| Medium | ~20M | ~80MB | ~40MB | 基准的92% | 中等 |
| Large-S | ~40M | ~160MB | ~80MB | 基准的95% | 较慢 |

注：实际大小和性能可能会因具体设备和优化策略而异。

## 支持的设备

- **iOS**: 系统要求iOS 12+
- **Android**: 系统要求Android 6.0+ (API级别23+)

## 项目文件

- `src/convert_to_onnx.py`: 转换核心代码
- `convert_to_onnx.sh`: 转换脚本
- `test_large_model.sh`: 测试大型模型的转换和推理
- `src/ios_helper/`: iOS集成相关代码和文档
  - `SentimentPredictorONNX.swift`: Swift实现的情感分析预测器
  - `README.md`: iOS集成指南

## 移动端优化技巧

1. **量化**: 可以将模型量化为int8/uint8格式以进一步减小模型大小
2. **剪枝**: 可以在转换前对模型进行剪枝以减少参数数量
3. **模型选择**: 根据设备性能选择适当大小的模型
4. **批处理**: 避免频繁初始化模型，复用模型实例
5. **缓存**: 考虑缓存常用输入的结果

## 问题排查

### 常见问题

1. **模型加载失败**
   - 检查ONNX Runtime版本是否与模型兼容
   - 确认模型文件路径正确且有访问权限

2. **推理结果不一致**
   - 确保与服务器端使用相同的预处理/后处理逻辑
   - 检查输入数据类型和形状是否正确

3. **性能问题**
   - 使用Instruments(iOS)或Profiler(Android)分析性能瓶颈
   - 考虑在低端设备上使用更小的模型

## 示例应用

我们提供了简单的示例代码，展示如何在iOS应用中集成模型。参见`src/ios_helper/`目录。

## 结论

通过将双语情感分析模型转换为ONNX格式，我们可以在移动设备上实现高效的情感分析，无需网络连接。这使得应用可以在离线环境中提供情感分析功能，同时保持较高的响应速度和用户体验。 