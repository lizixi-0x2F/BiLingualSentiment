# 在iOS中使用Large-S情感分析模型

本文档介绍如何在iOS应用中集成转换后的Large-S情感分析模型。

## 模型选择

我们提供了两种格式的模型供iOS应用使用：

1. **ONNX格式**（推荐）：`models/student_large_s.onnx`
   - 使用ONNX Runtime Mobile运行
   - 更好的跨平台兼容性
   - 更稳定的运行环境

2. **Core ML格式**：(当前环境下无法完成转换)
   - 如果需要Core ML格式，建议在macOS环境下使用提供的脚本进行转换

## 使用ONNX格式模型

### 步骤1: 安装ONNX Runtime

在Xcode项目中添加ONNX Runtime依赖：

**CocoaPods**:
```ruby
pod 'onnxruntime-objc'
```

**Swift Package Manager**:
```swift
dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift.git", from: "1.0.0")
]
```

### 步骤2: 添加模型文件

1. 将`models/student_large_s.onnx`文件添加到Xcode项目中
2. 确保在"Copy Bundle Resources"中包含该文件

### 步骤3: 使用提供的Swift辅助类

将`src/ios_helper/SentimentPredictorONNX.swift`文件添加到你的项目中。这个类提供了完整的模型加载和推理功能。

### 步骤4: 集成到应用代码中

```swift
import UIKit
import onnxruntime_objc

class ViewController: UIViewController {
    
    private var predictor: SentimentPredictorONNX?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            // 加载模型
            guard let modelPath = Bundle.main.path(forResource: "student_large_s", ofType: "onnx") else {
                print("找不到模型文件")
                return
            }
            
            predictor = try SentimentPredictorONNX(modelPath: modelPath)
            
            // 测试中文文本
            analyzeText("这是一个令人感动的故事")
            
            // 测试英文文本
            analyzeText("This is a very touching story")
        } catch {
            print("模型加载错误: \(error)")
        }
    }
    
    func analyzeText(_ text: String) {
        // 在后台线程执行推理
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self, let predictor = self.predictor else { return }
            
            do {
                let result = try predictor.predict(text: text)
                
                // 回到主线程更新UI
                DispatchQueue.main.async {
                    print("文本: \(result.text)")
                    print("价效值: \(result.valence)")
                    print("唤起值: \(result.arousal)")
                    
                    // 更新UI...
                }
            } catch {
                print("分析错误: \(error)")
            }
        }
    }
}
```

## 性能优化

### 1. 内存管理

- ONNX Runtime会占用一定内存，请在不需要时释放预测器
- 考虑懒加载模型，仅在需要时初始化

```swift
// 懒加载示例
lazy var predictor: SentimentPredictorONNX? = {
    guard let modelPath = Bundle.main.path(forResource: "student_large_s", ofType: "onnx") else {
        return nil
    }
    
    do {
        return try SentimentPredictorONNX(modelPath: modelPath)
    } catch {
        print("模型加载错误: \(error)")
        return nil
    }
}()
```

### 2. 批处理

- 避免频繁创建和销毁会话对象
- 使用同一预测器实例处理多个请求

### 3. 线程管理

- 在后台线程中执行推理
- 推理完成后在主线程更新UI

## 高级选项

### 1. 使用更小的模型

如果在低端设备上性能不佳，可以考虑使用更小的模型：
- `models/student_micro.onnx` (最小，约10MB)
- `models/student_medium.onnx` (推荐，约40MB)

### 2. 模型量化

如果需要进一步减小模型尺寸，可以考虑量化ONNX模型：

```bash
pip install onnx onnxruntime
python -m onnxruntime.quantization.quantize --input models/student_large_s.onnx --output models/student_large_s_quant.onnx --quantization_mode IntegerOps
```

## 排查问题

### 常见问题

1. **模型加载失败**
   - 检查文件路径是否正确
   - 确认模型文件已正确添加到应用bundle

2. **推理结果不准确**
   - 确保使用正确的分词逻辑
   - 检查输入格式是否符合要求

3. **内存占用过高**
   - 考虑使用更小的模型
   - 检查是否有内存泄漏

### 日志和调试

在`SentimentPredictorONNX`类中添加详细日志，以便调试：

```swift
// 启用详细日志
let env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
```

## 结论

使用ONNX格式的模型是在iOS应用中集成Large-S情感分析模型的最佳选择。它提供了良好的性能和跨平台兼容性，同时具有活跃的社区支持。

如果需要Core ML格式，建议在macOS环境中使用提供的转换脚本，因为Core ML工具在macOS上运行效果最佳。 