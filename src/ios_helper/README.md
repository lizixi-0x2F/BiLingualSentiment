# iOS集成指南

本指南介绍如何在iOS应用中集成双语情感分析模型。

## 准备工作

1. 下载ONNX Runtime Mobile SDK：
   - 访问 [ONNX Runtime官网](https://onnxruntime.ai/)
   - 下载适用于iOS的预编译库或使用CocoaPods安装

2. 模型文件
   - 将转换好的`.onnx`模型文件添加到Xcode项目中

## 使用CocoaPods安装ONNX Runtime

在`Podfile`中添加：

```ruby
pod 'onnxruntime-objc'
```

然后运行：

```bash
pod install
```

## 使用Swift Package Manager安装

如果您使用Swift Package Manager，可以添加以下依赖：

```swift
dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift.git", from: "1.0.0")
]
```

## 集成步骤

1. 将模型文件添加到项目中：
   - 将`student_micro.onnx`或`student_large_s.onnx`拖放到Xcode项目中
   - 确保在"Copy Bundle Resources"中包含该文件

2. 将`SentimentPredictorONNX.swift`文件添加到项目中
   - 这个文件提供了ONNX模型的包装器

3. 在代码中使用预测器：

```swift
import UIKit
import onnxruntime_objc

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            // 加载模型
            guard let modelPath = Bundle.main.path(forResource: "student_micro", ofType: "onnx") else {
                print("找不到模型文件")
                return
            }
            
            let predictor = try SentimentPredictorONNX(modelPath: modelPath)
            
            // 进行预测
            let text = "这是一个令人感动的故事"
            let result = try predictor.predict(text: text)
            
            // 使用预测结果
            print("文本: \(result.text)")
            print("价效值: \(result.valence)")
            print("唤起值: \(result.arousal)")
            
        } catch {
            print("错误: \(error)")
        }
    }
}
```

## 注意事项

1. 分词器
   - 当前提供的`SimpleTokenizer`是一个示例实现
   - 在实际应用中，您需要实现与Python端相同的分词逻辑
   - 理想情况下，您应该使用XLM-R的分词器的Swift/ObjC实现

2. 内存管理
   - ONNX Runtime会占用一定内存，请确保适当管理资源
   - 考虑在不需要时释放会话对象

3. 线程安全
   - 推理操作可能较为耗时，建议在后台线程中执行
   - 示例：

```swift
DispatchQueue.global(qos: .userInitiated).async {
    do {
        let result = try predictor.predict(text: text)
        
        DispatchQueue.main.async {
            // 更新UI
            self.emotionLabel.text = "价效: \(result.valence), 唤起: \(result.arousal)"
        }
    } catch {
        print("预测错误: \(error)")
    }
}
```

## 性能优化

1. 量化
   - 考虑使用量化模型以减小文件大小并提高推理速度
   - 可以使用`onnxruntime`和`onnx-simplifier`工具进行量化

2. 模型选择
   - `student_micro.onnx`：最小模型，适合资源受限设备
   - `student_medium.onnx`：平衡大小和性能，推荐用于大多数iOS设备
   - `student_large_s.onnx`：最高精度，但需要更多资源

## 测试和调试

- 使用多种中英文文本测试模型表现
- 检查价效值和唤起值是否在预期范围内 (-1到1)
- 注意保持与服务器端行为一致性 