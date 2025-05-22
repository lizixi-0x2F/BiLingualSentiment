import Foundation
import CoreML

/// 情感分析预测器
class SentimentPredictor {
    
    /// 预测结果结构
    struct PredictionResult {
        let valence: Float  // 价效值 (-1 到 1)
        let arousal: Float  // 唤起值 (-1 到 1)
        let text: String    // 输入文本
    }
    
    private let model: MLModel
    private let tokenizer: TokenizerProtocol
    private let maxLength: Int
    
    /// 初始化预测器
    /// - Parameters:
    ///   - modelURL: 模型文件URL
    ///   - tokenizer: 分词器 (默认为简单分词器)
    ///   - maxLength: 最大序列长度 (默认为128)
    init(modelURL: URL, tokenizer: TokenizerProtocol = SimpleTokenizer(), maxLength: Int = 128) throws {
        self.model = try MLModel(contentsOf: modelURL)
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        
        print("模型已加载: \(modelURL.lastPathComponent)")
    }
    
    /// 对文本进行情感分析
    /// - Parameter text: 输入文本
    /// - Returns: 预测结果包含价效值和唤起值
    func predict(text: String) throws -> PredictionResult {
        // 对文本进行分词
        let tokenized = tokenizer.tokenize(text: text, maxLength: maxLength)
        
        // 创建模型输入
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": tokenized.inputIds as NSArray,
            "attention_mask": tokenized.attentionMask as NSArray
        ])
        
        // 执行推理
        let prediction = try model.prediction(from: inputFeatures)
        
        // 解析输出
        guard let outputFeatures = prediction.featureValue(for: "output_0"),
              let multiArray = outputFeatures.multiArrayValue else {
            throw PredictorError.invalidOutput
        }
        
        // 获取价效和唤起值
        let valence = multiArray[0].floatValue
        let arousal = multiArray[1].floatValue
        
        return PredictionResult(valence: valence, arousal: arousal, text: text)
    }
    
    enum PredictorError: Error {
        case invalidOutput
    }
}

/// 分词器协议
protocol TokenizerProtocol {
    func tokenize(text: String, maxLength: Int) -> TokenizedResult
}

/// 分词结果
struct TokenizedResult {
    let inputIds: [Int32]
    let attentionMask: [Int32]
}

/// 简单分词器实现
/// 注意: 在实际应用中，你需要实现与Python端相同的分词逻辑
class SimpleTokenizer: TokenizerProtocol {
    func tokenize(text: String, maxLength: Int) -> TokenizedResult {
        // 简单实现，实际应用中需要替换为真实的分词逻辑
        var inputIds = [Int32](repeating: 0, count: maxLength)
        let attentionMask = [Int32](repeating: 1, count: maxLength)
        
        // 将每个字符转换为ID (简单示例)
        for (i, char) in text.enumerated() {
            if i < maxLength - 1 {
                inputIds[i + 1] = Int32(char.asciiValue ?? 0)
            }
        }
        
        return TokenizedResult(inputIds: inputIds, attentionMask: attentionMask)
    }
}

// 示例用法
/*
do {
    // 加载模型
    let modelURL = Bundle.main.url(forResource: "StudentLarge", withExtension: "mlmodel")!
    let predictor = try SentimentPredictor(modelURL: modelURL)
    
    // 预测情感
    let result = try predictor.predict(text: "这是一个非常感人的故事")
    
    // 打印结果
    print("价效值: \(result.valence), 唤起值: \(result.arousal)")
} catch {
    print("错误: \(error)")
}
*/ 