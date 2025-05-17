import torch
import argparse
import os
import sys
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.emotion_model import EmotionAnalysisModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, config):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        config: 配置对象
        
    Returns:
        model: 加载的模型
    """
    try:
        # 设置设备
        device = torch.device(config.DEVICE if torch.cuda.is_available() or hasattr(torch.backends, "mps") 
                              and torch.backends.mps.is_available() else "cpu")
        
        # 创建模型
        model = EmotionAnalysisModel(config).to(device)
        
        # 加载模型参数，设置weights_only=False以处理numpy数组
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 显示模型信息
        logger.info(f"模型加载成功，训练轮次: {checkpoint.get('epoch', 'unknown')}")
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            logger.info(f"验证集指标: RMSE={metrics.get('rmse', 'N/A'):.4f}, R2={metrics.get('r2', 'N/A'):.4f}")
        
        # 设置为评估模式
        model.eval()
        
        return model, device
    
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def predict(text, model, device):
    """
    对文本进行情感预测
    
    Args:
        text: 输入文本
        model: 模型
        device: 设备
        
    Returns:
        valence: 效价值
        arousal: 唤醒度值
    """
    try:
        # 使用模型内置的分词器处理文本
        # 这里我们直接使用模型的encoder.tokenizer
        tokenizer = model.encoder.tokenizer
        
        # 编码文本
        encoded = tokenizer.batch_encode([text])
        
        # 移到设备上
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # 推理
        with torch.no_grad():
                outputs = model(input_ids, attention_mask)
        
        # 检查输出是否包含NaN
        if torch.isnan(outputs).any():
            logger.warning("预测结果包含NaN值")
            return 0.0, 0.0
        
        # 获取结果
        valence, arousal = outputs[0].cpu().numpy()
        
        return valence, arousal
    
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return 0.0, 0.0

def batch_predict(texts, model, device, batch_size=16):
    """
    批量预测
    
    Args:
        texts: 文本列表
        model: 模型
        device: 设备
        batch_size: 批处理大小
        
    Returns:
        predictions: 预测结果列表 [(valence, arousal), ...]
    """
    predictions = []
    
    # 批量处理
    for i in range(0, len(texts), batch_size):
        try:
            batch_texts = texts[i:i + batch_size]
            
            # 使用模型内置的分词器处理文本
            tokenizer = model.encoder.tokenizer
            encodings = tokenizer.batch_encode(batch_texts)
            
            # 移到设备上
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # 推理
            with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
            
            # 检查输出是否包含NaN
            if torch.isnan(outputs).any():
                logger.warning(f"批次 {i//batch_size + 1} 预测结果包含NaN值")
                batch_predictions = [(0.0, 0.0)] * len(batch_texts)
            else:
                # 获取结果
                batch_predictions = [(valence, arousal) for valence, arousal in outputs.cpu().numpy()]
            
            # 添加到结果列表
            predictions.extend(batch_predictions)
            
            # 打印进度
            logger.info(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本 ({(min(i + batch_size, len(texts)) / len(texts) * 100):.1f}%)")
        
        except Exception as e:
            logger.error(f"批次 {i//batch_size + 1} 预测失败: {e}")
            predictions.extend([(0.0, 0.0)] * len(batch_texts))
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="NCP_LNN+Transformer+LightweightTextEncoder情感分析推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--text", type=str, help="要分析的文本")
    parser.add_argument("--input_file", type=str, help="输入文件（每行一个文本）")
    parser.add_argument("--output_file", type=str, help="批量预测结果的输出文件")
    parser.add_argument("--device", type=str, choices=["mps", "cuda", "cpu"], help="计算设备")
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = Config()
        
        # 更新设备配置
        if args.device:
            config.DEVICE = args.device
        
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        model, device = load_model(args.model_path, config)
        
        # 单文本预测或批量预测
        if args.text:
            # 单文本预测
            logger.info("进行单文本预测...")
            valence, arousal = predict(args.text, model, device)
            logger.info(f"文本: {args.text}")
            logger.info(f"效价 (Valence): {valence:.4f}")
            logger.info(f"唤醒度 (Arousal): {arousal:.4f}")
        
        elif args.input_file:
            # 批量预测
            logger.info(f"从文件 {args.input_file} 进行批量预测...")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"共读取 {len(texts)} 个文本")
            predictions = batch_predict(texts, model, device)
            
            # 输出结果
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write("Text,Valence,Arousal\n")
                    for i, (valence, arousal) in enumerate(predictions):
                        f.write(f'"{texts[i]}",{valence:.4f},{arousal:.4f}\n')
                
                logger.info(f"预测结果已保存到 {args.output_file}")
            else:
                for i, (valence, arousal) in enumerate(predictions):
                    logger.info(f"文本 {i+1}: {texts[i]}")
                    logger.info(f"效价 (Valence): {valence:.4f}")
                    logger.info(f"唤醒度 (Arousal): {arousal:.4f}")
                    logger.info("-" * 50)
        
        else:
            logger.error("请提供文本或输入文件")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"运行失败: {e}")

if __name__ == "__main__":
    main() 