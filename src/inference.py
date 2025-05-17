import torch
import argparse
import os
import sys
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel

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
        tokenizer: 模型的tokenizer
    """
    try:
        # 设置设备
        device = torch.device(config.DEVICE if torch.cuda.is_available() or hasattr(torch.backends, "mps") 
                              and torch.backends.mps.is_available() else "cpu")
        
        # 根据模型类型创建模型
        if config.MODEL_TYPE == 'distilbert':
            model = MultilingualDistilBERTModel(config).to(device)
        elif config.MODEL_TYPE == 'xlm-roberta':
            model = XLMRobertaDistilledModel(config).to(device)
        else:
            raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")
        
        # 加载预训练模型的tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.MULTILINGUAL_MODEL_NAME)
        
        # 加载模型参数
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 显示模型信息
        logger.info(f"模型加载成功，训练轮次: {checkpoint.get('epoch', 'unknown')}")
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            logger.info(f"验证集指标: RMSE={metrics.get('rmse', 'N/A'):.4f}, R2={metrics.get('r2', 'N/A'):.4f}")
        
        # 设置为评估模式
        model.eval()
        
        return model, tokenizer, device
    
    except Exception as e:
        logger.error(f"加载模型出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

def predict_emotion(model, tokenizer, texts, device, batch_size=16):
    """
    预测文本的情感效价和唤醒度
    
    Args:
        model: 模型
        tokenizer: tokenizer
        texts: 文本列表
        device: 设备
        batch_size: 批次大小
        
    Returns:
        predictions: numpy数组形状为 [len(texts), 2]，包含效价和唤醒度预测
    """
    # 确保输入是列表
    if isinstance(texts, str):
        texts = [texts]
    
    # 初始化存储结果的列表
    predictions = []
    
    # 分批处理文本
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 编码文本
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        ).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(**encodings)
        
        # 将预测结果添加到列表中
        predictions.append(outputs.cpu().numpy())
    
    # 合并所有预测结果
    predictions = np.vstack(predictions)
    
    return predictions

def main():
    """主函数，处理预测流程"""
    parser = argparse.ArgumentParser(description='使用预训练模型进行情感预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, default='distilbert', choices=['distilbert', 'xlm-roberta'], help='模型类型')
    parser.add_argument('--input_file', type=str, help='输入文件（CSV/TXT）')
    parser.add_argument('--output_file', type=str, help='输出文件（CSV）')
    parser.add_argument('--text_column', type=str, default='text', help='文本列名（针对CSV输入）')
    parser.add_argument('--device', type=str, default='cuda', help='设备(cuda/cpu/mps)')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--text', type=str, help='直接指定要分析的文本')
    
    args = parser.parse_args()
    
    # 确保模型路径存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA不可用，切换到CPU')
        args.device = 'cpu'
    elif args.device == 'mps' and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning('MPS不可用，切换到CPU')
        args.device = 'cpu'
    
    # 创建配置
    config = Config()
    config.DEVICE = args.device
    config.MODEL_TYPE = args.model_type
    
    # 设置多语言模型名称
    if args.model_type == 'distilbert':
        config.MULTILINGUAL_MODEL_NAME = 'distilbert-base-multilingual-cased'
    elif args.model_type == 'xlm-roberta':
        config.MULTILINGUAL_MODEL_NAME = 'xlm-roberta-base'
    
    # 加载模型
    logger.info("加载模型...")
    model, tokenizer, device = load_model(args.model_path, config)
    
    # 确定输入方式
    if args.text:
        # 直接使用命令行参数中的文本
        texts = [args.text]
    elif args.input_file:
        # 从文件读取文本
        file_ext = os.path.splitext(args.input_file)[1].lower()
        
        if file_ext == '.csv':
            # 从CSV文件读取
            df = pd.read_csv(args.input_file)
            if args.text_column not in df.columns:
                logger.error(f"CSV文件中找不到列 '{args.text_column}'")
                return
            texts = df[args.text_column].tolist()
        elif file_ext == '.txt':
            # 从TXT文件读取，每行一个文本
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"不支持的文件格式: {file_ext}")
            return
    else:
        # 使用样例文本
        try:
            with open('example_texts.txt', 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            logger.info("使用example_texts.txt中的样例文本")
        except:
            texts = [
                "我今天很开心，天气很好。",
                "这部电影让我很失望，情节太差了。",
                "I'm feeling great today, the weather is nice.",
                "This movie was disappointing, the plot was terrible."
            ]
            logger.info("使用默认样例文本")
    
    # 进行预测
    logger.info(f"开始预测{len(texts)}个文本的情感...")
    predictions = predict_emotion(model, tokenizer, texts, device, args.batch_size)
    
    # 显示结果
    logger.info("预测结果:")
    for i, (text, pred) in enumerate(zip(texts[:min(5, len(texts))], predictions[:min(5, len(predictions))])):
        short_text = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"{i+1}. 文本: {short_text}")
        logger.info(f"   效价(Valence): {pred[0]:.4f}, 唤醒度(Arousal): {pred[1]:.4f}")
    
    if len(texts) > 5:
        logger.info(f"... 等共{len(texts)}个文本")
    
    # 保存结果
    if args.output_file:
        try:
            results_df = pd.DataFrame({
                'text': texts,
                'valence': predictions[:, 0],
                'arousal': predictions[:, 1]
            })
            results_df.to_csv(args.output_file, index=False)
            logger.info(f"结果已保存到 {args.output_file}")
        except Exception as e:
            logger.error(f"保存结果出错: {e}")
    
if __name__ == "__main__":
    main()
