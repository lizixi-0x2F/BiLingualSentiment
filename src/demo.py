import os
import argparse
import torch
from inference import SentimentPredictor

def main():
    """双语情感价效和唤起演示程序"""
    
    parser = argparse.ArgumentParser(description="双语情感价效和唤起演示")
    parser.add_argument("--model_path", type=str, default="checkpoints/student/best_model.pt", 
                        help="模型权重路径 (默认为学生模型)")
    parser.add_argument("--model_type", type=str, default="student", choices=["student", "teacher"],
                        help="模型类型 (默认为学生模型)")
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        print("请先训练模型或提供正确的模型路径。")
        return
    
    # 创建预测器
    try:
        predictor = SentimentPredictor(model_path=args.model_path, model_type=args.model_type)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("欢迎使用双语情感价效和唤起分析系统")
    print("这个系统可以分析文本的情感价效(Valence)和唤起(Arousal)程度")
    print("价效值范围: -1(消极) 到 1(积极)")
    print("唤起值范围: -1(低唤起) 到 1(高唤起)")
    print("="*60)
    print("\n输入 'q' 或 'exit' 退出程序\n")
    
    # 示例文本
    examples = [
        "我今天非常开心，因为我收到了一个惊喜礼物！",
        "I am feeling extremely happy today because I received a surprise gift!",
        "这个消息让我很难过，我需要一些时间来接受。",
        "This news makes me very sad, I need some time to accept it.",
        "我对这个话题没有特别的感觉，挺中性的。",
        "I don't have particular feelings about this topic, it's quite neutral."
    ]
    
    print("示例文本:")
    for i, example in enumerate(examples):
        print(f"{i+1}. {example}")
    print("\n选择一个示例(输入编号)或直接输入您的文本:")
    
    # 交互循环
    while True:
        user_input = input("\n> ").strip()
        
        # 检查退出命令
        if user_input.lower() in ['q', 'exit', 'quit']:
            print("感谢使用！再见！")
            break
            
        # 检查是否是示例编号
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            text = examples[int(user_input) - 1]
            print(f"选择的示例: {text}")
        else:
            text = user_input
        
        # 分析文本
        try:
            result = predictor.analyze(text)
            
            # 美化输出
            print("\n" + "-"*60)
            print(f"文本: {result['text']}")
            print(f"价效值: {result['valence']:.3f} ({'-1=消极' if result['valence'] < 0 else '0=中性' if abs(result['valence']) < 0.3 else '1=积极'})")
            print(f"唤起值: {result['arousal']:.3f} ({'-1=低唤起' if result['arousal'] < 0 else '0=中等' if abs(result['arousal']) < 0.3 else '1=高唤起'})")
            print(f"情感倾向: {result['emotion']}")
            print("-"*60)
            
        except Exception as e:
            print(f"分析时出错: {e}")

if __name__ == "__main__":
    main() 