import torch
import json
import argparse
from transformers import XLMRobertaTokenizer
from src.models.student_model import StudentModel
from src.models.teacher_model import TeacherModel

class SentimentPredictor:
    """情感价效和唤起预测器"""
    
    def __init__(self, model_path, model_type="student", device=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            model_type: 模型类型 ('student' 或 'teacher')
            device: 计算设备
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() 
                                     else "mps" if torch.backends.mps.is_available() 
                                     else "cpu")
        else:
            self.device = device
            
        # 加载分词器
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("XLM-R")
        
        # 加载模型配置和权重
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = state["config"]
        
        # 创建模型
        if model_type == "student":
            self.model = self._create_student_model()
        elif model_type == "teacher":
            self.model = self._create_teacher_model()
        else:
            raise ValueError("model_type 必须是 'student' 或 'teacher'")
        
        # 加载模型权重
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载到 {self.device} 设备")
        
    def _create_student_model(self):
        """创建学生模型实例"""
        model_config = self.config["model"]
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
        return model
    
    def _create_teacher_model(self):
        """创建教师模型实例"""
        model_config = self.config["model"]
        model = TeacherModel(
            base_model_name=model_config["base_model_name"],
            ltc_hidden_size=model_config["ltc_hidden_size"],
            ltc_memory_size=model_config.get("ltc_memory_size", model_config.get("ltc_hidden_size", 32) // 4),  # 兼容旧配置
            ltc_num_layers=model_config["ltc_num_layers"],
            ltc_dropout=model_config["ltc_dropout"],
            output_dim=model_config["output_dim"]
        )
        return model
    
    def predict(self, text, return_dict=True):
        """
        预测文本的价效和唤起值
        
        Args:
            text: 输入文本
            return_dict: 是否返回字典形式的结果
            
        Returns:
            预测结果，字典或元组
        """
        # 对输入文本进行分词
        encoding = self.tokenizer(
            text,
            max_length=self.config["data"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 将编码移动到设备
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # 进行推理
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
        # 获取价效和唤起的预测值
        valence = logits[0, 0].item()
        arousal = logits[0, 1].item()
        
        # 返回结果
        if return_dict:
            return {
                "text": text,
                "valence": valence,
                "arousal": arousal
            }
        else:
            return (valence, arousal)
    
    def predict_batch(self, texts):
        """
        批量预测多个文本的价效和唤起值
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果字典列表
        """
        results = []
        
        # 对输入文本进行分词
        encoding = self.tokenizer(
            texts,
            max_length=self.config["data"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 将编码移动到设备
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # 进行推理
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
        # 获取价效和唤起的预测值
        valences = logits[:, 0].cpu().numpy()
        arousals = logits[:, 1].cpu().numpy()
        
        # 构建结果
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "valence": float(valences[i]),
                "arousal": float(arousals[i])
            })
        
        return results
    

    def analyze(self, text):
        """
        分析文本的情感，只返回价效值和唤起值
        
        Args:
            text: 输入文本
            
        Returns:
            分析结果字典
        """
        # 获取预测值
        prediction = self.predict(text)
        valence = prediction["valence"]
        arousal = prediction["arousal"]
        
        # 构建结果
        result = {
            "text": text,
            "valence": valence,
            "arousal": arousal,
            "analysis": f"文本情感分析: 价效值={valence:.3f}, 唤起值={arousal:.3f}"
        }
        
        return result

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="情感价效和唤起预测")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--model_type", type=str, default="student", choices=["student", "teacher"], help="模型类型")
    parser.add_argument("--text", type=str, help="要分析的文本")
    parser.add_argument("--batch_file", type=str, help="批量预测的文本文件，每行一个文本")
    parser.add_argument("--output", type=str, help="输出结果的文件路径")
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = SentimentPredictor(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # 单个文本预测
    if args.text:
        result = predictor.analyze(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 批量预测
    elif args.batch_file:
        with open(args.batch_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = []
        for text in texts:
            result = predictor.analyze(text)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 