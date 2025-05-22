import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from .ltc import NeuralCircuitLTC
import os

class TeacherModel(nn.Module):
    """
    教师模型：XLM-R-Base + LTC_NCP
    用于预测文本的价效和唤起值
    """
    def __init__(self, base_model_name="xlm-roberta-base", ltc_hidden_size=128, 
                 ltc_memory_size=32, ltc_num_layers=2, ltc_dropout=0.1, output_dim=2):
        super(TeacherModel, self).__init__()
        
        # 加载预训练的XLM-RoBERTa模型
        if os.path.exists("XLM-R"):
            # 从本地目录加载
            self.roberta = XLMRobertaModel.from_pretrained("XLM-R")
        else:
            # 如果本地目录不存在，尝试从Hugging Face下载
            self.roberta = XLMRobertaModel.from_pretrained(base_model_name, cache_dir="XLM-R")
        
        config = self.roberta.config
        
        # LTC层
        self.ltc = NeuralCircuitLTC(
            input_size=config.hidden_size,
            hidden_size=ltc_hidden_size,
            memory_size=ltc_memory_size,
            num_layers=ltc_num_layers,
            dropout=ltc_dropout
        )
        
        # 输出层
        self.output_layer = nn.Linear(ltc_memory_size, output_dim)
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            return_embeddings: 是否返回中间嵌入(用于蒸馏)
            
        Returns:
            logits: 模型输出 [valence, arousal]
            embeddings: (如果return_embeddings=True) 中间嵌入向量
        """
        # 通过BERT获取序列表示
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 通过LTC网络
        ltc_output, _ = self.ltc(sequence_output)  # [batch_size, seq_len, memory_size]
        
        # 取最后一个时间步的输出作为句子表示
        # 或者，使用注意力掩码确定每个序列的实际长度
        seq_lengths = attention_mask.sum(dim=1) - 1  # 减1获取最后一个非填充token的位置
        batch_size = input_ids.size(0)
        
        # 获取每个序列的最后一个实际token的表示
        sentence_repr = ltc_output[torch.arange(batch_size), seq_lengths]  # [batch_size, ltc_hidden_size]
        
        # 输出价效和唤起值
        logits = self.output_layer(sentence_repr)  # [batch_size, 2]
        
        if return_embeddings:
            return logits, sentence_repr
        else:
            return logits
            
    def get_embeddings(self, input_ids, attention_mask):
        """
        获取文本的嵌入表示（用于知识蒸馏）
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            
        Returns:
            embeddings: 嵌入向量
        """
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
            ltc_output, _ = self.ltc(sequence_output)
            
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            
            sentence_repr = ltc_output[torch.arange(batch_size), seq_lengths]
            
            return sentence_repr 