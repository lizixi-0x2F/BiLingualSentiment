import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultilingualDistilBERTModel(nn.Module):
    """
    基于多语言DistilBERT的情感分析模型
    适用于中英文双语情感分析，预测效价(Valence)和唤醒度(Arousal)
    """
    def __init__(self, config):
        super(MultilingualDistilBERTModel, self).__init__()
        self.config = config
        
        # 使用多语言DistilBERT模型
        model_name = config.MULTILINGUAL_MODEL_NAME  # 'distilbert-base-multilingual-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 获取模型隐藏层大小
        self.hidden_size = self.model.config.hidden_size  # 通常为768
        
        # 用于序列池化的层（可选用不同方式）
        self.pooling_type = config.POOLING_TYPE if hasattr(config, 'POOLING_TYPE') else 'cls'
        
        # 回归输出层 - 使用增强正则化和tanh激活确保输出范围在[-1,1]
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_size // 4, config.OUTPUT_DIM),
            nn.Tanh()  # 确保输出在[-1,1]范围
        )
        
        # 冻结某些层以提高训练效率（可选）
        if hasattr(config, 'FREEZE_LAYERS') and config.FREEZE_LAYERS > 0:
            self._freeze_layers(config.FREEZE_LAYERS)
    
    def _freeze_layers(self, num_layers):
        """冻结底层以提高训练效率和防止过拟合"""
        # 冻结embedding层
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
            
        # 冻结指定数量的transformer层
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.layer
        elif hasattr(self.model, 'encoder'):
            layers = self.model.encoder.layer
        else:
            return
            
        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: 标记类型ID（对于某些模型需要）
            
        Returns:
            outputs: [batch_size, output_dim] (valence, arousal)
        """
        # 检查模型是否需要token_type_ids
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if token_type_ids is not None and 'bert' in self.config.MULTILINGUAL_MODEL_NAME.lower():
            model_inputs['token_type_ids'] = token_type_ids
        
        # 获取模型输出
        outputs = self.model(**model_inputs)
        
        # 根据池化类型选择表示
        if self.pooling_type == 'cls':
            # 使用[CLS]标记的表示
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling_type == 'mean':
            # 使用平均池化
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            # 默认使用[CLS]
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # 通过输出层获得效价和唤醒度预测
        predictions = self.output(pooled_output)
        
        return predictions
    
    def get_tokenizer(self):
        """返回模型的tokenizer，用于数据处理"""
        return self.tokenizer
    
    def get_transformer_weights(self):
        """兼容性函数，返回模型权重"""
        return self.state_dict()
    
    def get_tau_regularization(self):
        """兼容性函数，提供空的时间常数正则化损失"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


class XLMRobertaDistilledModel(nn.Module):
    """
    基于XLM-RoBERTa-distilled的情感分析模型
    适用于中英文双语情感分析，预测效价(Valence)和唤醒度(Arousal)
    """
    def __init__(self, config):
        super(XLMRobertaDistilledModel, self).__init__()
        self.config = config
        
        # 使用XLM-RoBERTa模型
        model_name = config.MULTILINGUAL_MODEL_NAME  # 例如 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 获取模型隐藏层大小
        self.hidden_size = self.model.config.hidden_size  # 通常为768
        
        # 池化配置
        self.pooling_type = config.POOLING_TYPE if hasattr(config, 'POOLING_TYPE') else 'cls'
        
        # 回归输出层 - 使用与原模型一致的结构以确保可比性
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_size // 4, config.OUTPUT_DIM),
            nn.Tanh()  # 确保输出在[-1,1]范围
        )
        
        # 可选冻结层
        if hasattr(config, 'FREEZE_LAYERS') and config.FREEZE_LAYERS > 0:
            self._freeze_layers(config.FREEZE_LAYERS)
    
    def _freeze_layers(self, num_layers):
        """冻结底层以提高训练效率和防止过拟合"""
        # 冻结embeddings
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
            
        # 冻结指定数量的Transformer层
        if hasattr(self.model, 'encoder'):
            for i in range(min(num_layers, len(self.model.encoder.layer))):
                for param in self.model.encoder.layer[i].parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """前向传播"""
        # XLM-RoBERTa不使用token_type_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 根据池化类型选择表示
        if self.pooling_type == 'cls':
            # 使用第一个token的表示作为序列表示
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling_type == 'mean':
            # 使用平均池化
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            # 默认使用第一个token
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # 通过输出层获得效价和唤醒度预测
        predictions = self.output(pooled_output)
        
        return predictions
    
    def get_tokenizer(self):
        """返回模型的tokenizer，用于数据处理"""
        return self.tokenizer
    
    def get_transformer_weights(self):
        """兼容性函数，返回模型权重"""
        return self.state_dict()
    
    def get_tau_regularization(self):
        """兼容性函数，提供空的时间常数正则化损失"""
        return torch.tensor(0.0, device=next(self.parameters()).device)
