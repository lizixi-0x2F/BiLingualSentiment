import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from .ltc import NeuralCircuitLTC

class MiniTransformerConfig(PretrainedConfig):
    """Mini Transformer配置类"""
    
    model_type = "mini_transformer"
    
    def __init__(
        self,
        vocab_size=250002,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps

class MiniTransformerAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 查询、键、值投影层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 输出投影
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        """重塑张量以便进行多头注意力计算"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # 线性投影
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # 重塑为多头形式
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # 注意力分数计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 调整掩码维度以匹配注意力分数
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # 注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权聚合value
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 重塑回原始维度
        batch_size, seq_len, _, _ = context_layer.size()
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        # 最终投影
        output = self.output_proj(context_layer)
        
        return output

class MiniTransformerLayer(nn.Module):
    """Mini Transformer层"""
    
    def __init__(self, config):
        super().__init__()
        
        # 自注意力
        self.attention = MiniTransformerAttention(config)
        
        # 前馈网络
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # LayerNorm
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        
        # 第一个残差连接
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # 第二个残差连接
        layer_output = self.output_layer_norm(attention_output + layer_output)
        
        return layer_output

class MiniTransformer(nn.Module):
    """迷你transformer编码器"""
    
    def __init__(self, config):
        super().__init__()
        
        # 嵌入层
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # LayerNorm和Dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer层
        self.layers = nn.ModuleList([
            MiniTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 初始化参数
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        # 获取输入形状
        batch_size, seq_length = input_ids.shape
        
        # 创建位置IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 创建嵌入
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 合并嵌入
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 通过Transformer层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

class StudentModel(nn.Module):
    """
    学生模型：Mini Transformer + LTC_NCP
    具有更小的参数量和更快的推理速度
    """
    def __init__(self, vocab_size=250002, hidden_size=384, num_hidden_layers=6, 
                 num_attention_heads=6, intermediate_size=1536, 
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 ltc_hidden_size=128, ltc_memory_size=32, ltc_num_layers=2, 
                 ltc_dropout=0.1, output_dim=2):
        super(StudentModel, self).__init__()
        
        # 创建配置
        config = MiniTransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        
        # Mini Transformer编码器
        self.transformer = MiniTransformer(config)
        
        # LTC层
        self.ltc = NeuralCircuitLTC(
            input_size=hidden_size,
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
            return_embeddings: 是否返回中间嵌入
            
        Returns:
            logits: 模型输出 [valence, arousal]
            embeddings: (如果return_embeddings=True) 中间嵌入向量
        """
        # 通过Transformer获取序列表示
        sequence_output = self.transformer(input_ids, attention_mask)  # [batch_size, seq_len, hidden_size]
        
        # 通过LTC网络
        ltc_output, _ = self.ltc(sequence_output)  # [batch_size, seq_len, memory_size]
        
        # 使用注意力掩码确定每个序列的实际长度
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