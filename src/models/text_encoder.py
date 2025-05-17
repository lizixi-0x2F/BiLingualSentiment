import torch
import torch.nn as nn
import numpy as np
import os
import re
from collections import Counter

class SimpleTokenizer:
    """简单的自定义分词器"""
    def __init__(self, vocab_size=30000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx2word = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_size_actual = len(self.word2idx)  # 词汇表的实际大小，初始为特殊标记数量
    
    def build_vocab(self, texts):
        """根据文本构建词汇表"""
        # 收集所有单词
        word_counts = {}
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
        
        # 按频率排序
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表，限制大小
        for word, _ in sorted_words:
            if word not in self.word2idx and self.vocab_size_actual < self.vocab_size:
                self.word2idx[word] = self.vocab_size_actual
                self.idx2word[self.vocab_size_actual] = word
                self.vocab_size_actual += 1
        
        print(f"✓ 已构建词汇表，词汇量: {self.vocab_size_actual}")
    
    def _tokenize(self, text):
        """分词"""
        # 确保文本是字符串
        if not isinstance(text, str):
            if text is None:
                text = ""
            else:
                text = str(text)
        
        # 基本清理
        text = text.lower()
        # 分词 - 简单处理
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """将文本编码为ID序列"""
        tokens = self._tokenize(text)
        
        # 添加特殊标记
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 转换为ID
        ids = []
        for token in tokens:
            # 确保返回的token ID不会超出词汇表范围
            token_id = self.word2idx.get(token, self.word2idx['[UNK]'])
            if token_id >= self.vocab_size:
                token_id = self.word2idx['[UNK]']
            ids.append(token_id)
        
        # 截断或填充
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [self.word2idx['[PAD]']] * (self.max_length - len(ids))
        
        return torch.tensor(ids)
    
    def batch_encode(self, texts, add_special_tokens=True):
        """批量编码文本"""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded = self.encode(text, add_special_tokens)
            input_ids.append(encoded)
            
            # 创建注意力掩码
            attention_mask = torch.zeros_like(encoded)
            attention_mask[encoded != self.word2idx['[PAD]']] = 1
            attention_masks.append(attention_mask)
            
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }


class WordEmbedding(nn.Module):
    """词嵌入层，使用随机初始化的嵌入"""
    def __init__(self, vocab_size, embed_dim=300, padding_idx=0, use_pretrained=False):
        super(WordEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        # 增加安全边界
        self.actual_vocab_size = vocab_size + 100
        
        # 创建嵌入层
        self.embedding = nn.Embedding(self.actual_vocab_size, embed_dim, padding_idx=padding_idx)
        
        # 初始化嵌入
        with torch.no_grad():
            # 使用xavier均匀分布初始化
            nn.init.xavier_uniform_(self.embedding.weight)
            # 将padding索引设为零向量
            self.embedding.weight[padding_idx].fill_(0)
        
        print(f"✓ 已初始化词嵌入 (dim={embed_dim})")
            
    def forward(self, input_ids):
        """前向传播"""
        # 确保输入索引在有效范围内
        input_ids = torch.clamp(input_ids, max=self.actual_vocab_size-1)
        return self.embedding(input_ids)


class BiLSTMEncoder(nn.Module):
    """双向LSTM编码器"""
    def __init__(self, embed_dim, hidden_size, num_layers=2, dropout=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出投影层 - 将双向LSTM的输出映射回hidden_size
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, embedded_text, attention_mask=None):
        """
        前向传播
        
        Args:
            embedded_text: 嵌入后的文本 [batch_size, seq_len, embed_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            output: LSTM输出序列 [batch_size, seq_len, hidden_size]
        """
        # 将padding设置为零
        if attention_mask is not None:
            embedded_text = embedded_text * attention_mask.unsqueeze(-1)
        
        # 通过LSTM
        output, _ = self.lstm(embedded_text)  # [batch_size, seq_len, hidden_size*2]
        
        # 投影回原始维度
        output = self.proj(output)  # [batch_size, seq_len, hidden_size]
        
        return output


class CompressedTextEncoder(nn.Module):
    """轻量级文本编码器，更高效的压缩表示"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, compression_rate=0.25, dropout=0.2):
        super(CompressedTextEncoder, self).__init__()
        
        # 确保词汇表大小足够大，留出安全边界
        self.actual_vocab_size = vocab_size + 100  # 增加安全边界
        
        # 嵌入层
        self.embedding = nn.Embedding(self.actual_vocab_size, embedding_dim, padding_idx=0)
        
        # 压缩层 - 使用固定的压缩维度
        compressed_dim = 128  # 固定压缩维度为128
        self.compressor = nn.Linear(embedding_dim, compressed_dim)
        
        # 编码层
        self.encoder = nn.LSTM(
            input_size=compressed_dim,
            hidden_size=hidden_size // 2,  # 因为是双向LSTM，输出会是hidden_size
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)  # padding_idx=0
    
    def forward(self, x, mask=None):
        """前向传播"""
        # 确保输入索引在有效范围内
        x = torch.clamp(x, max=self.actual_vocab_size-1)
        
        # 应用嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 压缩表示
        compressed = self.compressor(embedded)  # [batch_size, seq_len, compressed_dim]
        
        # 如果有掩码，将padding位置的嵌入设为0
        if mask is not None:
            compressed = compressed * mask.unsqueeze(-1)
        
        # 编码
        outputs, _ = self.encoder(compressed)  # [batch_size, seq_len, hidden_size]
        
        # 输出投影
        outputs = self.output_projection(outputs)  # [batch_size, seq_len, hidden_size]
        
        return outputs


class LightweightTextEncoder(nn.Module):
    """轻量级文本编码器 - 使用压缩技术"""
    def __init__(self, config):
        super(LightweightTextEncoder, self).__init__()
        
        # 分词器
        self.tokenizer = SimpleTokenizer(vocab_size=config.VOCAB_SIZE)
        
        # 文本编码器
        self.text_encoder = CompressedTextEncoder(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            compression_rate=config.COMPRESSION_RATE,  # 兼容性参数，实际不再使用
            dropout=config.DROPOUT
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """前向传播"""
        # 编码文本
        encoded_text = self.text_encoder(input_ids, attention_mask)
        
        # 创建类似于Transformers库的输出结构
        class EncoderOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        
        return EncoderOutput(encoded_text) 