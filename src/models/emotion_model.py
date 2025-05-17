import torch
import torch.nn as nn
from .text_encoder import LightweightTextEncoder

class NCP_LNN(nn.Module):
    """
    神经回路策略(NCP)液体神经网络(LNN)模块
    将离散神经元类型与连续时间动力学相结合
    """
    def __init__(self, hidden_size, tau_min=1.0, tau_max=20.0, dt=0.5, tau_regularizer=1e-4, sparsity=0.15):
        super(NCP_LNN, self).__init__()
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = dt
        self.tau_regularizer = tau_regularizer
        self.sparsity = sparsity  # 使用15%的稀疏度而非默认的70%
        
        # 输入投影
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        
        # 内部动态权重
        self.recurrent_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # 连接掩码 - 实现稀疏连接
        self.connectivity_mask = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=False)
        
        # 时间常数
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        
        # 激活函数
        self.activation = nn.Tanh()
        
        # 输出层标准化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化网络参数"""
        # 输入投影的初始化
        nn.init.kaiming_uniform_(self.input_projection.weight, a=0.1)
        nn.init.zeros_(self.input_projection.bias)
        
        # 内部权重初始化 - Xavier×2使前几十step就有效信号
        nn.init.xavier_normal_(self.recurrent_weights, gain=2.0)
        
        # 初始化连接掩码 - 只保留sparsity比例的连接
        with torch.no_grad():
            mask = torch.rand_like(self.recurrent_weights) < self.sparsity
            self.connectivity_mask.copy_(mask.float())
            # 应用掩码到权重，使得只有部分连接存在
            self.recurrent_weights.data *= self.connectivity_mask
        
        # 时间常数初始化
        with torch.no_grad():
            # 初始化为对数均匀分布
            log_tau = torch.rand(self.hidden_size) * (torch.log(torch.tensor(self.tau_max)) - 
                                                       torch.log(torch.tensor(self.tau_min))) + torch.log(torch.tensor(self.tau_min))
            self.tau.copy_(torch.exp(log_tau))
        
    def forward(self, x, state=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, hidden_size]
            state: 初始状态 [batch_size, hidden_size]
            
        Returns:
            outputs: 处理后的序列 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        outputs = []
        
        # 初始化状态
        if state is None:
            state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 序列处理
        for t in range(seq_len):
            # 当前输入
            current_input = x[:, t, :]
            
            # 输入投影
            input_projection = self.input_projection(current_input)
            
            # 递归计算 - 应用连接掩码确保稀疏连接
            masked_weights = self.recurrent_weights * self.connectivity_mask
            recurrent_input = torch.matmul(state, masked_weights.t())
            
            # 组合输入
            combined = input_projection + recurrent_input
            
            # 目标激活
            target = self.activation(combined)
            
            # 应用时间常数
            tau = torch.clamp(self.tau, min=self.tau_min, max=self.tau_max)
            alpha = (self.dt / tau).unsqueeze(0).expand(batch_size, -1)
            alpha = torch.clamp(alpha, min=0.0, max=1.0)
        
            # 状态更新
            state = state + alpha * (target - state)
            state = torch.clamp(state, -1.0, 1.0)  # 稳定性裁剪
            
            # 输出收集
            outputs.append(state)
        
        # 堆叠输出
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # 应用层标准化
        outputs = self.layer_norm(outputs)
        
        return outputs
    
    def get_tau_regularization(self):
        """计算时间常数正则化损失"""
        tau = torch.clamp(self.tau, min=self.tau_min, max=self.tau_max)
        tau_reg = torch.mean(torch.square(tau))
        return self.tau_regularizer * tau_reg

class NCPAttention(nn.Module):
    """基于NCP的注意力机制"""
    def __init__(self, hidden_size, num_heads, excitatory_ratio=0.6, dropout=0.1):
        super(NCPAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.excitatory_ratio = excitatory_ratio
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # NCP特有的激励/抑制神经元分离 (调整比例为0.6)
        num_excitatory = int(hidden_size * excitatory_ratio)
        self.neuron_type = torch.ones(hidden_size)
        self.neuron_type[num_excitatory:] = -1.0  # 后面的神经元为抑制性
        self.neuron_type = nn.Parameter(self.neuron_type, requires_grad=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 调整维度顺序
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 掩码处理
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 使用NCP特性调整注意力
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # NCP特性：激励/抑制分离，注意neuron_type需要适当调整维度来匹配attn_output
        neuron_type_expanded = self.neuron_type.view(1, 1, -1).expand(batch_size, seq_len, -1)
        attn_output = attn_output * neuron_type_expanded
        
        output = self.output_projection(attn_output)
        
        return output

class NCPLTCLayer(nn.Module):
    """NCP+LNN层，集成NCP注意力和LNN动态"""
    def __init__(self, config):
        super(NCPLTCLayer, self).__init__()
        hidden_size = config.TRANSFORMER_HIDDEN_SIZE
        
        # NCP注意力
        self.attention = NCPAttention(
            hidden_size=hidden_size,
            num_heads=config.NUM_HEADS,
            excitatory_ratio=0.6,  # 显式设置为0.6
            dropout=config.DROPOUT
        )
        
        # NCP_LNN模块，替代原来的LTC模块
        self.ltc = NCP_LNN(
            hidden_size=hidden_size,
            tau_min=config.TAU_MIN,
            tau_max=config.TAU_MAX,
            dt=0.5,  # 调整dt值以增强稳定性
            tau_regularizer=1e-4,  # 添加时间常数正则化系数
            sparsity=config.SPARSITY  # 从配置获取稀疏度
        )
        
        # 层标准化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 增加额外的注意力后dropout
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        
        # 增强正则化的前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(config.DROPOUT)
        )
        
    def forward(self, x, mask=None, state=None):
        # 自注意力
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, mask)
        # 应用注意力后的dropout
        attn_out = self.attn_dropout(attn_out)
        
        # 添加残差连接，但要防止梯度爆炸
        x = residual + torch.clamp(attn_out, -1.0, 1.0)
        
        # LNN处理
        residual = x
        x = self.norm2(x)
        ltc_out = self.ltc(x, state)  # 使用NCP_LNN替代LTC
        ff_out = self.feed_forward(ltc_out)
        
        # 再次添加残差连接，同样防止梯度爆炸
        x = residual + torch.clamp(ff_out, -1.0, 1.0)
        
        return x
    
    def get_tau_regularization(self):
        """获取时间常数正则化损失"""
        return self.ltc.get_tau_regularization()

class EmotionAnalysisModel(nn.Module):
    """更新后的情感分析模型，使用NCP_LNN+Transformer+LightweightTextEncoder架构"""
    def __init__(self, config):
        super(EmotionAnalysisModel, self).__init__()
        self.config = config  # 保存config引用以便后续访问LAMBDA_WEIGHT等参数
        
        # 使用轻量级文本编码器
        self.encoder = LightweightTextEncoder(config)
        
        # 输入映射
        self.input_mapping = nn.Linear(config.HIDDEN_SIZE, config.TRANSFORMER_HIDDEN_SIZE)
        
        # 使用NCP_LTC层替代原来的NCP_LNN层
        self.ncp_ltc = NCPLTCLayer(config)
        
        # 池化层
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 增强正则化的输出层
        self.output = nn.Sequential(
            nn.Linear(config.TRANSFORMER_HIDDEN_SIZE, config.TRANSFORMER_HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),  # 使用配置的dropout值
            nn.Linear(config.TRANSFORMER_HIDDEN_SIZE // 2, config.TRANSFORMER_HIDDEN_SIZE // 4),  # 新增的中间层
            nn.GELU(),
            nn.Dropout(config.DROPOUT),  # 新增的dropout层
            nn.Linear(config.TRANSFORMER_HIDDEN_SIZE // 4, config.OUTPUT_DIM),
            nn.Tanh()  # 确保输出范围在[-1, 1]之间
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: 标记类型ID [batch_size, seq_len]
        Returns:
            outputs: [batch_size, output_dim] (valence, arousal)
        """
        # 使用轻量级编码器处理文本
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 映射到适当的维度
        x = self.input_mapping(sequence_output)
        
        # 通过NCP_LTC层
        x = self.ncp_ltc(x, attention_mask)
        
        # 池化
        x = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        x = self.pool(x)  # [batch_size, hidden_size]
        
        # 输出层
        outputs = self.output(x)
        
        return outputs
    
    def get_transformer_weights(self):
        """兼容性函数，返回模型权重"""
        return self.state_dict()
    
    def get_tau_regularization(self):
        """获取时间常数正则化损失"""
        return self.ncp_ltc.get_tau_regularization() 