import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LTCCell(nn.Module):
    """
    液态时间常数 (Liquid Time-Constant, LTC) 神经元单元
    """
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入权重
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # 时间常数参数
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        
        # 偏置
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.tau, 1.0)
        
    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, device=input.device)
            
        # 计算输入投影
        i_t = F.linear(input, self.weight_ih, self.bias)
        
        # 确保tau是正的并限制在合理范围内
        tau = torch.clamp(F.softplus(self.tau), min=0.1, max=10.0)
        
        # 计算动态时间常数
        decay = torch.exp(-1.0 / tau)
        
        # 更新隐藏状态 (神经网络中的液态行为)
        h_new = decay * hx + (1 - decay) * torch.tanh(i_t)
        
        return h_new

class SimpleLTCCell(nn.Module):
    """
    简化版液态时间常数神经元单元，避免梯度计算问题
    """
    def __init__(self, input_size, hidden_size, activation='tanh'):
        super(SimpleLTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 使用标准Linear层代替自定义权重
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 时间常数参数 - 每个神经元一个tau参数
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # 计算衰减因子时的缩放参数
        self.tau_scale = nn.Parameter(torch.tensor(1.0))
        
        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = torch.tanh
            
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.constant_(self.tau, 1.0)
        
    def forward(self, x, h=None):
        """
        前向传播
        
        Args:
            x: 输入向量 [batch_size, input_size]
            h: 隐藏状态 [batch_size, hidden_size]
            
        Returns:
            h_new: 新的隐藏状态 [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        # 如果没有提供隐藏状态，初始化为0
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 输入投影
        x_proj = self.input_proj(x)
        
        # 计算衰减系数 (确保tau始终为正)
        tau = torch.clamp(F.softplus(self.tau), min=0.1, max=10.0)
        # 添加缩放因子增强灵活性
        tau = tau * torch.clamp(F.softplus(self.tau_scale), min=0.5, max=5.0)
        decay = torch.exp(-1.0 / tau)
        
        # 使用torch.lerp避免就地操作
        # lerp公式: h_new = decay * h + (1 - decay) * activation(x_proj)
        h_new = torch.lerp(self.activation(x_proj), h, decay)
        
        return h_new

class EnhancedLTCCell(nn.Module):
    """
    增强版液态时间常数单元，添加多种门控机制
    """
    def __init__(self, input_size, hidden_size, use_gating=True):
        super(EnhancedLTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_gating = use_gating
        
        # 主投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 时间常数参数
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # 门控机制
        if use_gating:
            # 更新门，类似于GRU
            self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
            # 重置门
            self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
            
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.tau, 1.0)
    
    def forward(self, x, h=None):
        """增强的前向传播"""
        batch_size = x.size(0)
        
        # 如果没有提供隐藏状态，初始化为0
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 基本输入投影
        x_proj = self.input_proj(x)
        
        if self.use_gating:
            # 合并输入和隐藏状态用于门控
            combined = torch.cat([x, h], dim=1)
            
            # 计算更新门
            update_gate = torch.sigmoid(self.update_gate(combined))
            
            # 计算重置门
            reset_gate = torch.sigmoid(self.reset_gate(combined))
            
            # 应用重置门到隐藏状态
            h_reset = h * reset_gate
            
            # 计算候选隐藏状态
            h_candidate = torch.tanh(x_proj)
            
            # 计算tau和衰减因子，受更新门影响
            tau = torch.clamp(F.softplus(self.tau), min=0.1, max=10.0)
            decay = torch.exp(-1.0 / tau) * update_gate
            
            # 生成新的隐藏状态
            h_new = decay * h_reset + (1 - decay) * h_candidate
        else:
            # 基本LTC更新
            tau = torch.clamp(F.softplus(self.tau), min=0.1, max=10.0)
            decay = torch.exp(-1.0 / tau)
            h_new = torch.lerp(torch.tanh(x_proj), h, decay)
        
        return h_new

class NeuralCircuitLTC(nn.Module):
    """
    神经回路机制的液态时间常数网络
    实现分层连接和反馈机制
    """
    def __init__(self, input_size, hidden_size, memory_size=None, num_layers=2, dropout=0.1,
                 use_enhanced_cell=True, use_skip_connections=True, use_layer_norm=True):
        super(NeuralCircuitLTC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size if memory_size is not None else hidden_size // 4
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        self.use_layer_norm = use_layer_norm
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LTC层 - 可选择使用增强版LTC单元
        if use_enhanced_cell:
            self.ltc_cells = nn.ModuleList([
                EnhancedLTCCell(hidden_size, self.memory_size, use_gating=True)
                for _ in range(num_layers)
            ])
        else:
            self.ltc_cells = nn.ModuleList([
                SimpleLTCCell(hidden_size, self.memory_size)
                for _ in range(num_layers)
            ])
        
        # 层间连接 - 神经回路机制的核心
        self.layer_connections = nn.ModuleList([
            nn.Linear(self.memory_size, hidden_size)
            for _ in range(num_layers - 1)
        ])
        
        # 跳跃连接投影层
        if use_skip_connections and num_layers > 1:
            self.skip_connections = nn.ModuleList([
                nn.Linear(self.memory_size, hidden_size)
                for _ in range(num_layers - 1)
            ])
        
        # 反馈连接 - 允许高层信息回流到低层
        if num_layers > 1:
            self.feedback_connections = nn.ModuleList([
                nn.Linear(self.memory_size, hidden_size)
                for _ in range(num_layers - 1)
            ])
        
        # 层归一化
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size)
                for _ in range(num_layers)
            ])
            self.memory_norms = nn.ModuleList([
                nn.LayerNorm(self.memory_size)
                for _ in range(num_layers)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            hidden: 初始隐藏状态 [num_layers, batch_size, memory_size]
            
        Returns:
            outputs: 输出序列 [batch_size, seq_len, memory_size]
            hidden: 最终隐藏状态 [num_layers, batch_size, memory_size]
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.memory_size, device=x.device)
        
        # 创建新隐藏状态列表，避免就地修改
        new_hidden = []
        for i in range(self.num_layers):
            new_hidden.append(hidden[i].clone() if hidden is not None else 
                            torch.zeros(batch_size, self.memory_size, device=x.device))
        
        # 保存所有时间步的输出
        outputs = []
        
        # 保存每层每个时间步的状态，用于反馈连接
        layer_states = [[] for _ in range(self.num_layers)]
        
        # 处理每个时间步
        for t in range(seq_len):
            # 获取当前时间步输入
            x_t = x[:, t, :]
            
            # 投影到隐藏维度
            x_t = self.input_proj(x_t)
            
            # 通过每一层LTC处理
            layer_inputs = [x_t]  # 保存每层的输入，用于跳跃连接
            
            for i in range(self.num_layers):
                # 获取当前层的隐藏状态
                h_i = new_hidden[i]
                
                # 准备输入 - 基本输入
                layer_input = layer_inputs[-1]
                
                # 如果不是第一层，添加来自上一层的连接
                if i > 0:
                    # 投影上一层的输出并添加到当前输入
                    prev_h = new_hidden[i-1]
                    conn = self.layer_connections[i-1](prev_h)
                    layer_input = layer_input + conn
                
                # 反馈连接 - 从较高层到当前层
                # 只在t>0时应用，因为需要前一时间步的状态
                if t > 0 and i < self.num_layers - 1:
                    for j in range(i+1, self.num_layers):
                        # 获取高层在前一时间步的状态
                        if len(layer_states[j]) > 0:
                            higher_state = layer_states[j][-1]
                            feedback = self.feedback_connections[i](higher_state)
                            layer_input = layer_input + feedback * 0.1  # 缩小反馈影响
                
                # 应用层归一化
                if self.use_layer_norm:
                    layer_input = self.layer_norms[i](layer_input)
                
                # 通过LTC单元
                h_i = self.ltc_cells[i](layer_input, h_i)
                
                # 应用记忆层归一化
                if self.use_layer_norm:
                    h_i = self.memory_norms[i](h_i)
                
                # 应用dropout (除了最后一层)
                if self.dropout is not None and i < self.num_layers - 1:
                    h_i = self.dropout(h_i)
                
                # 更新隐藏状态
                new_hidden[i] = h_i
                
                # 保存每层的状态，用于反馈
                layer_states[i].append(h_i)
                
                # 如果使用跳跃连接，将当前层输出添加到输入列表
                if self.use_skip_connections and i < self.num_layers - 1:
                    # 使用预定义的投影层
                    skip_conn = self.skip_connections[i](h_i)
                    layer_inputs.append(skip_conn)
            
            # 保存最后一层的输出
            outputs.append(new_hidden[-1])
        
        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, memory_size]
        
        return outputs, torch.stack(new_hidden, dim=0) 