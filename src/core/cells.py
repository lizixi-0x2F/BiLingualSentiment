#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC (Liquid Time Constant) 细胞实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LTCCell(nn.Module):
    """
    液态时间常数单元
    
    隐状态满足一阶ODE：h_{t+1} = h_{t} + (Δt/τ_t) * (-h_t + W_x*x_t + W_h*σ(h_t))
    
    时间常数τ可学习，连续时间动态提升时间分辨率
    """
    
    def __init__(self, input_size, hidden_size, dt=1.0, method='euler'):
        """
        初始化LTC细胞
        
        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐状态维度
            dt (float): 时间步长
            method (str): 数值积分方法，'euler'或'rk4'
        """
        super(LTCCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.method = method
        
        # 输入到隐状态的连接（W_x）
        self.W_x = nn.Linear(input_size, hidden_size, bias=False)
        
        # 隐状态到隐状态的连接（W_h）
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 时间常数参数（τ）- 可学习，初始化为1.0
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # 初始化参数
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.orthogonal_(self.W_h.weight)
        
        # 初始化tau为稍微随机的值（围绕1.0）
        with torch.no_grad():
            self.tau.data = torch.ones_like(self.tau) + 0.1 * torch.randn_like(self.tau)
            # 确保tau为正数
            self.tau.data = torch.abs(self.tau.data)
    
    def _ode_func(self, h, x):
        """
        ODE函数: dh/dt = (-h + W_x*x + W_h*σ(h)) / τ
        
        参数:
            h: 隐状态 [batch_size, hidden_size]
            x: 输入 [batch_size, input_size]
            
        返回:
            dh: 隐状态变化率 [batch_size, hidden_size]
        """
        sigma_h = torch.tanh(h)  # 激活函数：tanh
        dh = (-h + self.W_x(x) + self.W_h(sigma_h) + self.bias) / self.tau
        return dh
    
    def _forward_euler(self, h, x):
        """使用欧拉法进行时间积分"""
        dh = self._ode_func(h, x)
        h_next = h + self.dt * dh
        return h_next
    
    def _forward_rk4(self, h, x):
        """使用四阶龙格-库塔法进行时间积分"""
        dt = self.dt
        k1 = self._ode_func(h, x)
        k2 = self._ode_func(h + dt * k1 / 2, x)
        k3 = self._ode_func(h + dt * k2 / 2, x)
        k4 = self._ode_func(h + dt * k3, x)
        h_next = h + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        return h_next
    
    def forward(self, x, h=None):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, input_size] 或 [batch_size, seq_len, input_size]
            h: 隐状态 [batch_size, hidden_size]，如果为None则初始化为零
            
        返回:
            h_next: 下一个隐状态 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size]
        """
        # 检查输入维度
        if x.dim() == 3:  # [batch_size, seq_len, input_size]
            batch_size, seq_len, _ = x.size()
            # 处理序列数据
            if h is None:
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                
            outputs = []
            for t in range(seq_len):
                h = self._step(x[:, t, :], h)
                outputs.append(h.unsqueeze(1))
                
            # 拼接所有时间步的输出 [batch_size, seq_len, hidden_size]
            return torch.cat(outputs, dim=1)
            
        else:  # [batch_size, input_size]
            return self._step(x, h)
            
    def _step(self, x, h=None):
        """
        单步前向传播
        
        参数:
            x: 输入张量 [batch_size, input_size]
            h: 隐状态 [batch_size, hidden_size]，如果为None则初始化为零
            
        返回:
            h_next: 下一个隐状态 [batch_size, hidden_size]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # 根据积分方法选择前向传播方式
        if self.method == 'euler':
            h_next = self._forward_euler(h, x)
        elif self.method == 'rk4':
            h_next = self._forward_rk4(h, x)
        else:
            raise ValueError(f"未知的积分方法: {self.method}")
        
        return h_next
    
    def get_param_stats(self):
        """返回参数统计信息，用于监控和调试"""
        stats = {
            'tau_mean': self.tau.mean().item(),
            'tau_std': self.tau.std().item(),
            'tau_min': self.tau.min().item(),
            'tau_max': self.tau.max().item(),
            'wx_norm': self.W_x.weight.norm().item(),
            'wh_norm': self.W_h.weight.norm().item()
        }
        return stats 