#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版NCP (Neural Circuit Policy) 连线生成器
模仿线虫神经回路构建复杂稀疏连接图，支持多类型连接模式
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union, Callable


class NCPWiring(nn.Module):
    """神经电路策略连线模型，生成稀疏三层连接图"""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 sparsity_level: float = 0.5,  # 降低默认稀疏度，增加连接数
                 wiring_type: str = 'structured',  # 新增连接类型选项
                 heterogeneous: bool = True,  # 是否使用异构连接密度
                 emotion_focused: bool = True,  # 情感感知连接
                 modularity: int = 4,  # 模块化连接数量
                 seed: Optional[int] = None):
        """
        初始化增强版NCP连线结构
        
        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
            sparsity_level: 连接稀疏性 (0-1)，值越大连接越稀疏
            wiring_type: 连接类型 ('random', 'structured', 'small_world', 'modular')
            heterogeneous: 是否使用异构连接密度(不同层有不同稀疏度)
            emotion_focused: 是否强化情感相关区域连接
            modularity: 模块化连接中的模块数量
            seed: 随机种子，确保连接可重复性
        """
        super(NCPWiring, self).__init__()  # 调用nn.Module初始化
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sparsity_level = sparsity_level
        self.wiring_type = wiring_type
        self.heterogeneous = heterogeneous
        self.emotion_focused = emotion_focused
        self.modularity = modularity
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # 如果使用异构连接密度，为每层设置不同稀疏度
        if self.heterogeneous:
            # 让输入-隐藏层连接更密集，隐藏-隐藏层适中，隐藏-输出层最稀疏
            self.input_to_hidden_sparsity = max(0.1, self.sparsity_level - 0.2)
            self.hidden_to_hidden_sparsity = self.sparsity_level
            self.hidden_to_output_sparsity = min(0.9, self.sparsity_level + 0.1)
        else:
            self.input_to_hidden_sparsity = self.sparsity_level
            self.hidden_to_hidden_sparsity = self.sparsity_level
            self.hidden_to_output_sparsity = self.sparsity_level
        
        # 生成连接图
        self._generate_wiring()
        
        # 将掩码注册为缓冲区，这样它们会被保存在模型状态中
        self.register_buffer('input_to_hidden_mask_buffer', self.input_to_hidden_mask)
        self.register_buffer('hidden_to_hidden_mask_buffer', self.hidden_to_hidden_mask)
        self.register_buffer('hidden_to_output_mask_buffer', self.hidden_to_output_mask)

    def forward(self, x=None):
        """
        nn.Module需要实现forward方法，但NCPWiring主要是提供连接模式
        返回连接掩码以符合Module接口
        """
        # 确保所有掩码已初始化
        if not hasattr(self, 'input_to_hidden_mask') or not hasattr(self, 'hidden_to_hidden_mask') or not hasattr(self, 'hidden_to_output_mask'):
            # 如果掩码未初始化，尝试生成
            self._generate_wiring()
        
        # 返回连接掩码
        return {
            'input_to_hidden': self.input_to_hidden_mask,
            'hidden_to_hidden': self.hidden_to_hidden_mask,
            'hidden_to_output': self.hidden_to_output_mask
        }
    
    def _generate_wiring(self):
        """生成增强型连接的稀疏图结构"""
        # 选择合适的掩码生成方法
        if self.wiring_type == 'random':
            mask_method = self._create_random_mask
        elif self.wiring_type == 'structured':
            mask_method = self._create_structured_mask
        elif self.wiring_type == 'small_world':
            mask_method = self._create_small_world_mask
        elif self.wiring_type == 'modular':
            mask_method = self._create_modular_mask
        else:
            raise ValueError(f"不支持的连接类型: {self.wiring_type}")
        
        # 第一层连接：输入到隐藏层
        self.input_to_hidden_mask = mask_method(
            self.input_size, 
            self.hidden_size, 
            self.input_to_hidden_sparsity,
            is_input_layer=True
        )
        
        # 第二层连接：隐藏层内部循环连接
        self.hidden_to_hidden_mask = mask_method(
            self.hidden_size, 
            self.hidden_size, 
            self.hidden_to_hidden_sparsity,
            is_recurrent=True
        )
        
        # 第三层连接：隐藏层到输出层
        self.hidden_to_output_mask = mask_method(
            self.hidden_size, 
            self.output_size, 
            self.hidden_to_output_sparsity,
            is_output_layer=True
        )
        
        # 如果启用情感感知连接，强化情感敏感区域
        if self.emotion_focused:
            self._enhance_emotion_pathways()
        
        # 保存连接统计信息
        self.total_connections = (self.input_to_hidden_mask.sum() + 
                                  self.hidden_to_hidden_mask.sum() + 
                                  self.hidden_to_output_mask.sum())
        self.max_connections = (self.input_size * self.hidden_size + 
                               self.hidden_size * self.hidden_size + 
                               self.hidden_size * self.output_size)
        self.sparsity = 1.0 - (self.total_connections / self.max_connections)
        
        # 可视化信息
        self._calculate_visualization_data()
    
    def _create_random_mask(self, rows, cols, sparsity, **kwargs):
        """
        创建随机稀疏连接掩码
        
        参数:
            rows: 源层神经元数量
            cols: 目标层神经元数量
            sparsity: 稀疏度级别 (0-1)
        
        返回:
            布尔掩码矩阵，表示连接存在(True)或不存在(False)
        """
        # 概率法：每个连接以 1-sparsity 的概率存在
        mask = torch.from_numpy(np.random.rand(rows, cols) > sparsity)
        return mask.float()
    
    def _create_structured_mask(self, rows, cols, sparsity, **kwargs):
        """
        创建结构化稀疏连接掩码 - 每个神经元连接到特定区域
        
        参数:
            rows: 源层神经元数量
            cols: 目标层神经元数量
            sparsity: 稀疏度级别 (0-1)
            
        返回:
            浮点掩码矩阵，表示连接强度 (0-1)
        """
        is_input_layer = kwargs.get('is_input_layer', False)
        is_output_layer = kwargs.get('is_output_layer', False)
        is_recurrent = kwargs.get('is_recurrent', False)
        
        # 创建基础掩码
        mask = torch.zeros(rows, cols)
        
        if is_input_layer:
            # 输入层：每个隐藏单元接收来自连续输入区域的信号（类似卷积）
            receptive_field_size = max(3, int(rows * (1-sparsity) / 2))
            for i in range(cols):
                # 将隐藏单元均匀分布在输入空间上
                center = int((i / cols) * rows)
                # 创建感受野
                start = max(0, center - receptive_field_size // 2)
                end = min(rows, center + receptive_field_size // 2 + 1)
                mask[start:end, i] = 1.0
                
                # 添加一些随机长距离连接
                if np.random.rand() < 0.2:  # 20%概率添加远距离连接
                    random_inputs = np.random.choice(rows, size=2, replace=False)
                    mask[random_inputs, i] = 1.0
        
        elif is_output_layer:
            # 输出层：确保每个输出单元接收来自多个隐藏单元的输入
            min_fans = max(3, int(rows * (1-sparsity)))
            for i in range(cols):
                # 为每个输出选择固定数量的随机输入
                random_inputs = np.random.choice(rows, size=min_fans, replace=False)
                mask[random_inputs, i] = 1.0
                
                # 确保情感输出单元 (V和A) 接收足够多样的输入
                if i < 2 and cols >= 2:  # V和A输出
                    extra_inputs = np.random.choice(rows, size=min_fans//2, replace=False)
                    mask[extra_inputs, i] = 1.0
        
        elif is_recurrent:
            # 循环层：创建小世界网络结构
            # 首先确保局部连接
            k = max(2, int(cols * (1-sparsity) / 3))  # 每个节点连接到k个邻居
            for i in range(rows):
                # 连接到邻近神经元
                for j in range(1, k+1):
                    mask[i, (i+j) % cols] = 1.0
                    mask[i, (i-j) % cols] = 1.0
                
                # 添加一些长距离连接
                rewire_prob = 0.2
                if np.random.rand() < rewire_prob:
                    target = np.random.randint(0, cols)
                    mask[i, target] = 1.0
            
            # 确保有一些强隐藏单元能与多数其他单元通信（中心节点）
            hub_count = max(1, int(rows * 0.05))  # 5%的神经元作为中心
            hubs = np.random.choice(rows, size=hub_count, replace=False)
            for hub in hubs:
                # 中心节点连接到更多其他节点
                targets = np.random.choice(cols, size=int(cols * 0.3), replace=False)
                mask[hub, targets] = 1.0
        
        else:
            # 默认情况：创建块状结构
            if rows > cols:
                # 每个目标接收多个源的输入
                sources_per_target = max(1, int(rows * (1-sparsity) / cols))
                for i in range(cols):
                    # 选择源
                    start_idx = (i * sources_per_target) % rows
                    indices = [(start_idx + j) % rows for j in range(sources_per_target)]
                    mask[indices, i] = 1.0
            else:
                # 每个源连接到多个目标
                targets_per_source = max(1, int(cols * (1-sparsity) / rows))
                for i in range(rows):
                    # 选择目标
                    start_idx = (i * targets_per_source) % cols
                    indices = [(start_idx + j) % cols for j in range(targets_per_source)]
                    mask[i, indices] = 1.0
        
        # 应用随机变化以打破对称性
        random_noise = torch.rand(rows, cols) * 0.1
        mask = mask + (mask > 0) * random_noise
        
        # 归一化为二值掩码
        mask = (mask > 0).float()
        
        # 检查连接数是否符合稀疏度要求，如果连接过多或过少，进行调整
        actual_density = mask.mean().item()
        target_density = 1 - sparsity
        
        if abs(actual_density - target_density) > 0.1:
            if actual_density > target_density:
                # 连接过多，随机删除一些
                prune_mask = torch.rand(rows, cols) < (target_density / actual_density)
                mask = mask * prune_mask.float()
            else:
                # 连接过少，随机添加一些
                add_prob = (target_density - actual_density) / (1 - actual_density)
                add_mask = (torch.rand(rows, cols) < add_prob) & (mask == 0)
                mask = mask + add_mask.float()
        
        return mask
    
    def _create_small_world_mask(self, rows, cols, sparsity, **kwargs):
        """
        创建小世界网络稀疏连接掩码 - 结合局部连接和随机长距离连接
        
        参数:
            rows: 源层神经元数量
            cols: 目标层神经元数量
            sparsity: 稀疏度级别 (0-1)
        
        返回:
            浮点掩码矩阵，表示连接强度 (0-1)
        """
        is_recurrent = kwargs.get('is_recurrent', False)
        
        # 小世界参数
        k = max(2, int(min(rows, cols) * (1-sparsity) / 2))  # 每个节点初始连接数
        beta = 0.2  # 重布线概率
        
        # 初始化掩码
        mask = torch.zeros(rows, cols)
        
        if rows == cols and is_recurrent:
            # 为循环层创建真正的小世界网络（要求行列相等）
            # 首先创建规则环形格局
            for i in range(rows):
                for j in range(1, k+1):
                    # 添加右邻居
                    right_neighbor = (i + j) % cols
                    mask[i, right_neighbor] = 1.0
                    # 添加左邻居
                    left_neighbor = (i - j) % cols
                    mask[i, left_neighbor] = 1.0
            
            # 应用重布线过程 - 以概率beta重布线
            for i in range(rows):
                for j in range(cols):
                    if mask[i, j] > 0 and np.random.rand() < beta:
                        # 删除当前连接
                        mask[i, j] = 0.0
                        # 添加随机长距离连接
                        new_target = np.random.randint(0, cols)
                        # 确保不是自环
                        while new_target == i or mask[i, new_target] > 0:
                            new_target = np.random.randint(0, cols)
                        mask[i, new_target] = 1.0
        else:
            # 对于不同大小的层，创建类小世界结构
            # 将行列映射到相同的空间
            scale_factor = cols / rows if rows < cols else 1.0
            
            for i in range(rows):
                # 映射后的位置
                mapped_i = int(i * scale_factor) if rows < cols else i
                
                # 确定连接数量
                num_connections = max(1, int(cols * (1-sparsity)))
                
                # 局部连接
                local_conn = int(num_connections * (1-beta))
                if local_conn > 0:
                    local_targets = []
                    for j in range(1, local_conn+1):
                        right = (mapped_i + j) % cols
                        left = (mapped_i - j) % cols
                        local_targets.extend([right, left])
                    
                    local_targets = list(set(local_targets))[:min(local_conn*2, cols)]
                    for target in local_targets:
                        mask[i, target] = 1.0
                
                # 随机长距离连接
                random_conn = num_connections - local_conn
                if random_conn > 0:
                    # 选择随机目标，避免已连接的
                    available_targets = [j for j in range(cols) if mask[i, j] == 0]
                    if available_targets:
                        random_targets = np.random.choice(
                            available_targets, 
                            size=min(random_conn, len(available_targets)), 
                            replace=False
                        )
                        for target in random_targets:
                            mask[i, target] = 1.0
        
        return mask
    
    def _create_modular_mask(self, rows, cols, sparsity, **kwargs):
        """
        创建模块化稀疏连接掩码 - 将神经元分组，组内连接密集，组间连接稀疏
        
        参数:
            rows: 源层神经元数量
            cols: 目标层神经元数量
            sparsity: 稀疏度级别 (0-1)
        
        返回:
            浮点掩码矩阵，表示连接强度 (0-1)
        """
        # 确定模块数量
        n_modules = min(self.modularity, min(rows, cols))
        
        # 初始化掩码
        mask = torch.zeros(rows, cols)
        
        # 将行和列分配到模块中
        row_modules = np.array_split(range(rows), n_modules)
        col_modules = np.array_split(range(cols), n_modules)
        
        # 组内连接密度
        intra_density = min(0.9, 1.0 - (sparsity * 0.5))
        
        # 组间连接密度
        inter_density = max(0.1, (1.0 - sparsity) - (intra_density / n_modules))
        
        # 创建模块内连接（密集）
        for i in range(n_modules):
            for row in row_modules[i]:
                for col in col_modules[i]:
                    # 以intra_density的概率创建连接
                    if np.random.rand() < intra_density:
                        mask[row, col] = 1.0
        
        # 创建模块间连接（稀疏）
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j:  # 跨模块
                    for row in row_modules[i]:
                        # 随机选择目标模块中的一些列
                        target_cols = np.random.choice(
                            col_modules[j], 
                            size=max(1, int(len(col_modules[j]) * inter_density)), 
                            replace=False
                        )
                        for col in target_cols:
                            mask[row, col] = 1.0
        
        return mask
    
    def _enhance_emotion_pathways(self):
        """增强情感通路 - 创建多层次情感特化连接结构"""
        # 我们假设输出的前两个神经元是Valence和Arousal
        if self.output_size >= 2:
            v_output, a_output = 0, 1
            
            # 1. 大幅增强情感输出连接密度
            enhancement_factor = 3.0  # 显著提高情感路径增强系数
            
            # 计算当前连接数
            v_connections = int(self.hidden_to_output_mask[:, v_output].sum().item())
            a_connections = int(self.hidden_to_output_mask[:, a_output].sum().item())
            
            # 目标是大幅增加V和A的连接数量，最多到隐藏层的75%
            v_target = min(int(self.hidden_size * 0.75), int(v_connections * enhancement_factor))
            a_target = min(int(self.hidden_size * 0.75), int(a_connections * enhancement_factor))
            
            # 添加新连接
            v_to_add = v_target - v_connections
            a_to_add = a_target - a_connections
            
            # 2. 创建V/A专用处理区域
            # 将隐藏层分为三部分：V特化区域、A特化区域、共享区域
            v_region_size = self.hidden_size // 3
            a_region_size = self.hidden_size // 3
            shared_region_size = self.hidden_size - v_region_size - a_region_size
            
            v_region = list(range(0, v_region_size))
            a_region = list(range(v_region_size, v_region_size + a_region_size))
            shared_region = list(range(v_region_size + a_region_size, self.hidden_size))
            
            # 3. 确保V特化区域强连接到V输出
            # 首先清空当前连接以建立新的结构
            for i in v_region:
                self.hidden_to_output_mask[i, v_output] = 1.0
                # 给一些弱连接到A
                if np.random.random() < 0.3:
                    self.hidden_to_output_mask[i, a_output] = 0.5
            
            # 4. 确保A特化区域强连接到A输出
            for i in a_region:
                self.hidden_to_output_mask[i, a_output] = 1.0
                # 给一些弱连接到V
                if np.random.random() < 0.3:
                    self.hidden_to_output_mask[i, v_output] = 0.5
            
            # 5. 共享区域连接两个输出，捕捉V/A相关性
            for i in shared_region:
                # 大多数共享单元同时连接V和A
                if np.random.random() < 0.8:
                    self.hidden_to_output_mask[i, v_output] = 1.0
                    self.hidden_to_output_mask[i, a_output] = 1.0
                # 其余单元随机连接一个
                else:
                    target = v_output if np.random.random() < 0.5 else a_output
                    self.hidden_to_output_mask[i, int(target)] = 1.0
            
            # 6. 增强输入层到情感特化区域的连接
            # 找出输入层可能与情感相关的单元(例如情感词汇嵌入)
            input_size = self.input_to_hidden_mask.size(0)
            
            # 假设输入的前20%与情感特征相关性更高
            emotion_inputs = int(input_size * 0.2)
            
            # 为V特化区域创建从情感输入的额外连接
            for i in range(emotion_inputs):
                for j in v_region:
                    if np.random.random() < 0.6:  # 60%概率建立连接
                        self.input_to_hidden_mask[i, j] = 1.0
            
            # 为A特化区域创建从情感输入的额外连接
            for i in range(emotion_inputs):
                for j in a_region:
                    if np.random.random() < 0.6:  # 60%概率建立连接
                        self.input_to_hidden_mask[i, j] = 1.0
            
            # 7. 增强隐藏层内部的情感区域互连
            # V区域内部建立密集连接
            for i in v_region:
                for j in v_region:
                    if i != j and np.random.random() < 0.4:  # 避免自连接，40%连接概率
                        self.hidden_to_hidden_mask[i, j] = 1.0
            
            # A区域内部建立密集连接
            for i in a_region:
                for j in a_region:
                    if i != j and np.random.random() < 0.4:  # 避免自连接，40%连接概率
                        self.hidden_to_hidden_mask[i, j] = 1.0
            
            # 区域间建立适度的交叉连接，允许信息交流
            for i in v_region:
                for j in a_region:
                    if np.random.random() < 0.2:  # 20%概率建立跨区域连接
                        self.hidden_to_hidden_mask[i, j] = 1.0
                        
            for i in a_region:
                for j in v_region:
                    if np.random.random() < 0.2:  # 20%概率建立跨区域连接
                        self.hidden_to_hidden_mask[i, j] = 1.0
            
            # 共享区域与V/A区域双向连接
            for i in shared_region:
                # 连接到V区域
                for j in v_region:
                    if np.random.random() < 0.3:
                        self.hidden_to_hidden_mask[i, j] = 1.0
                        
                # 连接到A区域
                for j in a_region:
                    if np.random.random() < 0.3:
                        self.hidden_to_hidden_mask[i, j] = 1.0
                        
                # V区域连接到共享区域
                for j in v_region:
                    if np.random.random() < 0.3:
                        self.hidden_to_hidden_mask[j, i] = 1.0
                
                # A区域连接到共享区域
                for j in a_region:
                    if np.random.random() < 0.3:
                        self.hidden_to_hidden_mask[j, i] = 1.0
    
    def _calculate_visualization_data(self):
        """计算连接图的可视化数据，用于分析和调试"""
        # 每层的连接密度
        self.in_to_hidden_density = self.input_to_hidden_mask.sum().item() / (self.input_size * self.hidden_size)
        self.hidden_to_hidden_density = self.hidden_to_hidden_mask.sum().item() / (self.hidden_size * self.hidden_size)
        self.hidden_to_out_density = self.hidden_to_output_mask.sum().item() / (self.hidden_size * self.output_size)
        
        # 每个神经元的平均出度和入度
        self.avg_in_degree = []
        self.avg_in_degree.append(self.input_to_hidden_mask.sum(dim=0).float().mean().item())  # 隐藏层神经元平均输入连接数
        self.avg_in_degree.append(self.hidden_to_hidden_mask.sum(dim=0).float().mean().item())  # 隐藏层神经元平均隐藏层输入连接数
        self.avg_in_degree.append(self.hidden_to_output_mask.sum(dim=0).float().mean().item())  # 输出层神经元平均输入连接数
        
        self.avg_out_degree = []
        self.avg_out_degree.append(self.input_to_hidden_mask.sum(dim=1).float().mean().item())  # 输入层神经元平均输出连接数
        self.avg_out_degree.append(self.hidden_to_hidden_mask.sum(dim=1).float().mean().item())  # 隐藏层神经元平均隐藏层输出连接数
        self.avg_out_degree.append(self.hidden_to_output_mask.sum(dim=1).float().mean().item())  # 隐藏层神经元平均输出层输出连接数
    
    def get_connection_stats(self):
        """返回连接统计信息"""
        return {
            'total_connections': self.total_connections,
            'max_possible_connections': self.max_connections,
            'sparsity': self.sparsity,
            'wiring_type': self.wiring_type,
            'layer_densities': {
                'input_to_hidden': self.in_to_hidden_density,
                'hidden_to_hidden': self.hidden_to_hidden_density,
                'hidden_to_output': self.hidden_to_out_density
            },
            'avg_in_degree': self.avg_in_degree,
            'avg_out_degree': self.avg_out_degree
        }
    
    def apply_mask(self, module, mask, name=None):
        """
        应用掩码到线性层，将不需要的连接权重设为0
        
        参数:
            module: 线性层模块
            mask: 稀疏连接布尔掩码
            name: 可选的名称用于日志
        """
        assert hasattr(module, 'weight'), "模块必须有权重属性"
        
        # 检查掩码与权重矩阵维度是否匹配
        weight_shape = module.weight.shape
        mask_shape = mask.shape
        
        # 如果维度不匹配，调整掩码
        if weight_shape != mask_shape:
            # 创建与权重形状匹配的新掩码
            new_mask = torch.zeros(weight_shape, dtype=torch.float, device=mask.device)
            
            # 复制原始掩码到新掩码（覆盖可能的范围）
            rows = min(mask_shape[0], weight_shape[0])
            cols = min(mask_shape[1], weight_shape[1])
            
            new_mask[:rows, :cols] = mask[:rows, :cols]
            mask = new_mask
        
        # 注册前向钩子来确保权重在前向传播时被掩蔽
        def apply_mask_hook(module, input, output):
            module.weight.data *= mask.to(module.weight.device)
            return output
        
        # 注册反向钩子来确保梯度在反向传播时被掩蔽
        def apply_mask_hook_backward(module, grad_input, grad_output):
            if module.weight.grad is not None:
                module.weight.grad.data *= mask.to(module.weight.device)
            # 确保我们不返回None作为grad_input
            return grad_input if grad_input is not None else None
        
        # 应用掩码到当前权重
        with torch.no_grad():
            module.weight.data *= mask.to(module.weight.device)
        
        # 注册钩子
        mask_hook = module.register_forward_hook(apply_mask_hook)
        mask_hook_backward = module.register_full_backward_hook(apply_mask_hook_backward)
        
        return {'forward': mask_hook, 'backward': mask_hook_backward}
    
    def to_dict(self):
        """将NCP配置导出为字典，方便保存和加载"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'sparsity_level': self.sparsity_level,
            'wiring_type': self.wiring_type,
            'heterogeneous': self.heterogeneous,
            'emotion_focused': self.emotion_focused,
            'modularity': self.modularity,
            'masks': {
                'input_to_hidden': self.input_to_hidden_mask,
                'hidden_to_hidden': self.hidden_to_hidden_mask,
                'hidden_to_output': self.hidden_to_output_mask
            },
            'stats': self.get_connection_stats()
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载NCP配置"""
        instance = cls(
            input_size=config_dict['input_size'],
            hidden_size=config_dict['hidden_size'],
            output_size=config_dict['output_size'],
            sparsity_level=config_dict['sparsity_level'],
            wiring_type=config_dict.get('wiring_type', 'structured'),
            heterogeneous=config_dict.get('heterogeneous', True),
            emotion_focused=config_dict.get('emotion_focused', True),
            modularity=config_dict.get('modularity', 4)
        )
        
        # 使用保存的掩码替换生成的掩码
        instance.input_to_hidden_mask = config_dict['masks']['input_to_hidden']
        instance.hidden_to_hidden_mask = config_dict['masks']['hidden_to_hidden']
        instance.hidden_to_output_mask = config_dict['masks']['hidden_to_output']
        
        # 同时更新缓冲区
        instance.register_buffer('input_to_hidden_mask_buffer', instance.input_to_hidden_mask)
        instance.register_buffer('hidden_to_hidden_mask_buffer', instance.hidden_to_hidden_mask)
        instance.register_buffer('hidden_to_output_mask_buffer', instance.hidden_to_output_mask)
        
        # 重新计算统计数据
        instance._calculate_visualization_data()
        
        return instance


class MultiLevelNCPWiring(nn.Module):
    """多层次神经电路策略连线，增强处理复杂情感模式的能力"""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 levels: int = 2,
                 sparsity_levels: List[float] = [0.5, 0.3],
                 wiring_types: List[str] = ['structured', 'small_world'],
                 emotion_focused: bool = True):
        """
        初始化多层次NCP连线结构
        
        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
            levels: 处理层次数量
            sparsity_levels: 每层的稀疏度
            wiring_types: 每层的连接类型
            emotion_focused: 是否添加情感感知连接
        """
        super(MultiLevelNCPWiring, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.levels = levels
        self.sparsity_levels = sparsity_levels
        self.wiring_types = wiring_types
        self.emotion_focused = emotion_focused
        
        # 确保稀疏度和连接类型列表长度匹配层数
        if len(self.sparsity_levels) < self.levels:
            self.sparsity_levels.extend([self.sparsity_levels[-1]] * (self.levels - len(self.sparsity_levels)))
        if len(self.wiring_types) < self.levels:
            self.wiring_types.extend([self.wiring_types[-1]] * (self.levels - len(self.wiring_types)))
        
        # 创建各层的连线结构
        self.wirings = nn.ModuleList()
        level_hidden_size = self.hidden_size // self.levels
        remainder = self.hidden_size % self.levels
        
        for i in range(self.levels):
            # 分配隐藏单元数量，处理除法余数
            level_size = level_hidden_size + (1 if i < remainder else 0)
            
            # 创建当前层的布线
            wiring = NCPWiring(
                input_size = self.input_size if i == 0 else self.wirings[i-1].hidden_size,
                hidden_size = level_size,
                output_size = self.output_size if i == self.levels - 1 else level_size,
                sparsity_level = self.sparsity_levels[i],
                wiring_type = self.wiring_types[i],
                emotion_focused = self.emotion_focused and i == self.levels - 1  # 只在最后一层启用情感感知
            )
            
            self.wirings.append(wiring)
        
        # 添加层间整合连接
        self._generate_integration_connections()
        
        # 计算掩码组合
        self._combine_masks()
    
    def forward(self, x=None):
        """
        返回合并后的掩码
        """
        # 确保掩码已生成
        if not hasattr(self, 'input_to_hidden_mask') or not hasattr(self, 'hidden_to_hidden_mask') or not hasattr(self, 'hidden_to_output_mask'):
            self._combine_masks()
        
        return {
            'input_to_hidden': self.input_to_hidden_mask,
            'hidden_to_hidden': self.hidden_to_hidden_mask,
            'hidden_to_output': self.hidden_to_output_mask
        }
    
    def _combine_masks(self):
        """组合各层掩码为整体掩码"""
        # 初始化组合掩码
        self.input_to_hidden_mask = self.wirings[0].input_to_hidden_mask
        
        # 隐藏层到隐藏层连接 - 组合所有层的内部连接和层间连接
        hidden_size = sum(wiring.hidden_size for wiring in self.wirings)
        self.hidden_to_hidden_mask = torch.zeros(hidden_size, hidden_size)
        
        # 填充掩码 - 块状对角矩阵结构
        row_offset = 0
        for i, wiring in enumerate(self.wirings):
            col_offset = 0
            for j, target_wiring in enumerate(self.wirings):
                if i == j:
                    # 层内连接
                    h_size = wiring.hidden_size
                    self.hidden_to_hidden_mask[
                        row_offset:row_offset+h_size, 
                        col_offset:col_offset+h_size
                    ] = wiring.hidden_to_hidden_mask
                elif i+1 == j:
                    # 层间前向连接
                    if hasattr(self, f'integration_mask_{i}_to_{j}'):
                        integration_mask = getattr(self, f'integration_mask_{i}_to_{j}')
                        self.hidden_to_hidden_mask[
                            row_offset:row_offset+wiring.hidden_size, 
                            col_offset:col_offset+target_wiring.hidden_size
                        ] = integration_mask
                
                col_offset += target_wiring.hidden_size
            row_offset += wiring.hidden_size
        
        # 隐藏层到输出层连接 - 只使用最后一层的连接
        self.hidden_to_output_mask = torch.zeros(hidden_size, self.output_size)
        last_wiring = self.wirings[-1]
        row_offset = hidden_size - last_wiring.hidden_size
        self.hidden_to_output_mask[row_offset:, :] = last_wiring.hidden_to_output_mask
    
    def _generate_integration_connections(self):
        """生成层间整合连接，用于信息交换"""
        self.integration_masks = []
        
        # 对每个层次对创建整合连接
        for i in range(self.levels - 1):
            source_size = self.wirings[i].hidden_size
            target_size = self.wirings[i+1].hidden_size
            
            # 创建稀疏整合掩码 - 45%连接概率
            integration_mask = torch.from_numpy(np.random.rand(source_size, target_size) < 0.45)
            self.integration_masks.append(integration_mask.float())
    
    def apply_masks_to_model(self, model):
        """将所有掩码应用到模型的相应层"""
        # 该方法需要根据具体模型实现定制
        pass
    
    def get_stats(self):
        """返回各层连接统计信息"""
        stats = {}
        
        for i, wiring in enumerate(self.wirings):
            stats[f'level_{i+1}'] = wiring.get_connection_stats()
        
        # 计算层间整合连接
        if hasattr(self, 'integration_masks'):
            integration_stats = {}
            for i, mask in enumerate(self.integration_masks):
                integration_stats[f'level_{i+1}_to_{i+2}'] = {
                    'connections': mask.sum().item(),
                    'max_possible': mask.numel(),
                    'density': mask.mean().item()
                }
            stats['integration'] = integration_stats
            
        return stats 