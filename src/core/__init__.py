#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-RNN模型核心模块
"""

from .cells import LTCCell
from .wiring import NCPWiring, MultiLevelNCPWiring
from .model import LTC_NCP_RNN
from .adversarial import FGM
from .boundary_weights import BoundarySampleWeighter
from .pos_features import extract_combined_pos_features, preprocess_batch_texts
try:
    from .utils import concordance_correlation_coefficient
except (ImportError, AttributeError):
    # 如果utils模块中没有此函数，我们可以提供一个简单实现
    def concordance_correlation_coefficient(y_true, y_pred):
        """计算一致性相关系数 (CCC)"""
        import torch
        # 确保输入是torch张量
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        
        y_true_var = torch.var(y_true, unbiased=False)
        y_pred_var = torch.var(y_pred, unbiased=False)
        
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
        return ccc.item()  # 返回标量值而不是张量

__all__ = [
    'LTCCell',
    'NCPWiring',
    'MultiLevelNCPWiring',
    'LTC_NCP_RNN',
    'FGM',
    'BoundarySampleWeighter',
    'extract_combined_pos_features',
    'preprocess_batch_texts',
    'concordance_correlation_coefficient'
]
