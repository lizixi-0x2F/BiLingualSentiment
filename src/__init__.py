#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 情感价效度回归模型
=================================

基于液态时间常数网络与神经电路策略的文本情感分析模型
"""

__version__ = '0.1.0'

# 从核心模块导入主要组件
from .core import (
    LTC_NCP_RNN,
    LTCCell,
    NCPWiring,
    MultiLevelNCPWiring,
    concordance_correlation_coefficient,
    XLMRTransformerBranch
)

# 公开导出组件
__all__ = [
    'LTC_NCP_RNN',
    'LTCCell',
    'NCPWiring',
    'MultiLevelNCPWiring',
    'concordance_correlation_coefficient',
    'XLMRTransformerBranch'
]