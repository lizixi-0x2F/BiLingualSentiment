#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTC-NCP-VA 模型的实用工具函数
包含象限评估、边界处理和增强的评估指标
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def get_quadrant(va_values):
    """
    根据VA值确定情感象限
    
    参数:
        va_values: 形状为[batch_size, 2]的VA值
        
    返回:
        象限索引 (0-3):
        - 0: 第一象限 (V+, A+) - 喜悦/兴奋
        - 1: 第二象限 (V+, A-) - 满足/平静
        - 2: 第三象限 (V-, A+) - 愤怒/焦虑
        - 3: 第四象限 (V-, A-) - 悲伤/抑郁
    """
    if isinstance(va_values, torch.Tensor):
        valence = va_values[:, 0]
        arousal = va_values[:, 1]
        
        quadrants = torch.zeros(len(valence), dtype=torch.long, device=va_values.device)
        quadrants[(valence >= 0) & (arousal >= 0)] = 0  # 第一象限
        quadrants[(valence >= 0) & (arousal < 0)] = 1   # 第二象限
        quadrants[(valence < 0) & (arousal >= 0)] = 2   # 第三象限
        quadrants[(valence < 0) & (arousal < 0)] = 3    # 第四象限
    else:
        valence = va_values[:, 0]
        arousal = va_values[:, 1]
        
        quadrants = np.zeros(len(valence), dtype=np.int64)
        quadrants[(valence >= 0) & (arousal >= 0)] = 0  # 第一象限
        quadrants[(valence >= 0) & (arousal < 0)] = 1   # 第二象限
        quadrants[(valence < 0) & (arousal >= 0)] = 2   # 第三象限
        quadrants[(valence < 0) & (arousal < 0)] = 3    # 第四象限
        
    return quadrants

def get_quadrant_metrics(true_va, pred_va):
    """
    计算四象限分类的评估指标
    
    参数:
        true_va: numpy数组，形状为[N, 2]的真实VA值
        pred_va: numpy数组，形状为[N, 2]的预测VA值
        
    返回:
        包含各种指标的字典
    """
    # 获取真实和预测的象限
    true_quadrants = get_quadrant(true_va)
    pred_quadrants = get_quadrant(pred_va)
    
    # 计算基础指标
    accuracy = np.mean(true_quadrants == pred_quadrants)
    
    # 计算每个象限的precision、recall和F1
    precision = precision_score(true_quadrants, pred_quadrants, average=None, zero_division=0)
    recall = recall_score(true_quadrants, pred_quadrants, average=None, zero_division=0)
    f1 = f1_score(true_quadrants, pred_quadrants, average=None, zero_division=0)
    
    # 计算加权指标
    weighted_precision = precision_score(true_quadrants, pred_quadrants, average='weighted', zero_division=0)
    weighted_recall = recall_score(true_quadrants, pred_quadrants, average='weighted', zero_division=0)
    weighted_f1 = f1_score(true_quadrants, pred_quadrants, average='weighted', zero_division=0)
    
    # 计算宏平均指标
    macro_precision = precision_score(true_quadrants, pred_quadrants, average='macro', zero_division=0)
    macro_recall = recall_score(true_quadrants, pred_quadrants, average='macro', zero_division=0)
    macro_f1 = f1_score(true_quadrants, pred_quadrants, average='macro', zero_division=0)
    
    # 生成混淆矩阵
    cm = confusion_matrix(true_quadrants, pred_quadrants, labels=[0, 1, 2, 3])
    
    # 计算符号错误率
    v_sign_error = np.mean(np.sign(true_va[:, 0]) != np.sign(pred_va[:, 0]))
    a_sign_error = np.mean(np.sign(true_va[:, 1]) != np.sign(pred_va[:, 1]))
    
    # 组织结果
    results = {
        'accuracy': accuracy,
        'per_quadrant': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'confusion_matrix': cm,
        'sign_error': {
            'V': v_sign_error,
            'A': a_sign_error,
            'avg': (v_sign_error + a_sign_error) / 2
        }
    }
    
    return results

def concordance_correlation_coefficient(y_true, y_pred):
    """
    计算一致性相关系数 (CCC)
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    返回:
        CCC值，范围[-1, 1]
    """
    # 确保输入是ndarray
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    
    y_true_var = np.var(y_true, ddof=0)
    y_pred_var = np.var(y_pred, ddof=0)
    
    covariance = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc

def get_va_metrics(true_va, pred_va):
    """
    计算VA值的评估指标
    
    参数:
        true_va: numpy数组，形状为[N, 2]的真实VA值
        pred_va: numpy数组，形状为[N, 2]的预测VA值
        
    返回:
        包含各种指标的字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # 计算V和A的RMSE
    rmse_v = np.sqrt(mean_squared_error(true_va[:, 0], pred_va[:, 0]))
    rmse_a = np.sqrt(mean_squared_error(true_va[:, 1], pred_va[:, 1]))
    
    # 计算V和A的MAE
    mae_v = mean_absolute_error(true_va[:, 0], pred_va[:, 0])
    mae_a = mean_absolute_error(true_va[:, 1], pred_va[:, 1])
    
    # 计算V和A的CCC
    ccc_v = concordance_correlation_coefficient(true_va[:, 0], pred_va[:, 0])
    ccc_a = concordance_correlation_coefficient(true_va[:, 1], pred_va[:, 1])
    
    # 计算V和A的偏差
    bias_v = np.mean(pred_va[:, 0] - true_va[:, 0])
    bias_a = np.mean(pred_va[:, 1] - true_va[:, 1])
    
    # 组织结果
    results = {
        'rmse': {
            'V': rmse_v,
            'A': rmse_a,
            'avg': (rmse_v + rmse_a) / 2
        },
        'mae': {
            'V': mae_v,
            'A': mae_a,
            'avg': (mae_v + mae_a) / 2
        },
        'ccc': {
            'V': ccc_v,
            'A': ccc_a,
            'avg': (ccc_v + ccc_a) / 2
        },
        'bias': {
            'V': bias_v,
            'A': bias_a
        }
    }
    
    return results

def format_quadrant_metrics(metrics):
    """格式化四象限指标，生成可读的字符串报告"""
    report = []
    
    cm = metrics['confusion_matrix']
    accuracy = metrics['accuracy']
    
    # 添加准确率
    report.append(f"象限准确率: {accuracy:.4f}")
    
    # 添加加权指标
    report.append(f"加权 F1: {metrics['weighted']['f1']:.4f}")
    report.append(f"加权 Precision: {metrics['weighted']['precision']:.4f}")
    report.append(f"加权 Recall: {metrics['weighted']['recall']:.4f}")
    
    # 添加宏平均指标
    report.append(f"宏平均 F1: {metrics['macro']['f1']:.4f}")
    
    # 添加每个象限的指标
    report.append("\n每个象限的指标:")
    quadrant_names = ["喜悦/兴奋", "满足/平静", "愤怒/焦虑", "悲伤/抑郁"]
    for i, name in enumerate(quadrant_names):
        p = metrics['per_quadrant']['precision'][i]
        r = metrics['per_quadrant']['recall'][i]
        f1 = metrics['per_quadrant']['f1'][i]
        report.append(f"  {name}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    # 添加混淆矩阵
    report.append("\n混淆矩阵:")
    report.append("预测 →")
    report.append("实际 ↓ | " + " | ".join(f"{name}" for name in quadrant_names))
    report.append("-" * 60)
    for i, name in enumerate(quadrant_names):
        report.append(f"{name} | " + " | ".join(f"{cm[i, j]:5d}" for j in range(4)))
    
    # 添加符号错误率
    report.append(f"\n符号错误率: V={metrics['sign_error']['V']:.4f}, A={metrics['sign_error']['A']:.4f}, Avg={metrics['sign_error']['avg']:.4f}")
    
    return "\n".join(report)

def format_va_metrics(metrics):
    """格式化VA指标，生成可读的字符串报告"""
    report = []
    
    # 添加RMSE
    report.append(f"RMSE: V={metrics['rmse']['V']:.4f}, A={metrics['rmse']['A']:.4f}, Avg={metrics['rmse']['avg']:.4f}")
    
    # 添加MAE
    report.append(f"MAE: V={metrics['mae']['V']:.4f}, A={metrics['mae']['A']:.4f}, Avg={metrics['mae']['avg']:.4f}")
    
    # 添加CCC
    report.append(f"CCC: V={metrics['ccc']['V']:.4f}, A={metrics['ccc']['A']:.4f}, Avg={metrics['ccc']['avg']:.4f}")
    
    # 添加偏差
    report.append(f"偏差: V={metrics['bias']['V']:.4f}, A={metrics['bias']['A']:.4f}")
    
    return "\n".join(report)

def combine_metrics(va_metrics, quadrant_metrics):
    """合并VA和象限指标"""
    combined = {}
    
    # 添加象限准确率和加权F1
    combined['quadrant_accuracy'] = quadrant_metrics['accuracy']
    combined['weighted_f1'] = quadrant_metrics['weighted']['f1']
    
    # 添加CCC
    combined['ccc_v'] = va_metrics['ccc']['V']
    combined['ccc_a'] = va_metrics['ccc']['A']
    combined['ccc_avg'] = va_metrics['ccc']['avg']
    
    # 添加RMSE
    combined['rmse_v'] = va_metrics['rmse']['V']
    combined['rmse_a'] = va_metrics['rmse']['A']
    combined['rmse_avg'] = va_metrics['rmse']['avg']
    
    # 添加符号错误率
    combined['sign_error_v'] = quadrant_metrics['sign_error']['V']
    combined['sign_error_a'] = quadrant_metrics['sign_error']['A']
    
    return combined

def is_better_model(current_metrics, best_metrics, objective='combined'):
    """
    判断当前模型是否优于最佳模型
    
    参数:
        current_metrics: 当前模型的指标
        best_metrics: 最佳模型的指标
        objective: 评判标准，可选值:
                  'combined' - 象限准确率和RMSE的加权组合
                  'quadrant' - 仅象限准确率
                  'rmse' - 仅RMSE
                  'ccc' - 仅CCC
    
    返回:
        布尔值，指示当前模型是否更好
    """
    if best_metrics is None:
        return True
    
    if objective == 'quadrant':
        return current_metrics['quadrant_accuracy'] > best_metrics['quadrant_accuracy']
    
    elif objective == 'rmse':
        return current_metrics['rmse_avg'] < best_metrics['rmse_avg']
    
    elif objective == 'ccc':
        return current_metrics['ccc_avg'] > best_metrics['ccc_avg']
    
    elif objective == 'combined':
        # 象限准确率越高越好，RMSE越低越好
        # 对每个指标进行归一化，然后使用加权和
        quad_weight = 0.6
        rmse_weight = 0.4
        
        quad_improvement = (current_metrics['quadrant_accuracy'] - best_metrics['quadrant_accuracy']) / max(0.001, best_metrics['quadrant_accuracy'])
        rmse_improvement = (best_metrics['rmse_avg'] - current_metrics['rmse_avg']) / max(0.001, best_metrics['rmse_avg'])
        
        combined_improvement = quad_weight * quad_improvement + rmse_weight * rmse_improvement
        
        return combined_improvement > 0
    
    return False 