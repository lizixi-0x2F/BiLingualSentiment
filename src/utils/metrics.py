import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

def concordance_correlation_coefficient(y_true, y_pred):
    """
    计算一致性相关系数 (CCC)
    CCC = 2 * cov(y_true, y_pred) / [var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2]
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        ccc: 一致性相关系数
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    
    covariance = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    numerator = 2 * covariance
    denominator = y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2
    
    # 避免除以零
    if denominator == 0:
        return 0
    
    ccc = numerator / denominator
    
    return ccc

def evaluate_predictions(true_valence, pred_valence, true_arousal, pred_arousal):
    """
    评估价效和唤起的预测性能
    
    Args:
        true_valence: 真实价效值
        pred_valence: 预测价效值
        true_arousal: 真实唤起值
        pred_arousal: 预测唤起值
        
    Returns:
        metrics: 包含多种评估指标的字典
    """
    # 确保输入为NumPy数组，而非Python列表
    true_valence = np.array(true_valence)
    pred_valence = np.array(pred_valence)
    true_arousal = np.array(true_arousal)
    pred_arousal = np.array(pred_arousal)
    
    # 计算CCC
    valence_ccc = concordance_correlation_coefficient(true_valence, pred_valence)
    arousal_ccc = concordance_correlation_coefficient(true_arousal, pred_arousal)
    avg_ccc = (valence_ccc + arousal_ccc) / 2
    
    # 计算皮尔逊相关系数
    valence_pearson, _ = pearsonr(true_valence, pred_valence)
    arousal_pearson, _ = pearsonr(true_arousal, pred_arousal)
    avg_pearson = (valence_pearson + arousal_pearson) / 2
    
    # 计算斯皮尔曼相关系数
    valence_spearman, _ = spearmanr(true_valence, pred_valence)
    arousal_spearman, _ = spearmanr(true_arousal, pred_arousal)
    avg_spearman = (valence_spearman + arousal_spearman) / 2
    
    # 计算MSE
    valence_mse = np.mean((true_valence - pred_valence) ** 2)
    arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)
    avg_mse = (valence_mse + arousal_mse) / 2
    
    # 汇总指标
    metrics = {
        "valence_ccc": valence_ccc,
        "arousal_ccc": arousal_ccc,
        "avg_ccc": avg_ccc,
        "valence_pearson": valence_pearson,
        "arousal_pearson": arousal_pearson,
        "avg_pearson": avg_pearson,
        "valence_spearman": valence_spearman,
        "arousal_spearman": arousal_spearman,
        "avg_spearman": avg_spearman,
        "valence_mse": valence_mse,
        "arousal_mse": arousal_mse,
        "avg_mse": avg_mse
    }
    
    return metrics 