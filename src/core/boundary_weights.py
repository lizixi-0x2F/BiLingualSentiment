#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
边界样本权重计算
为情感VA空间中的边界和困难样本分配更高权重
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class BoundarySampleWeighter:
    """
    边界样本权重计算器
    
    为VA空间中的边界样本和困难样本分配更高权重，
    提高模型对边界区域的鉴别能力
    """
    
    def __init__(
        self,
        boundary_threshold: float = 0.15,  # 边界阈值（距离坐标轴的距离）
        weight_multiplier: float = 2.5,    # 边界样本权重倍增
        quadrant_borders_bonus: float = 1.5,  # 象限边界额外权重
        hard_sample_percentile: float = 90.0,  # 困难样本百分位
        hard_sample_bonus: float = 2.0,    # 困难样本额外权重
        use_dynamic_adjustment: bool = True,  # 是否根据训练损失动态调整
        device: str = "cuda"
    ):
        """
        初始化边界样本权重计算器
        
        参数:
            boundary_threshold: 边界区域阈值
            weight_multiplier: 边界样本权重倍增
            quadrant_borders_bonus: 象限边界额外权重
            hard_sample_percentile: 困难样本百分位阈值
            hard_sample_bonus: 困难样本权重加成
            use_dynamic_adjustment: 是否根据训练动态调整权重
            device: 计算设备
        """
        self.boundary_threshold = boundary_threshold
        self.weight_multiplier = weight_multiplier
        self.quadrant_borders_bonus = quadrant_borders_bonus
        self.hard_sample_percentile = hard_sample_percentile
        self.hard_sample_bonus = hard_sample_bonus
        self.use_dynamic_adjustment = use_dynamic_adjustment
        self.device = device
        
        # 历史损失记录，用于识别困难样本
        self.sample_losses = {}
        self.epoch_losses = []
        self.current_epoch = 0
        
        # 全局样本计数器
        self.total_samples = 0
        self.boundary_samples = 0
        
        logger.info(f"初始化边界样本权重计算器: threshold={boundary_threshold}, "
                   f"multiplier={weight_multiplier}, 象限边界加成={quadrant_borders_bonus}")
    
    def compute_sample_weights(
        self, 
        v_values: torch.Tensor, 
        a_values: torch.Tensor,
        sample_ids: Optional[List[int]] = None,
        losses: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算样本权重
        
        参数:
            v_values: 价值(Valence)标签值
            a_values: 效度(Arousal)标签值
            sample_ids: 样本ID列表（用于跟踪困难样本）
            losses: 样本当前损失，用于标识困难样本
        
        返回:
            样本权重向量
        """
        batch_size = v_values.size(0)
        weights = torch.ones(batch_size, device=self.device)
        
        # 更新样本统计信息
        self.total_samples += batch_size
        
        # 1. 计算边界样本权重 - 接近坐标轴的样本
        v_near_zero = torch.abs(v_values) < self.boundary_threshold
        a_near_zero = torch.abs(a_values) < self.boundary_threshold
        
        # 任一维度接近坐标轴的样本被认为是边界样本
        axis_boundary_samples = v_near_zero | a_near_zero
        weights[axis_boundary_samples] *= self.weight_multiplier
        
        # 统计边界样本数量
        self.boundary_samples += torch.sum(axis_boundary_samples).item()
        
        # 2. 计算象限边界权重 - 靠近象限分界线的样本
        near_quadrant_boundary = (
            (torch.abs(v_values - self.boundary_threshold) < 0.05) | 
            (torch.abs(a_values - self.boundary_threshold) < 0.05)
        )
        weights[near_quadrant_boundary] *= self.quadrant_borders_bonus
        
        # 3. 困难样本权重 - 如果提供了损失和样本ID
        if losses is not None and sample_ids is not None:
            # 记录每个样本的损失
            for i, sample_id in enumerate(sample_ids):
                if sample_id in self.sample_losses:
                    # 使用指数移动平均更新损失
                    self.sample_losses[sample_id] = 0.7 * self.sample_losses[sample_id] + 0.3 * losses[i].item()
                else:
                    # 新样本
                    self.sample_losses[sample_id] = losses[i].item()
            
            # 识别困难样本 - 损失高于阈值的样本
            if len(self.sample_losses) > 20:  # 等待足够多的样本
                # 计算损失阈值（高百分位）
                loss_threshold = np.percentile(list(self.sample_losses.values()), 
                                              self.hard_sample_percentile)
                
                # 将高损失样本标记为困难样本
                for i, sample_id in enumerate(sample_ids):
                    if sample_id in self.sample_losses and self.sample_losses[sample_id] > loss_threshold:
                        # 为困难样本增加权重
                        weights[i] *= self.hard_sample_bonus
        
        return weights
    
    def update_epoch(self, epoch: int, mean_loss: float):
        """
        更新轮次信息
        
        参数:
            epoch: 当前轮次
            mean_loss: 平均损失
        """
        self.current_epoch = epoch
        self.epoch_losses.append(mean_loss)
        
        # 定期清理过期样本损失记录
        if epoch > 0 and epoch % 5 == 0:
            # 选择保留损失较大的前50%样本
            if len(self.sample_losses) > 1000:
                sorted_losses = sorted(self.sample_losses.items(), key=lambda x: x[1], reverse=True)
                self.sample_losses = dict(sorted_losses[:len(sorted_losses)//2])
                logger.info(f"清理样本损失记录，保留{len(self.sample_losses)}个样本")
        
        # 打印边界样本统计
        if self.total_samples > 0:
            boundary_ratio = self.boundary_samples / self.total_samples
            logger.info(f"轮次{epoch}边界样本比例: {boundary_ratio:.4f}, 总样本数: {self.total_samples}")
        
        # 根据训练进度动态调整参数
        if self.use_dynamic_adjustment and epoch > 5:
            # 如果损失下降变慢，增加边界样本权重
            if len(self.epoch_losses) >= 3:
                recent_improvement = self.epoch_losses[-3] - self.epoch_losses[-1]
                if recent_improvement < 0.01:
                    # 损失下降减缓，增加边界样本权重
                    self.weight_multiplier *= 1.1
                    self.quadrant_borders_bonus *= 1.1
                    logger.info(f"训练进度减缓，增加边界样本权重: {self.weight_multiplier:.2f}, 象限边界权重: {self.quadrant_borders_bonus:.2f}")
                elif recent_improvement > 0.1 and self.weight_multiplier > 2.0:
                    # 损失下降较快，可适度减小权重
                    self.weight_multiplier /= 1.05
                    self.quadrant_borders_bonus /= 1.05
                    logger.info(f"训练进度良好，调整边界样本权重: {self.weight_multiplier:.2f}, 象限边界权重: {self.quadrant_borders_bonus:.2f}")
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取边界样本统计数据
        
        返回:
            包含统计数据的字典
        """
        stats = {
            "boundary_threshold": self.boundary_threshold,
            "weight_multiplier": self.weight_multiplier,
            "quadrant_borders_bonus": self.quadrant_borders_bonus,
            "total_samples": self.total_samples,
            "boundary_sample_ratio": self.boundary_samples / max(1, self.total_samples)
        }
        
        if len(self.epoch_losses) > 0:
            stats["latest_epoch_loss"] = self.epoch_losses[-1]
        
        return stats 