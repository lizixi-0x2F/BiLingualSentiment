"""
可视化工具模块 - 用于情感分析模型输出的可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional, Union
import os
import torch
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

def plot_va_scatter(predictions: np.ndarray, labels: Optional[np.ndarray] = None, 
                    texts: Optional[List[str]] = None, title: str = "Valence-Arousal分布图",
                    interactive: bool = False, output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    在效价-唤醒度二维空间中绘制预测点和真实标签的散点图
    
    Args:
        predictions: 形状为[N, 2]的预测值数组，第一列为效价，第二列为唤醒度
        labels: 可选，形状为[N, 2]的真实标签数组
        texts: 可选，文本列表用于交互式标签
        title: 图表标题
        interactive: 是否使用Plotly创建交互式图表
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    if interactive:
        # 创建交互式Plotly图表
        df = pd.DataFrame({
            'Valence_pred': predictions[:, 0],
            'Arousal_pred': predictions[:, 1],
            'Text': texts if texts is not None else [f"Sample {i}" for i in range(len(predictions))]
        })
        
        fig = px.scatter(df, x='Valence_pred', y='Arousal_pred', 
                         hover_data=['Text'], title=title,
                         labels={'Valence_pred': 'Valence (效价)', 'Arousal_pred': 'Arousal (唤醒度)'},
                         color_discrete_sequence=['blue'])
        
        # 添加四象限分界线
        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, 
                      line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, 
                      line=dict(color="gray", width=1, dash="dash"))
        
        # 如果有真实标签，也添加到图上
        if labels is not None:
            fig.add_trace(go.Scatter(
                x=labels[:, 0], y=labels[:, 1],
                mode='markers',
                marker=dict(color='red', size=8, opacity=0.5),
                name='真实标签'
            ))
            
            # 添加连接线以显示预测与真实值的偏差
            for i in range(len(predictions)):
                fig.add_shape(
                    type="line",
                    x0=predictions[i, 0], y0=predictions[i, 1],
                    x1=labels[i, 0], y1=labels[i, 1],
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=1)
                )
        
        # 调整布局
        fig.update_layout(
            xaxis=dict(range=[-1.1, 1.1], title='Valence (效价)'),
            yaxis=dict(range=[-1.1, 1.1], title='Arousal (唤醒度)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # 标注四个象限
        annotations = [
            dict(x=0.5, y=0.5, text="快乐/兴奋", showarrow=False, font=dict(size=12)),
            dict(x=-0.5, y=0.5, text="愤怒/焦虑", showarrow=False, font=dict(size=12)),
            dict(x=-0.5, y=-0.5, text="悲伤/抑郁", showarrow=False, font=dict(size=12)),
            dict(x=0.5, y=-0.5, text="满足/平静", showarrow=False, font=dict(size=12))
        ]
        fig.update_layout(annotations=annotations)
        
        # 保存或显示图表
        if output_path:
            fig.write_html(output_path)
            return None
        return fig
    
    else:
        # 创建Matplotlib静态图表
        plt.figure(figsize=(10, 8))
        
        # 绘制四象限分界线
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 绘制预测点
        plt.scatter(predictions[:, 0], predictions[:, 1], c='blue', alpha=0.7, label='预测值')
        
        # 如果有真实标签，也绘制出来
        if labels is not None:
            plt.scatter(labels[:, 0], labels[:, 1], c='red', alpha=0.5, label='真实标签')
            
            # 绘制连接线
            for i in range(len(predictions)):
                plt.plot([predictions[i, 0], labels[i, 0]], 
                         [predictions[i, 1], labels[i, 1]], 
                         'gray', alpha=0.3, linestyle='-')
        
        # 添加象限标签
        plt.text(0.5, 0.5, "快乐/兴奋", fontsize=12, ha='center')
        plt.text(-0.5, 0.5, "愤怒/焦虑", fontsize=12, ha='center')
        plt.text(-0.5, -0.5, "悲伤/抑郁", fontsize=12, ha='center')
        plt.text(0.5, -0.5, "满足/平静", fontsize=12, ha='center')
        
        # 设置轴范围和标签
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Valence (效价)')
        plt.ylabel('Arousal (唤醒度)')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 保存或显示图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        
        return plt.gcf()

def plot_prediction_errors(predictions: np.ndarray, labels: np.ndarray, 
                          title: str = "预测误差分布", 
                          output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    绘制预测误差的直方图
    
    Args:
        predictions: 形状为[N, 2]的预测值数组
        labels: 形状为[N, 2]的真实标签数组
        title: 图表标题
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    # 计算预测误差
    errors = np.abs(predictions - labels)
    
    # 创建双子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 效价误差分布
    sns.histplot(errors[:, 0], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('效价 (Valence) 预测误差分布')
    axes[0].set_xlabel('绝对误差')
    axes[0].set_ylabel('频率')
    
    # 唤醒度误差分布
    sns.histplot(errors[:, 1], kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('唤醒度 (Arousal) 预测误差分布')
    axes[1].set_xlabel('绝对误差')
    axes[1].set_ylabel('频率')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig

def plot_attention_heatmap(attention_weights: np.ndarray, tokens: List[str], 
                          title: str = "注意力热力图", 
                          output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    为Transformer模型的注意力权重创建热力图
    
    Args:
        attention_weights: 形状为[num_heads, seq_len, seq_len]的注意力权重
        tokens: 输入标记列表
        title: 图表标题
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    num_heads = attention_weights.shape[0]
    
    # 确定最佳的网格布局
    n_cols = min(3, num_heads)
    n_rows = (num_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 创建一个更好的热力图配色方案
    cmap = sns.color_palette("rocket", as_cmap=True)
    
    # 绘制每个头的注意力热力图
    for i in range(num_heads):
        if i < len(axes):
            # 选择当前的注意力头
            attn = attention_weights[i]
            
            # 保持序列长度manageable，最多显示前25个token
            max_tokens = min(25, len(tokens))
            if len(tokens) > max_tokens:
                tokens_display = tokens[:max_tokens-3] + ['...'] + tokens[-2:]
                attn_display = attn[:max_tokens-3, :max_tokens-3]
            else:
                tokens_display = tokens
                attn_display = attn
            
            # 绘制热力图
            sns.heatmap(attn_display, ax=axes[i], cmap=cmap, 
                        xticklabels=tokens_display, yticklabels=tokens_display,
                        cbar=i==0)  # 只在第一个图上显示颜色条
            
            axes[i].set_title(f'注意力头 #{i+1}')
    
    # 隐藏额外的子图
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig

def plot_training_progress(metrics: Dict[str, List[float]], 
                           title: str = "训练进度", 
                           output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    绘制训练过程中的损失和指标变化
    
    Args:
        metrics: 包含训练指标的字典，每个键对应一个指标，值为该指标在不同轮次的值列表
        title: 图表标题
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    num_metrics = len(metrics)
    
    # 创建足够的子图
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    # 为每个指标创建折线图
    for i, (metric_name, values) in enumerate(metrics.items()):
        epochs = list(range(1, len(values) + 1))
        axes[i].plot(epochs, values, marker='o', linestyle='-', linewidth=2)
        axes[i].set_title(f'{metric_name} 变化曲线')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig

def plot_feature_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                           texts: Optional[List[str]] = None,
                           title: str = "特征嵌入可视化",
                           interactive: bool = True,
                           output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    使用t-SNE降维并可视化高维特征嵌入
    
    Args:
        embeddings: 形状为[N, D]的特征嵌入数组，N为样本数，D为嵌入维度
        labels: 形状为[N, 2]的标签数组，包含效价和唤醒度
        texts: 可选，文本列表用于交互式标签
        title: 图表标题
        interactive: 是否使用Plotly创建交互式图表
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    # 使用t-SNE将高维嵌入减少到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//2))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 计算情感象限
    sentiment_quadrants = []
    for v, a in labels:
        if v >= 0 and a >= 0:
            sentiment_quadrants.append("快乐/兴奋")
        elif v < 0 and a >= 0:
            sentiment_quadrants.append("愤怒/焦虑")
        elif v < 0 and a < 0:
            sentiment_quadrants.append("悲伤/抑郁")
        else:
            sentiment_quadrants.append("满足/平静")
    
    if interactive:
        # 创建交互式Plotly图表
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Valence': labels[:, 0],
            'Arousal': labels[:, 1],
            'Sentiment': sentiment_quadrants,
            'Text': texts if texts is not None else [f"Sample {i}" for i in range(len(embeddings))]
        })
        
        fig = px.scatter(df, x='x', y='y', 
                         color='Sentiment', hover_data=['Text', 'Valence', 'Arousal'],
                         title=title, color_discrete_sequence=px.colors.qualitative.Set1)
        
        # 调整布局
        fig.update_layout(
            xaxis_title="t-SNE维度1",
            yaxis_title="t-SNE维度2",
            legend_title="情感象限"
        )
        
        # 保存或显示图表
        if output_path:
            fig.write_html(output_path)
            return None
        return fig
    
    else:
        # 创建Matplotlib静态图表
        plt.figure(figsize=(12, 10))
        
        # 为不同象限的点设置不同的颜色
        unique_sentiments = list(set(sentiment_quadrants))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sentiments)))
        
        for i, sentiment in enumerate(unique_sentiments):
            mask = np.array(sentiment_quadrants) == sentiment
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=sentiment, alpha=0.7)
        
        plt.title(title)
        plt.xlabel("t-SNE维度1")
        plt.ylabel("t-SNE维度2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 保存或显示图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        
        return plt.gcf()

def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                         title: str = "情感分类混淆矩阵",
                         output_path: Optional[str] = None) -> Union[Figure, None]:
    """
    创建情感象限的混淆矩阵
    
    Args:
        predictions: 形状为[N, 2]的预测值数组
        labels: 形状为[N, 2]的真实标签数组
        title: 图表标题
        output_path: 输出路径，如果指定则保存图表
        
    Returns:
        如果不保存图表则返回Figure对象，否则返回None
    """
    # 定义象限类别
    quadrants = ["快乐/兴奋", "愤怒/焦虑", "悲伤/抑郁", "满足/平静"]
    
    # 函数将VA值转换为象限索引
    def va_to_quadrant(va):
        v, a = va
        if v >= 0 and a >= 0:
            return 0  # 快乐/兴奋
        elif v < 0 and a >= 0:
            return 1  # 愤怒/焦虑
        elif v < 0 and a < 0:
            return 2  # 悲伤/抑郁
        else:  # v >= 0 and a < 0
            return 3  # 满足/平静
    
    # 转换预测和标签为象限
    pred_quadrants = np.array([va_to_quadrant(p) for p in predictions])
    true_quadrants = np.array([va_to_quadrant(l) for l in labels])
    
    # 计算混淆矩阵
    cm = np.zeros((4, 4), dtype=int)
    for i in range(len(pred_quadrants)):
        cm[true_quadrants[i], pred_quadrants[i]] += 1
    
    # 创建混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=quadrants, yticklabels=quadrants)
    plt.xlabel('预测象限')
    plt.ylabel('真实象限')
    plt.title(title)
    
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return plt.gcf()

def get_html_visualization(predictions: np.ndarray, text: str, 
                           remove_special_tokens: bool = True) -> str:
    """
    为单个文本的预测生成HTML可视化
    
    Args:
        predictions: 形状为[2]的预测数组，包含效价和唤醒度
        text: 输入文本
        remove_special_tokens: 是否移除特殊token如[CLS], [SEP]等
        
    Returns:
        HTML格式的可视化字符串
    """
    valence, arousal = predictions
    
    # 确定情感象限和对应的颜色
    if valence >= 0 and arousal >= 0:
        quadrant = "快乐/兴奋"
        color = "#FF9933"  # 橙色
    elif valence < 0 and arousal >= 0:
        quadrant = "愤怒/焦虑"
        color = "#CC0000"  # 红色
    elif valence < 0 and arousal < 0:
        quadrant = "悲伤/抑郁"
        color = "#3366CC"  # 蓝色
    else:  # valence >= 0 and arousal < 0
        quadrant = "满足/平静"
        color = "#33CC33"  # 绿色
    
    # 创建情感坐标图
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.scatter(valence, arousal, c=color, s=100, marker='o')
    
    # 添加象限标签
    ax.text(0.5, 0.5, "快乐/兴奋", fontsize=10, ha='center')
    ax.text(-0.5, 0.5, "愤怒/焦虑", fontsize=10, ha='center')
    ax.text(-0.5, -0.5, "悲伤/抑郁", fontsize=10, ha='center')
    ax.text(0.5, -0.5, "满足/平静", fontsize=10, ha='center')
    
    ax.set_xlabel('Valence (效价)')
    ax.set_ylabel('Arousal (唤醒度)')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 转换图像为base64编码的字符串
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('ascii')
    
    # 生成HTML
    html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; max-width: 800px;">
        <h2 style="color: #333;">情感分析结果</h2>
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <p><strong>输入文本:</strong> {text}</p>
        </div>
        <div style="display: flex; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <img src="data:image/png;base64,{img_str}" alt="情感坐标图" style="width: 100%; max-width: 400px;">
            </div>
            <div style="flex: 1; min-width: 300px; padding: 15px;">
                <h3>预测结果</h3>
                <p><strong>情感象限:</strong> <span style="color: {color};">{quadrant}</span></p>
                <p><strong>效价 (Valence):</strong> {valence:.4f}</p>
                <p><strong>唤醒度 (Arousal):</strong> {arousal:.4f}</p>
                <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
                    注：效价表示情感的正负性，范围[-1,1]，值越大越正面。<br>
                    唤醒度表示情感的强度，范围[-1,1]，值越大表示情感越强烈。
                </p>
            </div>
        </div>
    </div>
    """
    
    return html
