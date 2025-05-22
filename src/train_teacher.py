import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from transformers import XLMRobertaTokenizer
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import random
import time
from pathlib import Path

from models.teacher_model import TeacherModel
from data.dataset import load_and_prepare_data, create_dataloaders
from utils.metrics import evaluate_predictions
from utils.training import GradNorm, FGM, WarmupCosineScheduler, compute_kl_loss

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def setup_logging(save_dir):
    """设置日志记录"""
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理程序
    file_handler = logging.FileHandler(os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    file_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_dir

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_ccc, step, config, checkpoint_dir):
    """保存模型检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_ccc": best_val_ccc,
        "step": step,
        "config": config
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(state, checkpoint_path)
    
    # 保存最佳模型
    if epoch == best_val_ccc[0]:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_model_path)

def train_teacher(config_path):
    """训练教师模型"""
    # 加载配置
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 创建输出目录
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志记录
    logger, log_dir = setup_logging(save_dir)
    logger.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 设置随机种子
    set_seed(config["training"]["seed"])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化分词器
    logger.info("加载本地XLM-R分词器...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("XLM-R")
    
    # 加载数据集
    logger.info("加载和准备数据集...")
    datasets = load_and_prepare_data(
        config["data"]["chinese_va_data_path"],
        config["data"]["english_va_data_path"],
        tokenizer,
        max_length=config["data"]["max_length"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["training"]["seed"]
    )
    
    # 创建数据加载器
    dataloaders = create_dataloaders(
        datasets,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )
    
    logger.info(f"训练集大小: {len(datasets['train'])}")
    logger.info(f"验证集大小: {len(datasets['val'])}")
    logger.info(f"测试集大小: {len(datasets['test'])}")
    
    # 初始化模型
    logger.info("初始化教师模型...")
    model = TeacherModel(
        base_model_name=config["model"]["base_model_name"],
        ltc_hidden_size=config["model"]["ltc_hidden_size"],
        ltc_num_layers=config["model"]["ltc_num_layers"],
        ltc_dropout=config["model"]["ltc_dropout"],
        output_dim=config["model"]["output_dim"]
    )
    
    model = model.to(device)
    
    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # 计算总训练步数
    total_steps = len(dataloaders["train"]) * config["training"]["max_epochs"]
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    
    # 初始化学习率调度器
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # 初始化GradNorm
    if config["regularization"]["use_gradnorm"]:
        gradnorm = GradNorm(model, initial_weights=[0.5, 0.5])
        # 确保优化器在正确的设备上
        gradnorm_optimizer = torch.optim.Adam([gradnorm.weights], lr=0.01)
        gradnorm.set_optimizer(gradnorm_optimizer)
    
    # 初始化FGM对抗训练
    if config["regularization"]["use_fgm"]:
        fgm = FGM(
            model,
            epsilon=config["regularization"]["fgm_epsilon"],
            emb_name='roberta.embeddings'
        )
    
    # 损失函数
    mse_loss = nn.MSELoss()
    
    # 训练循环
    best_val_ccc = (0, 0)  # (epoch, ccc)
    global_step = 0
    
    logger.info("开始训练...")
    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0
        
        # 进度条
        train_iterator = tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{config['training']['max_epochs']}")
        
        # 训练一个epoch
        for step, batch in enumerate(train_iterator):
            global_step += 1
            
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 计算价效和唤起损失
            valence_loss = mse_loss(logits[:, 0], labels[:, 0])
            arousal_loss = mse_loss(logits[:, 1], labels[:, 1])
            
            # 准备基础损失
            losses = [valence_loss, arousal_loss]
            
            # 应用R-Drop正则化
            if config["regularization"]["use_rdrop"]:
                # 第二次前向传播
                logits2 = model(input_ids, attention_mask)
                
                # 计算KL散度损失
                kl_loss = compute_kl_loss(logits, logits2)
                
                # 将R-Drop损失添加到第一个任务损失中
                rdrop_alpha = config["regularization"]["rdrop_alpha"]
                losses[0] = losses[0] + rdrop_alpha * kl_loss
            
            # 使用GradNorm自动平衡多任务损失
            if config["regularization"]["use_gradnorm"]:
                # 获取任务权重
                task_weights = gradnorm.get_weights(losses)
                # 手动计算加权损失
                weighted_loss = sum([task_weights[i] * losses[i] for i in range(len(losses))])
            else:
                # 总损失
                weighted_loss = sum(losses)
                
            # 执行反向传播
            weighted_loss.backward()
            
            # 如果使用GradNorm，在反向传播后更新梯度范数估计
            if config["regularization"]["use_gradnorm"]:
                gradnorm.update_norms(losses, model.parameters())
            
            # 应用FGM对抗训练
            if config["regularization"]["use_fgm"]:
                # 保存当前梯度
                fgm.attack()
                
                # 对抗前向传播
                adv_logits = model(input_ids, attention_mask)
                
                # 对抗损失
                adv_valence_loss = mse_loss(adv_logits[:, 0], labels[:, 0])
                adv_arousal_loss = mse_loss(adv_logits[:, 1], labels[:, 1])
                adv_losses = [adv_valence_loss, adv_arousal_loss]
                
                # 计算对抗损失并反向传播
                if config["regularization"]["use_gradnorm"]:
                    # 使用已经计算好的权重
                    adv_loss = sum([task_weights[i] * adv_losses[i] for i in range(len(adv_losses))])
                else:
                    adv_loss = sum(adv_losses)
                
                # 反向传播对抗损失，累积梯度
                adv_loss.backward()
                
                # 恢复嵌入
                fgm.restore()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            curr_lr = scheduler.step(global_step)
            
            # 更新进度条
            post_dict = {
                'loss': weighted_loss.item(),
                'valence_loss': valence_loss.item(),
                'arousal_loss': arousal_loss.item(),
                'lr': curr_lr
            }
            
            # 添加额外信息到进度条
            if config["regularization"]["use_rdrop"]:
                post_dict['kl_loss'] = kl_loss.item()
            if config["regularization"]["use_gradnorm"]:
                post_dict['v_weight'] = task_weights[0].item()
                post_dict['a_weight'] = task_weights[1].item()
                
            train_iterator.set_postfix(post_dict)
            
            # 累计损失
            train_loss += weighted_loss.item()
            
            # 记录到TensorBoard
            writer.add_scalar('train/weighted_loss', weighted_loss.item(), global_step)
            writer.add_scalar('train/valence_loss', valence_loss.item(), global_step)
            writer.add_scalar('train/arousal_loss', arousal_loss.item(), global_step)
            writer.add_scalar('train/learning_rate', curr_lr, global_step)
        
        # 计算该epoch的平均损失
        avg_train_loss = train_loss / len(dataloaders["train"])
        
        # 评估模型在验证集上的性能
        val_metrics = evaluate_teacher(model, dataloaders["val"], device, config)
        
        # 记录验证指标
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']} - "
                   f"训练损失: {avg_train_loss:.4f}, "
                   f"验证CCC: {val_metrics['avg_ccc']:.4f} "
                   f"(价效: {val_metrics['valence_ccc']:.4f}, 唤起: {val_metrics['arousal_ccc']:.4f})")
        
        # 记录到TensorBoard
        writer.add_scalar('val/avg_ccc', val_metrics['avg_ccc'], epoch + 1)
        writer.add_scalar('val/valence_ccc', val_metrics['valence_ccc'], epoch + 1)
        writer.add_scalar('val/arousal_ccc', val_metrics['arousal_ccc'], epoch + 1)
        
        # 检查是否是最佳模型
        if val_metrics['avg_ccc'] > best_val_ccc[1]:
            best_val_ccc = (epoch + 1, val_metrics['avg_ccc'])
            logger.info(f"新的最佳模型! Epoch: {epoch+1}, 验证CCC: {val_metrics['avg_ccc']:.4f}")
        
        # 保存检查点
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_val_ccc, global_step, config, save_dir
        )
        
        # 提前停止
        if epoch + 1 - best_val_ccc[0] >= config["training"]["early_stopping_patience"]:
            logger.info(f"提前停止! 自从 Epoch {best_val_ccc[0]} 以来没有改进")
            break
    
    # 训练结束
    logger.info(f"训练完成! 最佳验证CCC: {best_val_ccc[1]:.4f} (Epoch {best_val_ccc[0]})")
    
    # 在测试集上评估最佳模型
    logger.info("在测试集上评估最佳模型...")
    
    # 加载最佳模型
    best_model_path = os.path.join(save_dir, "best_model.pt")
    state = torch.load(best_model_path)
    model.load_state_dict(state["model"])
    
    # 评估测试集
    test_metrics = evaluate_teacher(model, dataloaders["test"], device, config)
    
    logger.info(f"测试集性能 - 平均CCC: {test_metrics['avg_ccc']:.4f} "
               f"(价效: {test_metrics['valence_ccc']:.4f}, 唤起: {test_metrics['arousal_ccc']:.4f})")
    
    # 检查是否达到了目标阈值
    ccc_threshold = 0.7  # 目标CCC阈值为70%
    if test_metrics['avg_ccc'] >= ccc_threshold:
        logger.info(f"✓ 模型达到了目标CCC阈值 ({test_metrics['avg_ccc']:.4f} >= {ccc_threshold:.4f})")
    else:
        logger.warning(f"✗ 模型未达到目标CCC阈值 ({test_metrics['avg_ccc']:.4f} < {ccc_threshold:.4f})")
    
    # 将测试结果写入文件
    results_path = os.path.join(save_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model, test_metrics

def evaluate_teacher(model, dataloader, device, config):
    """在数据集上评估模型"""
    model.eval()
    
    all_preds_valence = []
    all_preds_arousal = []
    all_labels_valence = []
    all_labels_arousal = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 收集预测和标签
            all_preds_valence.extend(logits[:, 0].cpu().numpy())
            all_preds_arousal.extend(logits[:, 1].cpu().numpy())
            all_labels_valence.extend(labels[:, 0].cpu().numpy())
            all_labels_arousal.extend(labels[:, 1].cpu().numpy())
    
    # 计算指标
    metrics = evaluate_predictions(
        all_labels_valence, all_preds_valence, 
        all_labels_arousal, all_preds_arousal
    )
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练教师模型 (XLM-R-Base + LTC_NCP)")
    parser.add_argument("--config", type=str, default="configs/teacher_config.json", help="配置文件路径")
    args = parser.parse_args()
    
    train_teacher(args.config) 