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

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.data.dataset import load_and_prepare_data, create_dataloaders
from src.utils.metrics import evaluate_predictions
from src.utils.training import GradNorm, FGM, WarmupCosineScheduler, compute_kl_loss

def numpy_to_python_type(obj):
    """将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_type(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python_type(item) for item in obj)
    else:
        return obj

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
    file_handler = logging.FileHandler(os.path.join(log_dir, f"distillation_{time.strftime('%Y%m%d_%H%M%S')}.log"))
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
    torch.save(state, checkpoint_path, pickle_protocol=4)
    
    # 保存最佳模型
    if epoch == best_val_ccc[0]:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_model_path, pickle_protocol=4)

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    计算知识蒸馏损失
    
    Args:
        student_logits: 学生模型的输出
        teacher_logits: 教师模型的输出
        labels: 真实标签
        temperature: 温度参数
        alpha: 软目标和硬目标的权重系数
        
    Returns:
        loss: 蒸馏损失
    """
    # 硬目标损失（MSE损失）
    hard_loss = F.mse_loss(student_logits, labels)
    
    # 软目标损失（KL散度）
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 加权结合
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return loss

def train_student(config_path):
    """训练学生模型（知识蒸馏）"""
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
    
    # 加载教师模型
    logger.info("加载教师模型...")
    teacher_model_path = config["distillation"]["teacher_model_path"]
    
    # 加载教师模型状态
    teacher_state = torch.load(teacher_model_path, map_location=device, weights_only=False)
    teacher_config = teacher_state["config"]
    
    # 创建教师模型实例
    teacher_model = TeacherModel(
        base_model_name=teacher_config["model"]["base_model_name"],
        ltc_hidden_size=teacher_config["model"]["ltc_hidden_size"],
        ltc_memory_size=teacher_config["model"]["ltc_memory_size"],
        ltc_num_layers=teacher_config["model"]["ltc_num_layers"],
        ltc_dropout=teacher_config["model"]["ltc_dropout"],
        output_dim=teacher_config["model"]["output_dim"]
    )
    
    # 加载预训练权重
    teacher_model.load_state_dict(teacher_state["model"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 设置为评估模式
    
    # 初始化学生模型
    logger.info("初始化学生模型...")
    student_model = StudentModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        ltc_hidden_size=config["model"]["ltc_hidden_size"],
        ltc_num_layers=config["model"]["ltc_num_layers"],
        ltc_dropout=config["model"]["ltc_dropout"],
        output_dim=config["model"]["output_dim"]
    )
    
    student_model = student_model.to(device)
    
    # 初始化优化器
    optimizer = AdamW(
        student_model.parameters(),
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
    
    # 初始化FGM对抗训练
    if config["regularization"]["use_fgm"]:
        fgm = FGM(
            student_model,
            epsilon=config["regularization"]["fgm_epsilon"],
            emb_name='transformer.embeddings'
        )
    
    # 训练循环
    best_val_ccc = (0, 0)  # (epoch, ccc)
    global_step = 0
    
    # 蒸馏参数
    temperature = config["distillation"]["temperature"]
    alpha = config["distillation"]["alpha"]
    soft_target_loss_weight = config["distillation"]["soft_target_loss_weight"]
    hard_target_loss_weight = config["distillation"]["hard_target_loss_weight"]
    
    # MSE损失用于硬目标
    mse_loss = nn.MSELoss()
    
    logger.info("开始知识蒸馏训练...")
    for epoch in range(config["training"]["max_epochs"]):
        student_model.train()
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
            
            # 前向传播学生模型
            student_logits = student_model(input_ids, attention_mask)
            
            # 获取教师模型的预测（无梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
            
            # 硬目标损失（MSE损失）
            hard_loss = mse_loss(student_logits, labels)
            
            # 软目标损失 - 使用KL散度
            # 对于回归任务，我们可以把logits看作是分布的参数
            soft_loss = F.mse_loss(student_logits, teacher_logits)
            
            # 加权组合损失
            loss = hard_target_loss_weight * hard_loss + soft_target_loss_weight * soft_loss
            
            # 反向传播
            loss.backward()
            
            # 应用FGM对抗训练
            if config["regularization"]["use_fgm"]:
                # 保存当前梯度
                fgm.attack()
                
                # 对抗前向传播
                adv_student_logits = student_model(input_ids, attention_mask)
                
                # 对抗损失计算
                adv_hard_loss = mse_loss(adv_student_logits, labels)
                adv_soft_loss = F.mse_loss(adv_student_logits, teacher_logits)
                adv_loss = hard_target_loss_weight * adv_hard_loss + soft_target_loss_weight * adv_soft_loss
                
                # 反向传播对抗损失，累积梯度
                adv_loss.backward()
                
                # 恢复嵌入
                fgm.restore()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            curr_lr = scheduler.step(global_step)
            
            # 更新进度条
            train_iterator.set_postfix({
                'loss': loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': soft_loss.item(),
                'lr': curr_lr
            })
            
            # 累计损失
            train_loss += loss.item()
            
            # 记录到TensorBoard
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/hard_loss', hard_loss.item(), global_step)
            writer.add_scalar('train/soft_loss', soft_loss.item(), global_step)
            writer.add_scalar('train/learning_rate', curr_lr, global_step)
        
        # 计算该epoch的平均损失
        avg_train_loss = train_loss / len(dataloaders["train"])
        
        # 评估模型在验证集上的性能
        val_metrics = evaluate_student(student_model, dataloaders["val"], device, config)
        
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
            student_model, optimizer, scheduler, epoch + 1, best_val_ccc, global_step, config, save_dir
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
    state = torch.load(best_model_path, weights_only=False)
    student_model.load_state_dict(state["model"])
    
    # 评估测试集
    test_metrics = evaluate_student(student_model, dataloaders["test"], device, config)
    
    logger.info(f"学生模型测试集性能 - 平均CCC: {test_metrics['avg_ccc']:.4f} "
               f"(价效: {test_metrics['valence_ccc']:.4f}, 唤起: {test_metrics['arousal_ccc']:.4f})")
    
    # 同时评估教师模型
    teacher_test_metrics = evaluate_student(teacher_model, dataloaders["test"], device, config)
    
    logger.info(f"教师模型测试集性能 - 平均CCC: {teacher_test_metrics['avg_ccc']:.4f} "
               f"(价效: {teacher_test_metrics['valence_ccc']:.4f}, 唤起: {teacher_test_metrics['arousal_ccc']:.4f})")
    
    # 计算知识蒸馏效率
    ccc_ratio = test_metrics['avg_ccc'] / teacher_test_metrics['avg_ccc'] * 100
    logger.info(f"知识蒸馏效率: 学生/教师 = {ccc_ratio:.2f}%")
    
    # 将测试结果写入文件 - 确保将NumPy类型转换为Python原生类型
    results = {
        "student": numpy_to_python_type(test_metrics),
        "teacher": numpy_to_python_type(teacher_test_metrics),
        "distillation_efficiency": {
            "ccc_ratio": float(ccc_ratio)
        }
    }
    
    results_path = os.path.join(save_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return student_model, test_metrics

def evaluate_student(model, dataloader, device, config):
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
    parser = argparse.ArgumentParser(description="训练学生模型 (Mini Transformer + LTC_NCP)")
    parser.add_argument("--config", type=str, default="configs/student_config.json", help="配置文件路径")
    args = parser.parse_args()
    
    train_student(args.config) 