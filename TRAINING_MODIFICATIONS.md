# LTC-NCP-VA 训练代码修改方案

为解决当前情感识别模型存在的问题，特别是"情感判断反转"和"CCC高但效果差"的现象，以下是针对训练代码的具体修改建议：

## 1. 情感标签处理修改

### 1.1 添加检查训练数据标签一致性的函数

在`train.py`中添加：

```python
def check_emotion_label_consistency(df, v_col, a_col, text_col=None):
    """
    检查情感标签的一致性
    
    参数:
        df: 数据框
        v_col: 价值列名
        a_col: 效度列名
        text_col: 文本列名(可选，用于示例)
    
    返回:
        问题数据统计字典
    """
    stats = {
        "total_samples": len(df),
        "v_range": [df[v_col].min(), df[v_col].max()],
        "a_range": [df[a_col].min(), df[a_col].max()],
        "v_mean": df[v_col].mean(),
        "a_mean": df[a_col].mean(),
        "extreme_pos_v": sum(df[v_col] > 0.8),
        "extreme_neg_v": sum(df[v_col] < -0.8),
        "extreme_pos_a": sum(df[a_col] > 0.8),
        "extreme_neg_a": sum(df[a_col] < -0.8),
        "neutral_samples": sum((abs(df[v_col]) < 0.2) & (abs(df[a_col]) < 0.2))
    }
    
    # 象限分布统计
    stats["q1_count"] = sum((df[v_col] > 0) & (df[a_col] > 0))  # 喜悦/兴奋
    stats["q2_count"] = sum((df[v_col] > 0) & (df[a_col] < 0))  # 满足/平静
    stats["q3_count"] = sum((df[v_col] < 0) & (df[a_col] > 0))  # 愤怒/焦虑
    stats["q4_count"] = sum((df[v_col] < 0) & (df[a_col] < 0))  # 悲伤/抑郁
    
    # 检查不一致样本 (示例：带有"开心"的文本但V值为负)
    if text_col is not None:
        positive_keywords = ["开心", "高兴", "喜欢", "happy", "joy", "喜悦", "满意", "满足"]
        negative_keywords = ["难过", "伤心", "生气", "愤怒", "悲伤", "讨厌", "失望", "焦虑", "sad", "angry", "fear"]
        
        potential_issues = []
        
        # 检查带正面词但V为负的样本
        for keyword in positive_keywords:
            mask = df[text_col].str.contains(keyword, na=False) & (df[v_col] < -0.3)
            if sum(mask) > 0:
                samples = df[mask].sample(min(3, sum(mask)))
                for _, row in samples.iterrows():
                    potential_issues.append({
                        "text": row[text_col],
                        "v": row[v_col],
                        "a": row[a_col],
                        "issue": f"包含正面词'{keyword}'但V值为负"
                    })
        
        # 检查带负面词但V为正的样本
        for keyword in negative_keywords:
            mask = df[text_col].str.contains(keyword, na=False) & (df[v_col] > 0.3)
            if sum(mask) > 0:
                samples = df[mask].sample(min(3, sum(mask)))
                for _, row in samples.iterrows():
                    potential_issues.append({
                        "text": row[text_col],
                        "v": row[v_col],
                        "a": row[a_col],
                        "issue": f"包含负面词'{keyword}'但V值为正"
                    })
        
        stats["potential_issues"] = potential_issues
    
    return stats
```

### 1.2 在数据加载前检查数据

修改`prepare_data`函数，在数据集创建前添加标签一致性检查：

```python
def prepare_data(config):
    # 原有代码...
    
    # 加载数据集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # 检查训练集标签一致性
    logger.info("检查训练集标签一致性...")
    train_stats = check_emotion_label_consistency(
        train_df, 
        config['data']['v_col'], 
        config['data']['a_col'],
        config['data']['text_col']
    )
    
    # 输出统计信息
    logger.info(f"训练集大小: {train_stats['total_samples']}")
    logger.info(f"价值范围: {train_stats['v_range']}, 平均值: {train_stats['v_mean']:.3f}")
    logger.info(f"效度范围: {train_stats['a_range']}, 平均值: {train_stats['a_mean']:.3f}")
    logger.info(f"象限分布: Q1(喜悦)={train_stats['q1_count']}, Q2(满足)={train_stats['q2_count']}, Q3(愤怒)={train_stats['q3_count']}, Q4(悲伤)={train_stats['q4_count']}")
    
    # 检查并报告潜在问题样本
    if 'potential_issues' in train_stats and len(train_stats['potential_issues']) > 0:
        logger.warning(f"发现 {len(train_stats['potential_issues'])} 个潜在标注问题样本")
        for i, issue in enumerate(train_stats['potential_issues'][:5]):  # 只显示前5个
            logger.warning(f"问题样本 {i+1}: {issue['text']} | V={issue['v']:.2f}, A={issue['a']:.2f} | {issue['issue']}")
        
        # 保存问题样本到文件
        issue_df = pd.DataFrame(train_stats['potential_issues'])
        issue_path = os.path.join(config['output_dir'], 'potential_label_issues.csv')
        issue_df.to_csv(issue_path, index=False)
        logger.warning(f"问题样本已保存到: {issue_path}")
        
        # 添加交互确认
        if not config.get('force_continue', False):
            confirm = input("检测到潜在标签问题，是否继续训练? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("用户选择退出训练")
                sys.exit(0)
    
    # 原有代码继续...
```

## 2. 损失函数修改

### 2.1 添加情感方向感知损失函数

```python
class EmotionDirectionLoss(nn.Module):
    """
    情感方向感知损失函数
    增加对错误方向(VA象限错误)的惩罚
    """
    def __init__(self, base_loss=nn.MSELoss(), direction_weight=0.5):
        super(EmotionDirectionLoss, self).__init__()
        self.base_loss = base_loss
        self.direction_weight = direction_weight
    
    def forward(self, predictions, targets):
        # 基础损失(如MSE)
        base_loss_val = self.base_loss(predictions, targets)
        
        # 计算方向损失 - 惩罚象限错误
        # 提取V和A维度
        pred_v, pred_a = predictions[:, 0], predictions[:, 1]
        target_v, target_a = targets[:, 0], targets[:, 1]
        
        # 计算符号是否一致(方向一致)
        v_sign_match = ((pred_v * target_v) >= 0)
        a_sign_match = ((pred_a * target_a) >= 0)
        
        # 方向损失 - 不同象限的样本会有较大惩罚
        direction_loss = (
            torch.mean(1.0 - v_sign_match.float()) + 
            torch.mean(1.0 - a_sign_match.float())
        ) / 2.0
        
        # 总损失
        total_loss = base_loss_val + self.direction_weight * direction_loss
        
        return total_loss
```

### 2.2 更新损失函数使用

在创建损失函数的部分，使用新的方向感知损失：

```python
def main():
    # 原代码...
    
    # 创建损失函数
    if config.get('use_direction_loss', False):
        # 使用情感方向感知损失
        criterion = EmotionDirectionLoss(
            base_loss=MSE_CCC_Loss(
                mse_weight=config.get('mse_weight', 1.0),
                ccc_weight=config.get('ccc_weight', 0.0)
            ),
            direction_weight=config.get('direction_weight', 0.5)
        )
        logger.info("使用情感方向感知损失函数")
    else:
        # 使用常规损失
        criterion = MSE_CCC_Loss(
            mse_weight=config.get('mse_weight', 1.0),
            ccc_weight=config.get('ccc_weight', 0.0)
        )
    
    # 原代码继续...
```

## 3. 训练逻辑增强

### 3.1 添加象限准确率评估

增加新的评估指标，追踪情感象限的准确率：

```python
def compute_metrics(y_true, y_pred):
    # 现有代码...
    
    # 新增：计算象限准确率
    true_v, true_a = y_true[:, 0], y_true[:, 1]
    pred_v, pred_a = y_pred[:, 0], y_pred[:, 1]
    
    # 计算象限
    true_quadrant = np.zeros(len(true_v), dtype=int)
    true_quadrant[(true_v >= 0) & (true_a >= 0)] = 1  # 喜悦/兴奋
    true_quadrant[(true_v >= 0) & (true_a < 0)] = 2   # 满足/平静
    true_quadrant[(true_v < 0) & (true_a >= 0)] = 3   # 愤怒/焦虑
    true_quadrant[(true_v < 0) & (true_a < 0)] = 4    # 悲伤/抑郁
    
    pred_quadrant = np.zeros(len(pred_v), dtype=int)
    pred_quadrant[(pred_v >= 0) & (pred_a >= 0)] = 1
    pred_quadrant[(pred_v >= 0) & (pred_a < 0)] = 2
    pred_quadrant[(pred_v < 0) & (pred_a >= 0)] = 3
    pred_quadrant[(pred_v < 0) & (pred_a < 0)] = 4
    
    # 计算象限准确率
    quadrant_accuracy = np.mean(true_quadrant == pred_quadrant)
    
    # 计算象限F1分数
    from sklearn.metrics import f1_score
    try:
        quadrant_f1_macro = f1_score(true_quadrant, pred_quadrant, average='macro')
        quadrant_f1_weighted = f1_score(true_quadrant, pred_quadrant, average='weighted')
    except:
        quadrant_f1_macro = 0.0
        quadrant_f1_weighted = 0.0
    
    # 添加到结果
    results["quadrant_accuracy"] = quadrant_accuracy
    results["quadrant_f1_macro"] = quadrant_f1_macro
    results["quadrant_f1_weighted"] = quadrant_f1_weighted
    
    return results
```

### 3.2 添加象限混淆矩阵

```python
def validate(model, val_loader, criterion, device, config):
    # 现有代码...
    
    # 计算并记录象限混淆矩阵
    true_v, true_a = all_targets[:, 0], all_targets[:, 1]
    pred_v, pred_a = all_predictions[:, 0], all_predictions[:, 1]
    
    # 计算象限
    true_quadrant = np.zeros(len(true_v), dtype=int)
    true_quadrant[(true_v >= 0) & (true_a >= 0)] = 1  # 喜悦/兴奋
    true_quadrant[(true_v >= 0) & (true_a < 0)] = 2   # 满足/平静
    true_quadrant[(true_v < 0) & (true_a >= 0)] = 3   # 愤怒/焦虑
    true_quadrant[(true_v < 0) & (true_a < 0)] = 4    # 悲伤/抑郁
    
    pred_quadrant = np.zeros(len(pred_v), dtype=int)
    pred_quadrant[(pred_v >= 0) & (pred_a >= 0)] = 1
    pred_quadrant[(pred_v >= 0) & (pred_a < 0)] = 2
    pred_quadrant[(pred_v < 0) & (pred_a >= 0)] = 3
    pred_quadrant[(pred_v < 0) & (pred_a < 0)] = 4
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_quadrant, pred_quadrant, labels=[1, 2, 3, 4])
    
    # 添加到结果
    eval_results["confusion_matrix"] = cm.tolist()
    
    # 输出混淆矩阵
    quadrant_names = ["喜悦/兴奋", "满足/平静", "愤怒/焦虑", "悲伤/抑郁"]
    logger.info("情感象限混淆矩阵:")
    cm_str = ""
    cm_str += "预测 →\n实际 ↓ | " + " | ".join(f"{name}" for name in quadrant_names) + "\n"
    cm_str += "-" * 60 + "\n"
    for i, name in enumerate(quadrant_names):
        cm_str += f"{name} | " + " | ".join(f"{cm[i, j]:5d}" for j in range(4)) + "\n"
    logger.info("\n" + cm_str)
    
    # 如果有明显的象限反转，输出警告
    if cm[0, 3] > cm[0, 0] or cm[3, 0] > cm[3, 3]:
        logger.warning("⚠️ 检测到可能的情感象限反转！正反象限预测不一致")
    
    return eval_results
```

## 4. 配置文件修改

添加以下配置选项到`configs/optimized_performance.yaml`：

```yaml
# 情感方向相关设置
use_direction_loss: true   # 使用情感方向感知损失
direction_weight: 0.5      # 方向损失权重
invert_valence: false      # 不反转价值预测
invert_arousal: false      # 不反转效度预测

# 数据标签检查设置
label_check:
  enabled: true
  force_continue: false    # 发现问题时是否强制继续
```

## 5. 模型修改

### 5.1 在LTC_NCP_RNN模型中添加情感极性预测支持

修改`ltc_ncp/model.py`中的`LTC_NCP_RNN`类：

```python
def __init__(self, 
          # 现有参数...
          invert_valence: bool = False,    # 新增：是否反转价值
          invert_arousal: bool = False):   # 新增：是否反转效度
    # 现有初始化代码...
    
    # 存储反转标志
    self.invert_valence = invert_valence
    self.invert_arousal = invert_arousal

def forward(self, tokens, lengths=None, meta_features=None):
    # 现有代码...
    
    # 在输出层修改
    valence = self.valence_output(valence_features).squeeze(-1)
    arousal = self.arousal_output(arousal_features).squeeze(-1)
    
    # 应用反转（如果需要）
    if self.invert_valence:
        valence = -valence
    
    if self.invert_arousal:
        arousal = -arousal
    
    # 输出拼接
    output = torch.stack([valence, arousal], dim=-1)
    
    # 现有代码继续...
```

### 5.2 添加辅助情感分类头

```python
def __init__(self, 
          # 现有参数...
          add_emotion_classifier: bool = False):  # 新增：是否添加情感分类头
    # 现有初始化代码...
    
    # 情感分类头（可选）
    self.add_emotion_classifier = add_emotion_classifier
    if add_emotion_classifier:
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)  # 4个情感象限
        )

def forward(self, tokens, lengths=None, meta_features=None):
    # 现有代码...直到获取valence和arousal
    
    # 输出拼接
    output = torch.stack([valence, arousal], dim=-1)
    
    # 添加情感分类输出（如果启用）
    if self.add_emotion_classifier and self.training:
        # 合并特征用于分类
        emotion_features = torch.cat([valence_features, arousal_features], dim=-1)
        emotion_logits = self.emotion_classifier(emotion_features)
        
        # 基于V-A值生成象限标签
        batch_size = valence.size(0)
        emotion_labels = torch.zeros(batch_size, dtype=torch.long, device=valence.device)
        emotion_labels[(valence >= 0) & (arousal >= 0)] = 0  # 喜悦/兴奋
        emotion_labels[(valence >= 0) & (arousal < 0)] = 1   # 满足/平静
        emotion_labels[(valence < 0) & (arousal >= 0)] = 2   # 愤怒/焦虑
        emotion_labels[(valence < 0) & (arousal < 0)] = 3    # 悲伤/抑郁
        
        # 返回包含分类信息的元组
        return output, (emotion_logits, emotion_labels)
    
    return output
```

## 6. 训练循环修改

### 6.1 增加多任务损失处理

```python
def train_epoch(model, train_loader, optimizer, criterion, device, config, scheduler=None, scaler=None):
    # 现有代码...
    
    # 前向传播
    with autocast() if use_amp else nullcontext():
        outputs = model(tokens, lengths, meta_features)
        
        # 检查是否返回了多任务输出
        if isinstance(outputs, tuple) and len(outputs) == 2:
            predictions, (emotion_logits, emotion_labels) = outputs
            
            # 回归损失
            loss = criterion(predictions, targets)
            
            # 分类损失
            emotion_criterion = nn.CrossEntropyLoss()
            emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
            
            # 总损失
            emotion_weight = config.get('emotion_classification_weight', 0.2)
            loss = loss + emotion_weight * emotion_loss
        else:
            predictions = outputs
            loss = criterion(predictions, targets)
    
    # 现有代码继续...
```

## 7. 主函数修改

增加模型训练启动前的数据检查：

```python
def main():
    # 现有代码...
    
    # 在创建模型前添加
    # 检查是否需要进行数据分析
    if config.get('analyze_data_before_training', True):
        logger.info("执行训练前数据分析...")
        
        # 检查是否存在情感反转问题
        train_df = pd.read_csv(train_path)
        positive_keywords = ["开心", "高兴", "喜欢", "happy", "joy", "喜悦", "满意", "满足"]
        negative_keywords = ["难过", "伤心", "生气", "愤怒", "悲伤", "讨厌", "失望", "焦虑", "sad", "angry", "fear"]
        
        text_col = config['data']['text_col']
        v_col = config['data']['v_col'] 
        a_col = config['data']['a_col']
        
        # 检查积极情感词与V值关系
        pos_with_neg_v = 0
        for keyword in positive_keywords:
            pos_with_neg_v += sum(train_df[text_col].str.contains(keyword, na=False) & (train_df[v_col] < -0.3))
        
        # 检查消极情感词与V值关系
        neg_with_pos_v = 0
        for keyword in negative_keywords:
            neg_with_pos_v += sum(train_df[text_col].str.contains(keyword, na=False) & (train_df[v_col] > 0.3))
        
        logger.info(f"积极词汇对应负面V值的样本数: {pos_with_neg_v}")
        logger.info(f"消极词汇对应正面V值的样本数: {neg_with_pos_v}")
        
        # 判断是否可能存在情感反转
        if pos_with_neg_v > neg_with_pos_v:
            logger.warning("⚠️ 检测到可能的情感标注反转! 建议检查训练数据")
            if config.get('confirm_training', True):
                confirm = input("检测到可能的情感标注问题，是否继续训练? (y/n): ")
                if confirm.lower() != 'y':
                    logger.info("用户选择退出训练")
                    sys.exit(0)
```

以上修改通过多个层面解决情感反转问题:
1. **数据层面**：增加标签一致性检查，主动识别问题样本
2. **损失函数**：添加方向感知损失，增强象限正确性
3. **多任务学习**：增加情感分类头，辅助回归任务
4. **反转机制**：增加数据驱动的反转标志
5. **评估监控**：增加象限混淆矩阵监控 