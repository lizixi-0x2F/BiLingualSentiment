class Config:
    # 数据配置
    CHINESE_DATASET_PATH = 'Chinese_VA_dataset_gaussNoise.csv'
    EMOBANK_DATASET_PATH = 'emobank_va_normalized.csv'
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 编码器配置 - 轻量级编码器
    HIDDEN_SIZE = 768  # 编码器输出维度
    VOCAB_SIZE = 50000  # 词汇表大小
    EMBEDDING_DIM = 512  # 词嵌入维度
    
    # Transformer配置
    NUM_LAYERS = 1  # 层数
    NUM_HEADS = 8   # 注意力头数量
    TRANSFORMER_HIDDEN_SIZE = 256  # 隐藏层大小
    
    # NCP_LNN配置
    EXCITATORY_RATIO = 0.6  # 激励神经元比例
    SPARSITY = 0.15  # 连接稀疏度，降低至15%使Xavier×2能让前几十step就有效信号
    TAU_MIN = 1.0   # 时间常数最小值
    TAU_MAX = 20.0  # 时间常数最大值
    TAU_REGULARIZER = 1e-4  # 时间常数正则化系数
    
    # 训练配置
    BATCH_SIZE = 128  # 批次大小
    LEARNING_RATE = 1e-3  # 优化学习率，批次增大4倍，学习率也增大相应倍数
    NUM_EPOCHS = 20  # 训练轮次上限
    DROPOUT = 0.2
    DEVICE = 'cuda'  # 'cuda', 'cpu', 'mps'
    
    # 优化器配置
    BETA1 = 0.9      # AdamW参数1
    BETA2 = 0.95     # AdamW参数2
    WARMUP_RATIO = 0.1  # 预热步数比例，增大批次后增加预热比例
    GRAD_CLIP = 1.5  # 梯度裁剪阈值，增大以适应更大批次
    
    # 早停配置
    PATIENCE = 5        # 早停耐心值
    EARLY_STOPPING_METRIC = 'val_loss'  # 早停监控指标，可选 'val_loss', 'r2', 'rmse', 'ccc'
    EARLY_STOPPING_MIN_DELTA = 0.001    # 早停最小变化阈值
    
    # 输出配置
    OUTPUT_DIR = 'outputs/model'
    OUTPUT_DIM = 2  # valence, arousal 
    
    # 压缩率 - 压缩维度固定为128
    COMPRESSION_DIM = 128
    COMPRESSION_RATE = 0.25  # 兼容旧代码
    
    # 模型类型和编码器类型
    MODEL_TYPE = 'ncp_lnn'  # 模型类型
    ENCODER_TYPE = 'lightweight'  # 编码器类型
    
    # 其他配置
    WEIGHT_DECAY = 0.01
    DEBUG_INFO = False  # 是否显示调试信息
    
    # 损失混合权重
    LAMBDA_WEIGHT = 0.7  # 回归损失权重，值越大则回归损失占比越高
    
    def __init__(self, device=None, batch_size=None, 
                 learning_rate=None, epochs=None, output_dir=None, 
                 model_type=None, encoder_type=None):
        """初始化配置"""
        if device:
            self.DEVICE = device
            
        if batch_size:
            self.BATCH_SIZE = batch_size
            
        if learning_rate:
            self.LEARNING_RATE = learning_rate
            
        if epochs:
            self.NUM_EPOCHS = epochs
            
        if output_dir:
            self.OUTPUT_DIR = output_dir
            
        if model_type:
            self.MODEL_TYPE = model_type
            
        if encoder_type:
            self.ENCODER_TYPE = encoder_type
