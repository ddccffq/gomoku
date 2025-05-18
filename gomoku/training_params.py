# 训练参数建议
TRAINING_PARAMS = {
    # 基础参数
    'learning_rate': 0.001,     # 初始学习率
    'weight_decay': 1e-4,       # L2正则化参数
    'epochs': 100,              # 训练轮数(建议增加到100轮以上)
    'batch_size': 64,           # 批大小(根据内存调整)
    
    # 学习率调度
    'lr_scheduler': True,       # 启用学习率调度
    'lr_decay_step': 20,        # 每20轮降低学习率
    'lr_decay_rate': 0.5,       # 学习率衰减率
    
    # 数据增强
    'data_augmentation': True,  # 棋盘旋转/翻转增强
    
    # 棋盘编码方式
    'board_encoding': 'binary', # 二进制编码，区分黑白棋子
    
    # 训练策略
    'self_play_games': 500,     # 自我对弈生成的游戏数(建议增加)
    'mcts_simulations': 400,    # 每步MCTS模拟次数
    'c_puct': 5,                # MCTS探索常数
    
    # 模型架构
    'num_residual_blocks': 19,  # 残差块数量
    'num_filters': 256,         # 卷积层滤波器数量
}
