import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyValueNetwork(nn.Module):
    """策略价值网络"""
    def __init__(self, board_size=15, num_channels=128, num_res_blocks=10):
        super(PolicyValueNetwork, self).__init__()
        self.board_size = board_size
        
        # 共同部分：输入处理
        self.conv_block = ConvBlock(3, num_channels)
        
        # 共同部分：残差块
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征, shape=[batch_size, 3, board_size, board_size]
        
        Returns:
            tuple: (policy_logits, value), 形状分别为[batch_size, board_size*board_size]和[batch_size, 1]
        """
        # 共同特征提取
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)
        log_policy = F.log_softmax(policy_logits, dim=1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_policy, value

class TinyPolicyValueNetwork(nn.Module):
    """轻量级策略价值网络，用于快速训练和测试"""
    def __init__(self, board_size=15):
        super(TinyPolicyValueNetwork, self).__init__()
        self.board_size = board_size
        
        # 共同部分
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 策略头
        self.policy_conv = nn.Conv2d(128, 4, 1)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(128, 2, 1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """前向传播"""
        # 共同特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 4 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)
        log_policy = F.log_softmax(policy_logits, dim=1)
        
        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 2 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_policy, value

def create_gomoku_model(board_size=15, device=None, model_size='tiny'):
    """创建适合五子棋的策略价值网络
    
    Args:
        board_size: 棋盘大小
        device: 计算设备
        model_size: 模型大小 ('tiny', 'small', 'medium', 'large')
        
    Returns:
        策略价值网络模型
    """
    try:
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"创建五子棋模型, 大小={model_size}, 设备={device}")
        
        # 根据模型大小设置参数
        if model_size == 'tiny':
            filters = 16       # 原来是32，减少为16
            blocks = 2         # 原来是3，减少为2
        elif model_size == 'small':
            filters = 64
            blocks = 5
        elif model_size == 'medium':
            filters = 128
            blocks = 10
        else:  # large
            filters = 256
            blocks = 19
            
        # 输出详细的模型配置信息
        print(f"检测到模型大小: {model_size}")
        print(f"滤波器数量: {filters}")
        print(f"残差块数量: {blocks}")
        print(f"棋盘大小: {board_size}x{board_size}")
        
        # 创建模型
        model = PolicyValueNetwork(
            board_size=board_size,
            num_channels=filters,
            num_res_blocks=blocks
        ).to(device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 输出模型结构特征
        has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        print(f"模型{'包含' if has_batch_norm else '不包含'}批归一化层")
        
        # 保存一个打包好的独立模型（不依赖代码）
        try:
            from datetime import datetime
            import os
            
            # 创建可独立加载的模型
            class StandaloneGomokuModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.base_model = model
                
                def forward(self, x):
                    return self.base_model(x)
            
            standalone_model = StandaloneGomokuModel()
            
            # 保存到临时目录
            import tempfile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f"gomoku_model_init_{timestamp}.pth")
            
            torch.save(standalone_model.state_dict(), save_path)
            print(f"初始模型已保存到临时目录: {save_path}")
        except Exception as e:
            print(f"保存独立模型失败（不影响训练）: {str(e)}")
        
        print(f"五子棋模型创建成功")
        return model
    except Exception as e:
        import traceback
        print(f"创建模型失败: {str(e)}")
        print(traceback.format_exc())
        
        # 创建一个简单的备用模型
        print("创建备用简单模型...")
        class SimplePolicyValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.policy_head = nn.Sequential(
                    nn.Conv2d(32, 16, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(16 * board_size * board_size, board_size * board_size),
                    nn.LogSoftmax(dim=1)
                )
                self.value_head = nn.Sequential(
                    nn.Conv2d(32, 8, kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(8 * board_size * board_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Tanh()
                )
            
            def forward(self, x):
                x = F.relu(self.conv(x))
                return self.policy_head(x), self.value_head(x)
        
        return SimplePolicyValueNet().to(device)
