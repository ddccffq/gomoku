import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """卷积块，包含卷积层、批归一化和ReLU激活"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """残差块，包含两个卷积块和一个跳跃连接"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(self.conv2(x))
        x += residual
        return F.relu(x)

class GomokuModel(nn.Module):
    """五子棋神经网络模型，包含策略头和价值头"""
    def __init__(self, board_size=15, num_res_blocks=10, num_filters=128):
        super(GomokuModel, self).__init__()
        
        # 初始卷积层
        self.conv_input = ConvBlock(3, num_filters)  # 3个通道: 黑棋,白棋,轮到谁走
        
        # 残差层
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # 策略头 - 预测落子概率
        self.policy_head = nn.Sequential(
            ConvBlock(num_filters, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Flatten(),
            nn.LogSoftmax(dim=1)
        )
        
        # 价值头 - 预测胜率
        self.value_head = nn.Sequential(
            ConvBlock(num_filters, 32, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # 输出范围(-1,1)，表示胜率
        )

    def forward(self, x):
        # 主干网络
        x = self.conv_input(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头和价值头
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

def create_gomoku_model(board_size=15, device='cpu', model_size='small'):
    """创建不同大小的五子棋模型"""
    if model_size == 'tiny':
        model = GomokuModel(board_size=board_size, num_res_blocks=3, num_filters=32)
    elif model_size == 'small':
        model = GomokuModel(board_size=board_size, num_res_blocks=5, num_filters=64)
    elif model_size == 'medium':
        model = GomokuModel(board_size=board_size, num_res_blocks=10, num_filters=128)
    else:  # large
        model = GomokuModel(board_size=board_size, num_res_blocks=15, num_filters=192)
    
    return model.to(device)
