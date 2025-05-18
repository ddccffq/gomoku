import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def board_to_tensor(board, current_player):
    """将棋盘转换为模型输入格式
    
    Args:
        board: 棋盘状态，2D列表或numpy数组
        current_player: 当前玩家（1=黑，2=白）
        
    Returns:
        3通道状态表示
    """
    # 转换为numpy数组
    if isinstance(board, list):
        board_np = np.array(board, dtype=np.int8)
    else:
        board_np = board
        
    board_size = board_np.shape[0]
    opponent = 3 - current_player
    
    # 创建3通道状态
    state = np.zeros((3, board_size, board_size), dtype=np.float32)
    
    # 填充通道1：当前玩家的棋子
    state[0] = (board_np == current_player).astype(np.float32)
    
    # 填充通道2：对手的棋子
    state[1] = (board_np == opponent).astype(np.float32)
    
    # 填充通道3：表示当前玩家是否是黑棋
    state[2].fill(1.0 if current_player == 1 else 0.0)
    
    return state

class GomokuDataset(Dataset):
    """五子棋训练数据集"""
    
    def __init__(self, states, policies, values):
        """初始化数据集
        
        Args:
            states: 棋盘状态数组，形状为 [n_samples, 3, board_size, board_size]
            policies: 策略数组，形状为 [n_samples, board_size*board_size]
            values: 价值数组，形状为 [n_samples, 1] 或 [n_samples]
        """
        self.states = torch.FloatTensor(states)
        self.policies = torch.FloatTensor(policies)
        
        # 确保values的形状为 [n_samples, 1]
        values_tensor = torch.FloatTensor(values)
        if values_tensor.dim() == 1:
            values_tensor = values_tensor.unsqueeze(1)
        self.values = values_tensor
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def get_data_loaders(states, policies, values, batch_size=64, val_split=0.2):
    """从NumPy数组创建PyTorch数据加载器
    
    Args:
        states: 棋盘状态数组
        policies: 策略数组
        values: 价值数组
        batch_size: 批次大小
        val_split: 验证集比例
        
    Returns:
        (train_loader, val_loader)
    """
    # 确保values具有正确的形状
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = values.reshape(-1, 1)
    
    # 创建数据集
    dataset = GomokuDataset(states, policies, values)
    
    # 分割为训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # 确保至少有一个批次用于验证
    if val_size < batch_size:
        val_size = min(batch_size, len(dataset) // 2)
        train_size = len(dataset) - val_size
    
    # 随机分割
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        pin_memory=torch.cuda.is_available()  # 如果使用GPU，将数据固定在内存中
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def find_training_data_files(root_dir):
    """递归查找包含训练数据的所有目录
    
    Args:
        root_dir: 根目录
        
    Returns:
        包含训练数据的目录列表
    """
    required_files = ['states.npy', 'policies.npy', 'values.npy']
    data_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查是否包含所有必要的文件
        if all(f in filenames for f in required_files):
            data_dirs.append(dirpath)
    
    return data_dirs


def load_batch_data(data_dirs, max_dirs=10):
    """加载一批训练数据目录中的数据
    
    Args:
        data_dirs: 数据目录列表
        max_dirs: 最多加载的目录数量
        
    Returns:
        (states, policies, values) 元组
    """
    all_states = []
    all_policies = []
    all_values = []
    
    # 限制加载的目录数量
    data_dirs = data_dirs[:min(len(data_dirs), max_dirs)]
    
    for data_dir in data_dirs:
        try:
            # 检查必要文件是否存在
            states_path = os.path.join(data_dir, 'states.npy')
            policies_path = os.path.join(data_dir, 'policies.npy')
            values_path = os.path.join(data_dir, 'values.npy')
            
            if not (os.path.exists(states_path) and 
                   os.path.exists(policies_path) and 
                   os.path.exists(values_path)):
                continue
                
            # 加载数据
            states = np.load(states_path)
            policies = np.load(policies_path)
            values = np.load(values_path)
            
            # 确保values的形状为 [n_samples, 1]
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            
            # 检查形状是否匹配
            if len(states) == len(policies) == len(values):
                all_states.append(states)
                all_policies.append(policies)
                all_values.append(values)
        except Exception as e:
            print(f"加载 {data_dir} 时出错: {str(e)}")
    
    # 合并数据
    if all_states:
        states = np.concatenate(all_states)
        policies = np.concatenate(all_policies)
        values = np.concatenate(all_values)
        
        return states, policies, values
    else:
        return None, None, None
