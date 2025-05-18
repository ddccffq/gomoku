import os
import time
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


class GomokuDataset(Dataset):
    """五子棋数据集类"""
    
    def __init__(self, game_records):
        """初始化数据集
        
        Args:
            game_records: 游戏记录列表，每条记录包含棋盘状态和结果
        """
        self.states = []
        self.policies = []
        self.values = []
        
        # 处理游戏记录，提取状态和结果
        for record in game_records:
            self._process_game(record)
    
    def _process_game(self, record):
        """处理单个游戏记录"""
        moves = record.get('move_history', [])
        winner = record.get('winner', 0)
        board_size = 15
        
        # 重建游戏过程
        board = np.zeros((board_size, board_size), dtype=np.int8)
        current_player = 1  # 黑棋先行
        
        for move in moves:
            row, col = move[0], move[1]
            
            # 将当前状态加入训练数据
            state = self._board_to_state(board, current_player)
            
            # 根据游戏结果得到价值标签
            # 如果当前玩家是赢家，价值为1；输家为-1；和棋为0
            if winner == 0:  # 和棋
                value = 0
            elif winner == current_player:  # 当前玩家赢
                value = 1
            else:  # 当前玩家输
                value = -1
            
            # 构建策略标签（简单实现：仅在落子位置为1）
            policy = np.zeros(board_size * board_size)
            policy[row * board_size + col] = 1
            
            # 添加数据
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
            
            # 在棋盘上落子
            board[row][col] = current_player
            
            # 切换玩家
            current_player = 3 - current_player  # 1->2, 2->1
    
    def _board_to_state(self, board, current_player):
        """将棋盘转换为模型输入状态
        
        使用3通道表示：
        - 通道1：当前玩家的棋子（1表示有棋子，0表示无）
        - 通道2：对手的棋子（1表示有棋子，0表示无）
        - 通道3：常量平面，表示当前玩家（全1表示黑棋，全0表示白棋）
        """
        board_size = len(board)
        opponent = 3 - current_player
        
        # 创建3通道状态
        state = np.zeros((3, board_size, board_size), dtype=np.float32)
        
        # 填充通道1：当前玩家的棋子
        state[0] = (board == current_player).astype(np.float32)
        
        # 填充通道2：对手玩家的棋子
        state[1] = (board == opponent).astype(np.float32)
        
        # 填充通道3：表示当前玩家
        state[2].fill(1.0 if current_player == 1 else 0.0)
        
        return state
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }


class GomokuNetwork(nn.Module):
    """五子棋神经网络模型"""
    
    def __init__(self, board_size=15, num_filters=128, num_residual_blocks=19):
        """初始化网络
        
        Args:
            board_size: 棋盘大小
            num_filters: 卷积核数量
            num_residual_blocks: 残差块数量
        """
        super(GomokuNetwork, self).__init__()
        
        self.board_size = board_size
        self.num_filters = num_filters
        
        # 输入层：(batch, 3, 15, 15)
        self.conv_input = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.relu_input = nn.ReLU(inplace=True)
        
        # 残差层
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(self._create_residual_block())
        
        # 策略头：输出动作概率
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.policy_softmax = nn.Softmax(dim=1)
        
        # 价值头：评估局面
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu1 = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()
    
    def _create_residual_block(self):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入状态，形状为(batch, 3, board_size, board_size)
            
        Returns:
            policy: 策略输出，形状为(batch, board_size*board_size)
            value: 价值输出，形状为(batch, 1)
        """
        # 输入层
        x = self.relu_input(self.bn_input(self.conv_input(x)))
        
        # 残差层
        for block in self.residual_blocks:
            res = x
            x = block(x)
            x += res
            x = nn.functional.relu(x)
        
        # 策略头
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = self.policy_softmax(policy)
        
        # 价值头
        value = self.value_relu1(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = self.value_relu2(self.value_fc1(value))
        value = self.value_tanh(self.value_fc2(value))
        
        return policy, value


class GomokuTrainer:
    """五子棋AI训练器"""
    
    def __init__(self, config=None):
        """初始化训练器
        
        Args:
            config: 训练配置
        """
        # 默认配置
        self.default_config = {
            'board_size': 15,
            'num_filters': 64,
            'num_residual_blocks': 9,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'num_epochs': 10,
            'train_val_split': 0.8,
            'optimizer': 'Adam',
            'model1_path': '',
            'model2_path': '',
            'output_dir': './trained_models',
            'save_interval': 5
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 初始化模型
        self.model1 = GomokuNetwork(
            board_size=self.config['board_size'],
            num_filters=self.config['num_filters'],
            num_residual_blocks=self.config['num_residual_blocks']
        )
        
        self.model2 = GomokuNetwork(
            board_size=self.config['board_size'],
            num_filters=self.config['num_filters'],
            num_residual_blocks=self.config['num_residual_blocks']
        )
        
        # 加载预训练模型（如果存在）
        if self.config['model1_path'] and os.path.exists(self.config['model1_path']):
            self.model1.load_state_dict(torch.load(self.config['model1_path']))
            print(f"已加载模型1: {self.config['model1_path']}")
        
        if self.config['model2_path'] and os.path.exists(self.config['model2_path']):
            self.model2.load_state_dict(torch.load(self.config['model2_path']))
            print(f"已加载模型2: {self.config['model2_path']}")
        
        # 创建优化器
        if self.config['optimizer'] == 'Adam':
            self.optimizer1 = optim.Adam(
                self.model1.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            self.optimizer2 = optim.Adam(
                self.model2.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'SGD':
            self.optimizer1 = optim.SGD(
                self.model1.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
            
            self.optimizer2 = optim.SGD(
                self.model2.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # 训练记录
        self.training_history = {
            'model1': {'policy_loss': [], 'value_loss': [], 'total_loss': [], 'policy_accuracy': [], 'value_mse': []},
            'model2': {'policy_loss': [], 'value_loss': [], 'total_loss': [], 'policy_accuracy': [], 'value_mse': []}
        }
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1.to(self.device)
        self.model2.to(self.device)
    
    def train_on_games(self, game_records, callback=None):
        """基于游戏记录训练模型
        
        Args:
            game_records: 游戏记录列表
            callback: 回调函数，用于更新训练进度
        """
        # 创建数据集
        dataset = GomokuDataset(game_records)
        
        # 分割训练集和验证集
        train_size = int(len(dataset) * self.config['train_val_split'])
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        # 训练两个模型
        print("开始训练模型1（黑棋）...")
        self._train_model(
            self.model1, self.optimizer1, train_loader, val_loader, 
            epoch_callback=lambda e, m: callback('model1', e, m) if callback else None
        )
        
        print("开始训练模型2（白棋）...")
        self._train_model(
            self.model2, self.optimizer2, train_loader, val_loader,
            epoch_callback=lambda e, m: callback('model2', e, m) if callback else None
        )
        
        # 保存训练后的模型
        self.save_models()
    
    def _train_model(self, model, optimizer, train_loader, val_loader, epoch_callback=None):
        """训练单个模型
        
        Args:
            model: 要训练的模型
            optimizer: 优化器
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epoch_callback: 每个epoch结束的回调函数
        """
        model.train()
        best_val_loss = float('inf')
        
        # 从配置中获取保存间隔，如果未设置则默认为5
        save_interval = self.config.get('save_interval', 5)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # 训练阶段
            train_policy_loss = 0
            train_value_loss = 0
            train_total_loss = 0
            train_policy_accuracy = 0
            train_value_mse = 0
            
            for batch in train_loader:
                # 获取批次数据
                states = batch['state'].to(self.device)
                target_policies = batch['policy'].to(self.device)
                target_values = batch['value'].to(self.device)
                
                # 前向传播
                policies, values = model(states)
                
                # 计算损失
                policy_loss = self.policy_loss_fn(policies, torch.max(target_policies, dim=1)[1])
                value_loss = self.value_loss_fn(values.squeeze(), target_values.float())
                total_loss = policy_loss + value_loss
                
                # 计算策略准确率
                _, predicted_moves = torch.max(policies, 1)
                _, target_moves = torch.max(target_policies, 1)
                correct = (predicted_moves == target_moves).sum().item()
                accuracy = correct / states.size(0)
                train_policy_accuracy += accuracy
                
                # 计算价值预测与实际的均方误差
                value_mse = ((values.squeeze() - target_values.float()) ** 2).mean().item()
                train_value_mse += value_mse
                
                # 反向传播与优化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 累加损失
                train_policy_loss += policy_loss.item()
                train_value_loss += value_loss.item()
                train_total_loss += total_loss.item()
            
            # 计算平均损失和准确率
            train_policy_loss /= len(train_loader)
            train_value_loss /= len(train_loader)
            train_total_loss /= len(train_loader)
            train_policy_accuracy /= len(train_loader)
            train_value_mse /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_policy_loss = 0
            val_value_loss = 0
            val_total_loss = 0
            val_policy_accuracy = 0
            val_value_mse = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    states = batch['state'].to(self.device)
                    target_policies = batch['policy'].to(self.device)
                    target_values = batch['value'].to(self.device)
                    
                    policies, values = model(states)
                    
                    policy_loss = self.policy_loss_fn(policies, torch.max(target_policies, dim=1)[1])
                    value_loss = self.value_loss_fn(values.squeeze(), target_values.float())
                    total_loss = policy_loss + value_loss
                    
                    # 计算策略准确率
                    _, predicted_moves = torch.max(policies, 1)
                    _, target_moves = torch.max(target_policies, 1)
                    correct = (predicted_moves == target_moves).sum().item()
                    accuracy = correct / states.size(0)
                    val_policy_accuracy += accuracy
                    
                    # 计算价值预测与实际的均方误差
                    value_mse = ((values.squeeze() - target_values.float()) ** 2).mean().item()
                    val_value_mse += value_mse
                    
                    val_policy_loss += policy_loss.item()
                    val_value_loss += value_loss.item()
                    val_total_loss += total_loss.item()
            
            val_policy_loss /= len(val_loader)
            val_value_loss /= len(val_loader)
            val_total_loss /= len(val_loader)
            val_policy_accuracy /= len(val_loader)
            val_value_mse /= len(val_loader)
            
            # 记录训练历史
            model_key = 'model1' if model is self.model1 else 'model2'
            self.training_history[model_key]['policy_loss'].append((train_policy_loss, val_policy_loss))
            self.training_history[model_key]['value_loss'].append((train_value_loss, val_value_loss))
            self.training_history[model_key]['total_loss'].append((train_total_loss, val_total_loss))
            self.training_history[model_key]['policy_accuracy'].append((train_policy_accuracy, val_policy_accuracy))
            self.training_history[model_key]['value_mse'].append((train_value_mse, val_value_mse))
            
            # 输出进度
            print(f"Epoch {epoch}/{self.config['num_epochs']} - "
                  f"训练损失: {train_total_loss:.4f}, 验证损失: {val_total_loss:.4f}, "
                  f"策略准确率: {train_policy_accuracy:.4f}/{val_policy_accuracy:.4f}, "
                  f"价值MSE: {train_value_mse:.4f}/{val_value_mse:.4f}")
            
            # 保存最佳模型
            if val_total_loss < best_val_loss and self.config['output_dir']:
                best_val_loss = val_total_loss
                torch.save(model.state_dict(), os.path.join(self.config['output_dir'], f'best_model.pth'))
                print(f"轮次 {epoch}: 保存了最佳模型")
            
            # 定期保存检查点 - 使用配置的间隔
            if self.config['output_dir'] and epoch % save_interval == 0:
                try:
                    checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"轮次 {epoch}: 保存了检查点模型")
                except Exception as e:
                    print(f"轮次 {epoch}: 保存检查点模型失败: {str(e)}")
            
            # 回调
            if epoch_callback:
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_total_loss,
                    'val_loss': val_total_loss,
                    'policy_loss': train_policy_loss,
                    'value_loss': train_value_loss,
                    'policy_accuracy': train_policy_accuracy,
                    'val_policy_accuracy': val_policy_accuracy,
                    'value_mse': train_value_mse,
                    'val_value_mse': val_value_mse
                }
                epoch_callback(epoch, metrics)
            
            model.train()
    
    def save_models(self, timestamp=None):
        """保存训练好的模型
        
        Args:
            timestamp: 时间戳，用于命名
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 保存模型1
        model1_path = os.path.join(self.config['output_dir'], f"model1_{timestamp}.pth")
        torch.save(self.model1.state_dict(), model1_path)
        
        # 保存模型2
        model2_path = os.path.join(self.config['output_dir'], f"model2_{timestamp}.pth")
        torch.save(self.model2.state_dict(), model2_path)
        
        # 保存训练历史
        history_path = os.path.join(self.config['output_dir'], f"training_history_{timestamp}.json")
        with open(history_path, 'w') as f:
            json.dump({
                'config': self.config,
                'history': self.training_history,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"模型已保存: {model1_path}, {model2_path}")
        print(f"训练历史已保存: {history_path}")
        
        return model1_path, model2_path, history_path
    
    def get_move(self, board, model, player):
        """使用模型获取下一步走法
        
        Args:
            board: 当前棋盘状态 (15x15的二维数组)
            model: 使用哪个模型 (1或2)
            player: 当前玩家 (1=黑棋, 2=白棋)
            
        Returns:
            下一步最佳走法(row, col)
        """
        # 选择模型
        model_to_use = self.model1 if model == 1 else self.model2
        model_to_use.eval()
        
        # 转换棋盘状态
        board_np = np.array(board, dtype=np.int8)
        state = self._prepare_state(board_np, player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取模型预测
        with torch.no_grad():
            policy, value = model_to_use(state_tensor)
        
        # 获取动作概率
        policy = policy.squeeze().cpu().numpy()
        
        # 创建合法走法掩码（只考虑空位）
        valid_moves = (board_np == 0).flatten()
        
        # 将不合法的走法概率设为0
        policy = policy * valid_moves
        
        # 如果所有走法都不合法，选择一个随机合法走法
        if np.sum(policy) == 0:
            empty_positions = np.where(valid_moves)[0]
            if len(empty_positions) > 0:
                best_move = np.random.choice(empty_positions)
            else:
                # 棋盘已满
                return None
        else:
            # 选择概率最高的走法
            best_move = np.argmax(policy)
        
        # 转换为(row, col)坐标
        row = best_move // self.config['board_size']
        col = best_move % self.config['board_size']
        
        return row, col
    
    def _prepare_state(self, board, player):
        """准备输入状态
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            3通道输入状态
        """
        board_size = self.config['board_size']
        opponent = 3 - player
        
        # 创建3通道状态
        state = np.zeros((3, board_size, board_size), dtype=np.float32)
        
        # 填充通道1：当前玩家的棋子
        state[0] = (board == player).astype(np.float32)
        
        # 填充通道2：对手玩家的棋子
        state[1] = (board == opponent).astype(np.float32)
        
        # 填充通道3：表示当前玩家
        state[2].fill(1.0 if player == 1 else 0.0)
        
        return state


# 测试代码
if __name__ == "__main__":
    # 创建简单的训练配置
    config = {
        'num_filters': 32,
        'num_residual_blocks': 3,
        'learning_rate': 0.001,
        'num_epochs': 2,
        'batch_size': 8,
        'output_dir': './test_models'
    }
    
    # 生成一些测试数据
    test_records = []
    for i in range(10):
        # 模拟一局游戏
        moves = []
        for j in range(20):
            row = np.random.randint(0, 15)
            col = np.random.randint(0, 15)
            player = 1 if j % 2 == 0 else 2
            moves.append((row, col, player))
        
        winner = np.random.choice([0, 1, 2])  # 随机胜者
        test_records.append({
            'move_history': moves,
            'winner': winner
        })
    
    # 创建训练器并训练
    trainer = GomokuTrainer(config)
    trainer.train_on_games(test_records)
