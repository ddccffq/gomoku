# coding:utf-8
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
from copy import deepcopy
from .base_ai import BaseAIPlayer, StoneColor, AILevel
from .data_handler import board_to_tensor, policy_to_move
from .board_evaluator import GomokuPatternEvaluator

class MCTSNode:
    """Monte Carlo树搜索节点"""
    def __init__(self, board_state, parent=None, move=None, prior_p=0.0):
        self.board_state = board_state  # 棋盘状态
        self.parent = parent  # 父节点
        self.move = move  # 到达该节点的动作，(row, col)
        self.children = {}  # 子节点 {action: node}
        self.n_visits = 0  # 访问次数
        self.q_value = 0  # 当前节点的价值 (平均评分)
        self.u_value = 0  # UCB公式中的探索项
        self.prior_p = prior_p  # 该动作的先验概率
        self.player = self._get_player()  # 当前玩家 (1 or 2)
        self._expanded = False  # 是否已经扩展
    
    def _get_player(self):
        """确定当前节点的玩家"""
        # 计算棋盘上黑白棋子数量，兼容 numpy 数组和 Python 列表
        if isinstance(self.board_state, np.ndarray):
            black_count = int((self.board_state == 1).sum())
            white_count = int((self.board_state == 2).sum())
        else:
            black_count = sum(row.count(1) for row in self.board_state)
            white_count = sum(row.count(2) for row in self.board_state)
        
        return 1 if black_count == white_count else 2
    
    def expand(self, action_priors):
        """扩展当前节点的子节点
        
        Args:
            action_priors: 由策略网络给出的(action, prior)列表
        """
        self._expanded = True
        for action, prior in action_priors:
            row, col = action
            if self.board_state[row][col] == 0:  # 确保位置为空
                # 创建新的棋盘状态
                new_board = deepcopy(self.board_state)
                new_board[row][col] = self.player
                self.children[action] = MCTSNode(new_board, self, action, prior)
    
    def select(self, c_puct):
        """选择最有价值的子节点
        
        Args:
            c_puct: PUCT算法中的探索参数
            
        Returns:
            tuple (child_node, action)
        """
        # 选择Q + U值最大的子节点
        max_value = float('-inf')
        best_action = None
        best_child = None
        
        # 如果没有子节点，直接返回None
        if not self.children:
            return None, None
            
        # 对每个子节点更新UCB值并选择最大的
        for action, child in self.children.items():
            # 更新子节点的U值
            try:
                if child.n_visits > 0:
                    # UCB1公式: Q + U = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
                    child.u_value = c_puct * child.prior_p * math.sqrt(self.n_visits) / (1 + child.n_visits)
                else:
                    # 当访问次数为0时使用一个较大的默认值鼓励探索
                    child.u_value = c_puct * child.prior_p * math.sqrt(self.n_visits + 1e-8)
                
                # 计算总价值
                value = child.q_value + child.u_value
            except ZeroDivisionError:
                # 防止除零错误
                value = float('inf')  # 给未访问节点一个很高的值
            
            if value > max_value:
                max_value = value
                best_action = action
                best_child = child
        
        return best_child, best_action
    
    def update(self, value):
        """更新节点的价值和访问次数
        
        Args:
            value: 节点的新评估值
        """
        self.n_visits += 1
        # 增量更新Q值
        self.q_value += (value - self.q_value) / self.n_visits
    
    def is_leaf(self):
        """检查是否为叶节点"""
        return len(self.children) == 0
    
    def is_root(self):
        """检查是否为根节点"""
        return self.parent is None


class NeuralMCTS:
    """基于神经网络的蒙特卡洛树搜索"""
    def __init__(self, policy_value_net, c_puct=5, n_playout=400, is_selfplay=False):
        """
        Args:
            policy_value_net: 策略价值网络
            c_puct: PUCT算法中的探索参数
            n_playout: 每次移动的模拟次数
            is_selfplay: 是否为自我对弈模式
        """
        self.root = None
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.is_selfplay = is_selfplay
        self.temperature = 1.0  # 温度参数，控制探索程度
        self.device = next(policy_value_net.parameters()).device
        self.pattern_evaluator = GomokuPatternEvaluator()  # 添加棋型评估器
    
    def get_action(self, board_state, temp=1e-3, return_prob=False, check_interrupt=None):
        """获取最佳动作
        
        Args:
            board_state: 当前棋盘状态
            temp: 温度参数，接近0则选择访问次数最多的动作，较大值增加探索性
            return_prob: 是否返回动作概率分布
            check_interrupt: 中断检查函数，用于检查是否应该中断
            
        Returns:
            如果return_prob为False，仅返回最佳动作(row, col)
            如果return_prob为True，返回(最佳动作, 动作概率分布)
            如果被中断，返回(None, None)或None
        """
        # 首先检查是否已被中断
        if check_interrupt and check_interrupt():
            if return_prob:
                return None, None
            return None
            
        # 首先检查是否有直接获胜的走法
        winning_move = self._check_winning_move(board_state)
        if winning_move:
            # 如果找到直接获胜走法，立即返回
            if return_prob:
                # 创建一个只在获胜位置有概率的分布
                board_size = len(board_state)
                probs = np.zeros(board_size * board_size)
                row, col = winning_move
                idx = row * board_size + col
                probs[idx] = 1.0
                return winning_move, probs
            return winning_move
        
        # 确保输入是numpy数组
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state)
            
        # 创建根节点
        self.root = MCTSNode(deepcopy(board_state))
        
        # 进行n_playout次模拟
        for i in range(self.n_playout):
            # 更频繁地检查中断，每5次模拟检查一次
            if check_interrupt and i % 5 == 0 and check_interrupt():
                if return_prob:
                    return None, None
                return None
                
            if not self._playout(self.root, check_interrupt):
                # 如果playout被中断，直接返回中断
                if return_prob:
                    return None, None
                return None
        
        # 获取子节点的访问次数
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        
        # 如果没有可行动作，尝试发现关键防守点
        if not act_visits:
            defense_points = self.pattern_evaluator.find_key_defense_points(board_state, self.root.player)
            if defense_points:
                # 取威胁最大的防守点
                y, x, _ = defense_points[0]
                move = (y, x)
                
                if return_prob:
                    # 创建一个只有这个防守点的概率分布
                    probs = np.zeros(len(board_state) * len(board_state[0]))
                    probs[y * len(board_state) + x] = 1.0
                    return move, probs
                return move
            
        # 如果没有可行动作，返回None
        if not act_visits:
            if return_prob:
                return None, np.zeros(len(board_state) * len(board_state[0]))
            return None
            
        actions, visits = zip(*act_visits)
        
        # 根据温度参数计算动作概率
        if temp > 1e-5:  # 不为零时使用softmax
            visits_temp = [v ** (1.0 / temp) for v in visits]
            visits_sum = sum(visits_temp)
            act_probs = [v / visits_sum for v in visits_temp]
        else:  # 接近零时选择访问次数最大的
            idx = np.argmax(visits)
            act_probs = [0.0] * len(actions)
            act_probs[idx] = 1.0
        
        if self.is_selfplay:
            # 添加Dirichlet噪声以增加探索性 (仅用于自我对弈)
            act_probs = 0.75 * np.array(act_probs) + 0.25 * np.random.dirichlet(0.3 * np.ones(len(actions)))
            # 根据概率选择动作
            move_idx = np.random.choice(len(actions), p=act_probs)
            move = actions[move_idx]
        else:
            # 选择概率最高的动作
            move = actions[np.argmax(act_probs)]
            
        # 如果返回概率，创建整个棋盘的动作概率分布
        if return_prob:
            act_prob_dict = {a: p for a, p in zip(actions, act_probs)}
            board_size = len(board_state)
            move_probs = np.zeros(board_size * board_size)
            for move_a, prob in act_prob_dict.items():
                row, col = move_a
                idx = row * board_size + col
                move_probs[idx] = prob
            return move, move_probs
            
        return move
    
    def _check_winning_move(self, board_state):
        """检查是否有直接获胜的走法
        
        Args:
            board_state: 当前棋盘状态
            
        Returns:
            tuple: (row, col) 获胜位置，如果没有则返回None
        """
        # 确定当前玩家
        if isinstance(board_state, np.ndarray):
            black_stones = (board_state == 1).sum()
            white_stones = (board_state == 2).sum()
        else:
            black_stones = sum(row.count(1) for row in board_state)
            white_stones = sum(row.count(2) for row in board_state)
        
        current_player = 1 if black_stones <= white_stones else 2
        
        # 检查每个空位，看是否形成五连
        board_size = len(board_state)
        for row in range(board_size):
            for col in range(board_size):
                if board_state[row][col] != 0:  # 跳过非空位置
                    continue
                
                # 临时落子
                board_state[row][col] = current_player
                
                # 检查是否形成五连
                if self._check_win(board_state, row, col, current_player):
                    # 恢复棋盘并返回获胜位置
                    board_state[row][col] = 0
                    return (row, col)
                
                # 恢复棋盘
                board_state[row][col] = 0
        
        return None
    
    def _check_win(self, board, row, col, player):
        """检查是否形成五连获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、两个对角线
        board_size = len(board)
        
        for dx, dy in directions:
            count = 1  # 当前位置已经算一个
            
            # 正方向检查
            for step in range(1, 5):  # 最多再检查4步，总共5个
                nx, ny = row + dx * step, col + dy * step
                if 0 <= nx < board_size and 0 <= ny < board_size and board[nx][ny] == player:
                    count += 1
                else:
                    break
            
            # 反方向检查
            for step in range(1, 5):
                nx, ny = row - dx * step, col - dy * step
                if 0 <= nx < board_size and 0 <= ny < board_size and board[nx][ny] == player:
                    count += 1
                else:
                    break
            
            # 判断是否达到五连
            if count >= 5:
                return True
                
        return False
    
    def _playout(self, node, check_interrupt=None):
        """执行一次模拟
        
        Args:
            node: 开始模拟的节点
            check_interrupt: 中断检查函数
            
        Returns:
            float: 最终得到的价值
            如果被中断，返回False
        """
        # 检查是否中断
        if check_interrupt and check_interrupt():
            return False
            
        # 检查节点是否已经终止（游戏结束）
        try:
            is_terminal, winner = self._check_game_end(node.board_state)
        except Exception as e:
            print(f"检查游戏结束时出错: {e}")
            return False
            
        if is_terminal:
            # 确定终止节点的价值
            if winner == 0:  # 平局
                value = 0.0
            else:
                # 如果当前节点玩家是赢家，价值为1；否则为-1
                value = 1.0 if winner == node.player else -1.0
            
            # 返回值 (从当前玩家角度)
            return -value
        
        # 如果节点未扩展，进行扩展并返回值估计
        if not node._expanded:
            try:
                # 使用神经网络获取动作概率和状态价值
                action_probs, value = self._policy_value_fn(node.board_state, node.player)
                
                # 根据动作概率扩展节点
                valid_actions_probs = []
                for action, prob in action_probs:
                    row, col = action
                    if (0 <= row < len(node.board_state) and 
                        0 <= col < len(node.board_state) and 
                        node.board_state[row][col] == 0):
                        valid_actions_probs.append((action, prob))
                
                if valid_actions_probs:  # 确保有有效动作
                    node.expand(valid_actions_probs)
                
                # 增强回报：使用棋型评估来调整神经网络的评估值
                if hasattr(self, 'pattern_evaluator'):
                    pattern_score = self.pattern_evaluator.evaluate_board(node.board_state, node.player)
                    if abs(pattern_score) > 0:
                        # 将棋型分数归一化到[-0.5, 0.5]范围，然后加到神经网络评估值上
                        normalized_score = np.clip(pattern_score / 10000, -0.5, 0.5)
                        value = np.clip(value + normalized_score, -1.0, 1.0)
            except Exception as e:
                print(f"节点扩展或评估时出错: {e}")
                return False
            
            # 从当前玩家角度返回负值（因为切换玩家）
            return -value
        
        # 检查中断
        if check_interrupt and check_interrupt():
            return False
            
        # 选择最佳子节点进行递归搜索
        try:
            best_child, best_action = node.select(self.c_puct)
            
            # 如果没有可选择的子节点，返回0（中性评估）
            if best_child is None:
                return 0.0
                
            value = self._playout(best_child, check_interrupt)
        except Exception as e:
            print(f"MCTS搜索过程出错: {e}")
            return False
        
        # 如果子节点搜索被中断，终止当前搜索
        if value is False:
            return False
        
        # 更新当前节点
        try:
            node.update(-value)  # 注意负号，因为是从对手角度
        except Exception as e:
            print(f"更新节点时出错: {e}")
            return False
            
        return -value
    
    def _policy_value_fn(self, board_state, current_player):
        """策略-价值函数：使用神经网络评估棋盘状态
        
        Args:
            board_state: 棋盘状态
            current_player: 当前玩家(1或2)
            
        Returns:
            动作概率列表和状态价值
        """
        # 将棋盘转换为网络输入格式
        state_tensor = board_to_tensor(board_state, current_player)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
        
        # 使用神经网络进行前向传播
        self.policy_value_net.eval()
        with torch.no_grad():
            log_action_probs, value = self.policy_value_net(state_tensor)
            action_probs = torch.exp(log_action_probs).cpu().numpy().flatten()
            
        # 转换为(action, prob)列表
        board_size = len(board_state)
        actions_probs = []
        for action_idx in range(board_size * board_size):
            row, col = action_idx // board_size, action_idx % board_size
            if board_state[row][col] == 0:  # 只考虑空位
                actions_probs.append(((row, col), action_probs[action_idx]))
        
        # 检查是否有获胜走法，如果有，大幅提高其概率
        winning_move = self._check_winning_move(board_state)
        if winning_move:
            row, col = winning_move
            winning_idx = row * board_size + col
            
            # 标准化后极大地提高获胜走法的概率
            action_probs = np.ones_like(action_probs) * 0.001
            action_probs[winning_idx] = 0.999
        
        # 返回结果
        return actions_probs, value.item()
    
    def _check_game_end(self, board_state):
        """检查游戏是否结束
        
        Returns:
            tuple: (is_end, winner), winner: 0表示平局，1表示黑棋胜，2表示白棋胜
        """
        # 检查是否有五连
        for player in [1, 2]:
            for i in range(len(board_state)):
                for j in range(len(board_state[0])):
                    if board_state[i][j] == player:
                        # 检查四个方向
                        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                            # 检查是否有五连
                            count = 1  # 当前位置已有一个
                            for step in range(1, 5):
                                x, y = i + dx * step, j + dy * step
                                if (0 <= x < len(board_state) and 
                                    0 <= y < len(board_state[0]) and 
                                    board_state[x][y] == player):
                                    count += 1
                                else:
                                    break
                            
                            if count >= 5:
                                return True, player
        
        # 检查是否所有位置已满（平局）
        if all(board_state[i][j] != 0 for i in range(len(board_state)) for j in range(len(board_state[0]))):
            return True, 0
        
        # 游戏还在进行中
        return False, -1


class NeuralMCTSPlayer(BaseAIPlayer):
    """基于神经网络和MCTS的AI玩家"""
    
    def __init__(self, level=AILevel.EXPERT, model_path=None):
        super().__init__(level)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"初始化NeuralMCTSPlayer，使用设备: {self.device}")
        self.model_path = None
        self.set_model(model_path)
    
    def set_model(self, model_path):
        """设置使用的模型"""
        print(f"\n===== 尝试加载AI模型 =====")
        print(f"模型路径: {model_path or '使用默认模型'}")
        
        # 尝试导入模型创建函数
        try:
            from ai.models import create_gomoku_model
        except ImportError as e:
            print(f"错误: 无法导入模型创建函数: {e}")
            return False
            
        # 保存模型路径以供后续参考
        self.model_path = model_path
            
        # 如果指定了模型路径，尝试加载它
        if model_path and os.path.exists(model_path):
            try:
                print(f"开始加载模型文件: {model_path}")
                
                # 1. 先创建网络结构
                self.model = create_gomoku_model(board_size=15, device=self.device)
                
                # 2. 记录加载前的模型参数信息
                pre_params = next(iter(self.model.parameters()))
                pre_sum = pre_params.sum().item()
                print(f"加载前参数检查: 首层参数和={pre_sum:.6f}")
                
                # 3. 加载模型权重
                state_dict = torch.load(model_path, map_location=self.device)
                
                # 4. 验证权重结构
                model_keys = set(self.model.state_dict().keys())
                load_keys = set(state_dict.keys())
                
                if model_keys != load_keys:
                    missing = model_keys - load_keys
                    extra = load_keys - model_keys
                    if missing:
                        print(f"警告: 模型缺少参数: {missing}")
                    if extra:
                        print(f"警告: 模型包含额外参数: {extra}")
                
                # 5. 应用权重到模型
                self.model.load_state_dict(state_dict)
                self.model.eval()  # 设置为评估模式
                
                # 6. 加载后验证参数是否变化
                post_params = next(iter(self.model.parameters()))
                post_sum = post_params.sum().item()
                print(f"加载后参数检查: 首层参数和={post_sum:.6f}")
                
                # 7. 验证模型参数是否已更改（加载成功）
                if abs(pre_sum - post_sum) < 1e-6:
                    print("警告: 模型参数加载可能不成功，前后参数几乎相同")
                else:
                    print("模型参数已成功更改，确认加载成功")
                
                # 8. 输出模型统计信息
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"模型参数统计: 总数={total_params:,}, 可训练={trainable_params:,}")
                print(f"模型结构类型: {type(self.model).__name__}")
                
                return True
                
            except Exception as e:
                print(f"加载模型失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 无论是没指定模型路径，还是加载失败，都创建默认模型
        try:
            print("创建新的默认模型")
            self.model = create_gomoku_model(board_size=15, device=self.device)
            self.model.eval()
            
            # 输出默认模型统计信息
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"默认模型参数统计: 总数={total_params:,}, 可训练={trainable_params:,}")
            
            return True
        except Exception as e:
            print(f"创建默认模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def think(self, board, color):
        """思考下一步"""
        # 在每次思考前都确认模型状态
        if self.model is None:
            print("AI思考: 没有可用模型，将使用随机落子")
            return self.random_move(board)
        
        try:
            # 记录推理状态和路径
            print(f"AI思考: 使用{'预加载模型' if self.model_path else '默认模型'}")
            if self.model_path:
                print(f"当前模型路径: {self.model_path}")
            
            # 将颜色转换为玩家ID (1=黑, 2=白)
            player_id = 1 if color == StoneColor.BLACK else 2
            
            # 将棋盘转换为神经网络输入格式
            state_tensor = torch.FloatTensor(board_to_tensor(board, player_id)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 获取策略和价值
                policy_logits, value = self.model(state_tensor)
                policy = F.softmax(policy_logits.squeeze(), dim=0).cpu().numpy()
                print(f"模型输出价值评估: {value.item():.4f}")
            
            # 过滤已有棋子的位置
            valid_moves = (board.reshape(-1) == 0).astype(np.float32)
            policy = policy * valid_moves
            
            # 如果没有有效移动，返回随机移动
            if policy.sum() == 0:
                return self.random_move(board)
            
            # 根据策略选择移动，考虑探索（根据难度级别）
            if self.level == AILevel.EASY:
                # 添加更多随机性
                policy = np.power(policy, 0.5)  # 减小概率差距
            elif self.level == AILevel.MEDIUM:
                # 中等随机性
                policy = np.power(policy, 0.75)
            # EXPERT级别使用原始策略
            
            # 重新归一化
            policy = policy / policy.sum()
            
            # 根据策略选择动作
            move_index = np.random.choice(len(policy), p=policy)
            y = move_index // 15
            x = move_index % 15
            
            return (x, y)
            
        except Exception as e:
            print(f"神经网络思考出错: {str(e)}")
            return self.random_move(board)
    
    def random_move(self, board):
        """随机选择一个有效移动"""
        empty_positions = []
        for y in range(15):
            for x in range(15):
                if board[y][x] == 0:
                    empty_positions.append((x, y))
        
        if empty_positions:
            return empty_positions[np.random.randint(0, len(empty_positions))]
        return None  # 棋盘已满
