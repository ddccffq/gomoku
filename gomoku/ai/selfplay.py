import numpy as np
import torch
import torch.nn.functional as F
import random
import time
from multiprocessing import Pool, cpu_count
from .data_handler import board_to_tensor
from .board_evaluator import GomokuPatternEvaluator
import matplotlib.pyplot as plt
import os
import datetime

class Node:
    """MCTS节点类"""
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.state = None
        
    def expanded(self):
        """检查节点是否已展开"""
        return len(self.children) > 0
    
    def value(self):
        """返回节点的平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    """蒙特卡洛树搜索实现"""
    def __init__(self, model, board_size=15, num_simulations=800, c_puct=5.0, device='cpu'):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
    
    def search(self, state):
        """执行MCTS搜索"""
        root = Node(0)
        root.state = state
        
        # 展开根节点
        self._expand_node(root)
        
        # 执行模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # 选择
            while node.expanded():
                action, node = self._select_child(node)
                self._apply_action(current_state, action)
                search_path.append(node)
            
            # 展开
            value = self._evaluate_and_expand(node, current_state)
            
            # 反向传播
            self._backpropagate(search_path, value)
        
        # 计算根节点的访问计数分布
        visit_counts = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 归一化
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            # 如果所有访问计数都是0，使用均匀分布
            policy = np.ones_like(visit_counts) / len(visit_counts)
        
        return policy
    
    def _select_child(self, node):
        """使用PUCT算法选择最佳子节点"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # 计算总访问计数
        sum_visits = sum(child.visit_count for child in node.children.values())
        
        # 考虑所有可能的行动
        for action, child in node.children.items():
            # UCB公式
            exploit = child.value()
            explore = self.c_puct * child.prior * (np.sqrt(sum_visits) / (1 + child.visit_count))
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand_node(self, node):
        """展开节点，添加所有可能的子节点"""
        state_tensor = torch.FloatTensor(self._encode_state(node.state)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        # 获取策略和价值
        policy = F.softmax(policy_logits.squeeze(), dim=0).cpu().numpy()
        value = value.item()
        
        # 过滤无效动作
        valid_moves = self._get_valid_moves(node.state)
        policy = policy * valid_moves
        
        # 如果所有动作都无效，给一个均匀分布
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = valid_moves / valid_moves.sum()
        
        # 为每个有效动作创建子节点
        for action in range(len(policy)):
            if valid_moves[action]:
                node.children[action] = Node(prior=policy[action])
    
    def _evaluate_and_expand(self, node, state):
        """评估并展开叶子节点"""
        # 检查游戏是否结束
        game_over, winner = self._check_game_over(state)
        
        if game_over:
            # 游戏结束，直接返回结果
            return 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0
        
        # 游戏未结束，展开节点
        node.state = state.copy()
        self._expand_node(node)
        
        # 使用神经网络评估当前状态
        state_tensor = torch.FloatTensor(self._encode_state(state)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.model(state_tensor)
        
        return value.item()
    
    def _backpropagate(self, search_path, value):
        """反向传播更新节点统计信息"""
        # 当前玩家视角的值
        player = 1  # 假设根节点是玩家1(黑棋)的回合
        
        for node in reversed(search_path):
            node.value_sum += value if player == 1 else -value
            node.visit_count += 1
            player = 3 - player  # 切换玩家 1->2, 2->1
    
    def _apply_action(self, state, action):
        """将动作应用到状态上"""
        y, x = divmod(action, self.board_size)
        current_player = self._get_current_player(state)
        state[y, x] = current_player
    
    def _get_current_player(self, state):
        """获取当前玩家"""
        black_stones = (state == 1).sum()
        white_stones = (state == 2).sum()
        
        # 如果黑棋数量等于白棋数量，则轮到黑棋(1)
        # 否则轮到白棋(2)
        return 1 if black_stones <= white_stones else 2
    
    def _encode_state(self, state):
        """将游戏状态编码为神经网络输入"""
        # 创建3通道表示:
        # 通道0: 当前玩家的棋子 (1表示存在，0表示不存在)
        # 通道1: 对手的棋子 (1表示存在，0表示不存在)
        # 通道2: 当前玩家是否是黑棋 (全1或全0)
        
        current_player = self._get_current_player(state)
        opponent = 3 - current_player  # 1->2, 2->1
        
        encoded = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 设置当前玩家的棋子
        encoded[0] = (state == current_player)
        
        # 设置对手的棋子
        encoded[1] = (state == opponent)
        
        # 设置当前玩家是否是黑棋
        encoded[2] = 1.0 if current_player == 1 else 0.0
        
        return encoded
    
    def _get_valid_moves(self, state):
        """获取所有有效动作"""
        # 空位为有效动作
        return (state.reshape(-1) == 0).astype(np.float32)
    
    def _check_game_over(self, state):
        """检查游戏是否结束"""
        # 检查每个玩家是否有五子连珠
        for player in [1, 2]:
            # 检查水平方向
            for y in range(self.board_size):
                for x in range(self.board_size - 4):
                    if all(state[y, x+i] == player for i in range(5)):
                        return True, player
            
            # 检查垂直方向
            for y in range(self.board_size - 4):
                for x in range(self.board_size):
                    if all(state[y+i, x] == player for i in range(5)):
                        return True, player
            
            # 检查右下对角线
            for y in range(self.board_size - 4):
                for x in range(self.board_size - 4):
                    if all(state[y+i, x+i] == player for i in range(5)):
                        return True, player
            
            # 检查左下对角线
            for y in range(self.board_size - 4):
                for x in range(4, self.board_size):
                    if all(state[y+i, x-i] == player for i in range(5)):
                        return True, player
        
        # 检查是否有空位
        if not (state == 0).any():
            return True, 0  # 平局
        
        # 游戏未结束
        return False, 0

from .mcts_nn import NeuralMCTS

class SelfPlayManager:
    """自我对弈管理器"""
    
    def __init__(self, model, board_size=15, mcts_simulations=800, device='cpu', exploration_temp=1.0):
        self.model = model
        self.board_size = board_size
        self.device = device
        self.mcts = NeuralMCTS(model, c_puct=5, n_playout=mcts_simulations, is_selfplay=True)
        self.exploration_temp = exploration_temp  # 探索温度参数
        self.pattern_evaluator = GomokuPatternEvaluator(board_size)  # 添加棋型评估器
        
        # 创建本次训练的图像保存目录
        self.training_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'game_images')
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.training_dir = os.path.join(self.base_save_dir, f"training_{self.training_id}")
        os.makedirs(self.training_dir, exist_ok=True)
        print(f"本次训练的棋盘图像将保存至: {self.training_dir}")
    
    def play_game(self, board_callback=None, log_patterns=False, check_interrupt=None, save_all_boards=False):
        """进行一局自我对弈游戏，生成训练数据
        
        Args:
            board_callback: 回调函数，用于实时显示棋盘状态
            log_patterns: 是否记录棋型评估日志
            check_interrupt: 回调函数，用于检查是否应该中断
            save_all_boards: 是否保存每一步的棋盘图像，默认False只保存最终获胜棋盘
            
        Returns:
            tuple: (states, policies, values) 或 (states, policies, values, pattern_scores)
        """
        # 初始化空棋盘
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        states, policies, values = [], [], []
        pattern_scores = []  # 记录每步的棋型评估分数
        current_player = 1  # 黑棋先行
        game_over = False
        winner = None
        move_history = []
        move_counter = 0  # 使用从0开始的计数器
        last_move = None  # 初始化

        # 生成游戏唯一的ID
        game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 游戏循环
            while not game_over:
                # 首先检查是否有获胜走法
                winning_move = None
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        if board[row][col] == 0:  # 只考虑空位
                            # 模拟落子
                            board[row][col] = current_player
                            
                            # 检查是否能形成五连
                            result, winner = self._check_game_end(board)
                            if result and winner == current_player:
                                winning_move = (row, col)
                                board[row][col] = 0  # 恢复棋盘
                                break
                            
                            # 恢复棋盘
                            board[row][col] = 0
                    
                    if winning_move:
                        break
                
                # 如果有获胜走法，直接使用
                if winning_move:
                    move = winning_move
                    # 创建一个集中于获胜走法的策略
                    policy = np.zeros(self.board_size * self.board_size)
                    policy[winning_move[0] * self.board_size + winning_move[1]] = 1.0
                else:
                    # 使用MCTS生成走法概率分布 - 根据游戏阶段调整温度
                    temp = self.exploration_temp if len(move_history) < 10 else 0.5  # 前10步使用高温度增加探索
                    try:
                        move, policy = self.mcts.get_action(board, temp=temp, return_prob=True, check_interrupt=check_interrupt)
                        
                        # 如果MCTS搜索返回None，表示被中断
                        if move is None:
                            print("MCTS搜索被中断")
                            return [], [], []  # 返回空数据，表示中断
                    except Exception as e:
                        print(f"MCTS搜索出错: {e}")
                        import traceback
                        traceback.print_exc()
                        return [], [], []  # 搜索出错也返回空数据
                
                # 保存当前状态、策略和玩家信息
                states.append(board_to_tensor(board.copy(), current_player))
                policies.append(policy)

                # 应用走法
                row, col = move
                board[row][col] = current_player

                # 添加到历史记录（仅此一次）
                move_history.append(move)
                
                # 增加计数器 - 在添加历史记录后增加
                move_counter += 1
                
                # 仅在与上次不同才回调
                if board_callback:
                    current = (move_counter, current_player, move)
                    if current != last_move:
                        last_move = current
                        board_callback(board.copy(), move_history.copy(), current_player)
                
                # 检查胜负
                result, winner = self._check_game_end(board)
                if result:
                    game_over = True
                    values = [1.0 if ((i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == 2)) else -1.0 
                             for i in range(len(states))]
                    
                    print(f"游戏结束，赢家: {winner}, 步数: {len(move_history)}, 样本: {len(states)}")
                    
                    # 回调通知游戏结束
                    if board_callback:
                        try:
                            # 同样处理参数不匹配问题
                            import inspect
                            callback_params = len(inspect.signature(board_callback).parameters)
                            if callback_params == 3:
                                board_callback(board.copy(), move_history.copy(), 0)  # 0表示游戏结束
                            elif callback_params >= 4:
                                board_callback(board.copy(), move_history.copy(), 0, None)
                        except Exception as e:
                            print(f"结束游戏回调出错: {e}")
                    
                    # 无论save_all_boards如何设置，总是保存最终棋盘和完整序列
                    try:
                        self._save_board_image(board.copy(), move_history.copy(), 0, game_id, move_counter, is_final=True)
                        self._save_final_board_with_moves(board.copy(), move_history.copy(), game_id)
                    except Exception as e:
                        print(f"保存最终棋盘图像失败: {e}")
                    
                    if len(states) == 0:
                        print("⚠️ 警告: 生成的训练样本为空")
                    else:
                        print(f"生成训练数据: {len(states)} 个样本, 第一个状态形状: {states[0].shape}")
                    
                    return (states, policies, values) if not log_patterns else (states, policies, values, pattern_scores)
                
                # 切换玩家
                current_player = 3 - current_player  # 1->2, 2->1
                
                # 增加小延迟，使界面能够看清每步变化
                if board_callback:  # 只在有可视化时添加延迟
                    import time
                    try:
                        # 小间隔频繁检查中断，以便及时响应
                        for _ in range(10):  # 10次检查，共100ms
                            if check_interrupt and check_interrupt():
                                print("自我对弈在延迟期间被中断")
                                return [], [], []
                            time.sleep(0.01)  # 10ms
                    except Exception as e:
                        print(f"延迟处理中出错: {e}")
                        # 继续游戏，不中断

            return (states, policies, values) if not log_patterns else (states, policies, values, pattern_scores)
        except Exception as e:
            print(f"❌ 自我对弈过程中出错: {e}")
            import traceback
            print(traceback.format_exc())
            return [], [], []

    def _save_board_image(self, board, move_history, current_player, game_id, move_number, is_final=False):
        """保存棋盘状态为图像"""
        try:
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 画棋盘背景和网格
            ax.set_facecolor('#F0C070')  # 棋盘背景色
            for i in range(self.board_size):
                ax.axhline(y=i, color='black', linewidth=0.5)
                ax.axvline(x=i, color='black', linewidth=0.5)
            
            # 绘制棋盘上的点（天元和星位）
            star_positions = [3, 7, 11] if self.board_size == 15 else [3, self.board_size//2, self.board_size-4]
            for y in star_positions:
                for x in star_positions:
                    ax.plot(x, y, 'o', color='black', markersize=8)
            
            # 画棋子
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board[y][x] == 1:  # 黑棋
                        ax.plot(x, y, 'o', color='black', markersize=15, markeredgecolor='black')
                    elif board[y][x] == 2:  # 白棋
                        ax.plot(x, y, 'o', color='white', markersize=15, markeredgecolor='black')
            
            # 标记最后一手棋
            if move_history:
                last_y, last_x = move_history[-1]
                ax.plot(last_x, last_y, 'x', color='red', markersize=10)
            
            # 设置坐标范围和标签
            ax.set_xlim(-0.5, self.board_size - 0.5)
            ax.set_ylim(-0.5, self.board_size - 0.5)
            ax.invert_yaxis()  # 反转Y轴，使(0,0)显示在左上角
            
            # 移除刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加标题
            if current_player == 0:
                if len(move_history) > 0:
                    last_row, last_col = move_history[-1]
                    winner_text = "黑棋胜" if board[last_row][last_col] == 1 else "白棋胜"
                    ax.set_title(f"游戏 {game_id} - 结束 - {winner_text}", fontsize=14)
                else:
                    ax.set_title(f"游戏 {game_id} - 结束 - 和局", fontsize=14)
            else:
                player_text = "黑棋" if current_player == 1 else "白棋"
                ax.set_title(f"游戏 {game_id} - 回合 {move_number} - {player_text}落子", fontsize=14)
            
            # 保存图像到当前训练的目录中
            filename_suffix = "final" if is_final else f"move_{move_number:03d}"
            filename = f"game_{game_id}_{filename_suffix}.png"
            filepath = os.path.join(self.training_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"保存棋盘图像失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_final_board_with_moves(self, board, move_history, game_id):
        """保存带有落子顺序的最终棋盘图像"""
        try:
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # 画棋盘背景和网格
            ax.set_facecolor('#F0C070')
            for i in range(self.board_size):
                ax.axhline(y=i, color='black', linewidth=0.5)
                ax.axvline(x=i, color='black', linewidth=0.5)
            
            # 绘制棋盘上的点
            star_positions = [3, 7, 11] if self.board_size == 15 else [3, self.board_size//2, self.board_size-4]
            for y in star_positions:
                for x in star_positions:
                    ax.plot(x, y, 'o', color='black', markersize=8)
            
            # 画棋子并标记落子顺序
            for i, (y, x) in enumerate(move_history):
                color = 'black' if i % 2 == 0 else 'white'
                ax.plot(x, y, 'o', color=color, markersize=15, markeredgecolor='black')
                
                # 添加落子顺序编号
                text_color = 'white' if color == 'black' else 'black'
                ax.text(x, y, str(i+1), color=text_color, ha='center', va='center', fontsize=8)
            
            # 设置坐标范围和标签
            ax.set_xlim(-0.5, self.board_size - 0.5)
            ax.set_ylim(-0.5, self.board_size - 0.5)
            ax.invert_yaxis()
            
            # 移除刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 确定胜者
            winner_text = "和局"
            if len(move_history) > 0:
                last_idx = len(move_history) - 1
                winner_player = 1 if last_idx % 2 == 0 else 2
                winner_text = "黑棋胜" if winner_player == 1 else "白棋胜"
            
            # 添加标题
            ax.set_title(f"游戏 {game_id} - 完整落子顺序 - {winner_text}", fontsize=14)
            
            # 保存图像到当前训练的目录中
            filename = f"game_{game_id}_full_sequence.png"
            filepath = os.path.join(self.training_dir, filename)
            plt.savefig(filepath, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"保存带落子顺序的棋盘图像失败: {e}")
            import traceback
            traceback.print_exc()

    def _invert_state(self, state):
        """反转棋盘状态（黑白交换）"""
        inverted = state.copy()
        inverted[state == 1] = 2
        inverted[state == 2] = 1
        return inverted
    
    def _check_game_end(self, state):
        """检查游戏是否结束，返回(是否结束, 胜者)"""
        # 检查胜利条件 - 五子连珠
        for player in [1, 2]:
            # 横向检查
            for row in range(self.board_size):
                for col in range(self.board_size - 4):
                    if all(state[row][col+i] == player for i in range(5)):
                        return True, player
            
            # 纵向检查
            for row in range(self.board_size - 4):
                for col in range(self.board_size):
                    if all(state[row+i][col] == player for i in range(5)):
                        return True, player
            
            # 正对角线检查
            for row in range(self.board_size - 4):
                for col in range(self.board_size - 4):
                    if all(state[row+i][col+i] == player for i in range(5)):
                        return True, player
            
            # 反对角线检查
            for row in range(self.board_size - 4):
                for col in range(4, self.board_size):
                    if all(state[row+i][col-i] == player for i in range(5)):
                        return True, player
        
        # 检查是否平局（棋盘已满）
        if np.all(state != 0):
            return True, 0
        
        # 游戏尚未结束
        return False, -1

    def generate_data(self, num_games):
        """并行化生成自玩数据"""
        with Pool(min(cpu_count(), num_games)) as pool:
            results = pool.map(self._play_single, range(num_games))
        # results: list of (states, policies, values)
        states, policies, values = zip(*results)
        return sum(states, []), sum(policies, []), sum(values, [])

    def _play_single(self, idx):
        """单局自玩（供Pool调用）"""
        return self.play_game(board_callback=None)

    @staticmethod
    def evaluate_models(model_old, model_new, n_games=50, board_size=15, sims=200):
        """让新旧模型对抗，胜率高者胜出"""
        manager_old = SelfPlayManager(model_old, board_size, sims)
        manager_new = SelfPlayManager(model_new, board_size, sims)
        wins_new = 0
        for i in range(n_games):
            # 黑新白旧，交叉对弈
            winner = SelfPlayManager._play_vs(manager_new, manager_old, first_black=True)
            if winner == 'new': wins_new += 1
            winner = SelfPlayManager._play_vs(manager_old, manager_new, first_black=False)
            if winner == 'new': wins_new += 1
        return wins_new / (2*n_games) > 0.55  # 新模型胜率>55%则替换

    @staticmethod
    def _play_vs(mgr_black, mgr_white, first_black=True):
        state = np.zeros((mgr_black.board_size, mgr_black.board_size), np.int8)
        current_mgr = mgr_black if first_black else mgr_white
        current_player = 1 if first_black else 2
        while True:
            # 修复：使用正确的方法获取走法和策略
            move, policy = current_mgr.mcts.get_action(state, temp=0.1, return_prob=True)
            if move:  # 确保move不是None
                row, col = move
                state[row][col] = current_player
                over, winner = mgr_black._check_game_end(state)
                if over:
                    if winner == 0: return 'draw'
                    # 新模型为 mgr_black if first_black else mgr_white
                    return 'new' if (winner == 1 and first_black) or (winner == 2 and not first_black) else 'old'
            else:
                # 没有可行走法，判断为平局
                return 'draw'
            current_mgr = mgr_white if current_mgr is mgr_black else mgr_black
            current_player = 3 - current_player
