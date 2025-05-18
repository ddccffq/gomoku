# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import torch
from copy import deepcopy

# 修复这行导入，使用 BaseAIPlayer 而不是 BaseAI
from .base_ai import BaseAIPlayer, StoneColor, AILevel

class MCTSNode:
    """Monte Carlo Tree Search节点"""
    
    def __init__(self, board_state, parent=None, move=None):
        self.board_state = board_state  # 棋盘状态，一个二维列表
        self.parent = parent  # 父节点
        self.move = move  # 到达这个节点的走法，(行, 列)格式
        
        self.children = []  # 子节点列表
        self.wins = 0  # 获胜次数
        self.visits = 0  # 访问次数
        self.untried_moves = self._get_valid_moves()  # 未尝试的走法
        self.player = self._get_player()  # 当前要走棋的玩家
    
    def _get_valid_moves(self):
        """获取所有有效走法"""
        valid_moves = []
        for row in range(len(self.board_state)):
            for col in range(len(self.board_state[0])):
                if self.board_state[row][col] == 0:  # 空位
                    valid_moves.append((row, col))
        return valid_moves
    
    def _get_player(self):
        """确定当前节点的玩家"""
        # 计算棋盘上黑白棋子数量
        # 修改为适应numpy数组的计数方式
        if isinstance(self.board_state, np.ndarray):
            black_count = np.sum(self.board_state == 1)
            white_count = np.sum(self.board_state == 2)
        else:
            # 如果是普通列表，继续使用list.count
            black_count = sum(row.count(1) for row in self.board_state)
            white_count = sum(row.count(2) for row in self.board_state)
        
        return 1 if black_count == white_count else 2
    
    def select_child(self):
        """使用UCB1公式选择最有价值的子节点"""
        # 对于0次访问的节点，设置一个很大的值确保它被选择
        exploration = math.sqrt(2.0)
        return max(self.children, key=lambda child: 
                  (child.wins / child.visits + exploration * math.sqrt(math.log(self.visits) / child.visits))
                  if child.visits > 0 else float('inf'))
    
    def expand(self):
        """扩展一个未尝试的走法"""
        if not self.untried_moves:
            return None
            
        # 随机选择一个未尝试的走法
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # 创建新的棋盘状态
        new_state = deepcopy(self.board_state)
        row, col = move
        new_state[row][col] = self.player
        
        # 创建子节点
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """更新节点的统计信息"""
        self.visits += 1
        self.wins += result
    
    def is_terminal(self):
        """检查是否达到终止状态（游戏结束）"""
        # 检查是否有人赢得游戏
        for row in range(len(self.board_state)):
            for col in range(len(self.board_state[0])):
                if self.board_state[row][col] != 0:  # 有棋子的位置
                    if self._check_win(row, col):
                        return True
        
        # 检查是否棋盘已满
        for row in self.board_state:
            if 0 in row:
                return False
        return True
    
    def _check_win(self, row, col):
        """检查指定位置的棋子是否形成五连"""
        player = self.board_state[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、右斜、左斜
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 向两个方向检查
            for direction in [1, -1]:
                for step in range(1, 5):  # 最多检查4步
                    x, y = row + dx * step * direction, col + dy * step * direction
                    if (0 <= x < len(self.board_state) and 
                        0 <= y < len(self.board_state[0]) and 
                        self.board_state[x][y] == player):
                        count += 1
                    else:
                        break
            
            if count >= 5:
                return True
                
        return False
    
    def get_result(self, player):
        """从当前玩家角度获取模拟结果"""
        for row in range(len(self.board_state)):
            for col in range(len(self.board_state[0])):
                if self.board_state[row][col] != 0:  # 有棋子的位置
                    if self._check_win(row, col):
                        winner = self.board_state[row][col]
                        return 1 if winner == player else 0
        
        # 平局
        return 0.5


class MCTSAI(BaseAIPlayer):
    """基于蒙特卡洛树搜索的AI"""
    
    def __init__(self, color, level=AILevel.HARD):
        super().__init__(color, level)
        # 根据难度设置迭代次数
        if level == AILevel.EASY:
            self.iterations = 500
        elif level == AILevel.HARD:
            self.iterations = 1000
        else:  # EXPERT
            self.iterations = 2000
    
    def get_move(self, board_state):
        """根据当前棋盘状态获取最优走法"""
        # 确保board_state是一个15x15的二维列表
        if len(board_state) != 15 or len(board_state[0]) != 15:
            raise ValueError("棋盘必须是15x15的")
            
        # 创建根节点
        root = MCTSNode(board_state)
        
        # 如果根节点是终止状态，直接返回
        if root.is_terminal():
            return None
        
        # 如果只有一个可能的走法，直接返回
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        # 执行蒙特卡洛树搜索
        for _ in range(self.iterations):
            node = root
            board = deepcopy(board_state)
            
            # 选择阶段: 选择最有价值的子节点直到找到未完全扩展的节点
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                if node.move:
                    row, col = node.move
                    board[row][col] = 3 - node.parent._get_player()  # 交替玩家
            
            # 扩展阶段: 除非达到终止状态，否则添加一个新的子节点
            if not node.is_terminal():
                node = node.expand()
                if node:  # 如果扩展成功
                    row, col = node.move
                    board[row][col] = node.parent._get_player()
            
            # 模拟阶段: 随机游戏直至游戏结束
            current_player = node._get_player() if node else root._get_player()
            simulation_board = deepcopy(board)
            simulation_result = self._simulate(simulation_board, current_player)
            
            # 回溯阶段: 更新所有经过的节点的统计信息
            while node:
                # 从当前玩家角度更新结果
                result = simulation_result if node._get_player() == self.color_to_player() else 1 - simulation_result
                node.update(result)
                node = node.parent
        
        # 选择访问次数最多的子节点作为最终走法
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def _simulate(self, board, player):
        """随机模拟游戏直至结束"""
        current_player = player
        
        while True:
            # 检查是否有人获胜
            for row in range(len(board)):
                for col in range(len(board[0])):
                    if board[row][col] != 0:  # 有棋子的位置
                        if self._check_win(board, row, col):
                            winner = board[row][col]
                            # 返回从AI角度看的结果
                            if winner == self.color_to_player():
                                return 1  # AI赢
                            else:
                                return 0  # AI输
            
            # 获取所有可能的走法
            valid_moves = []
            for row in range(len(board)):
                for col in range(len(board[0])):
                    if board[row][col] == 0:  # 空位
                        valid_moves.append((row, col))
            
            # 如果没有合法走法，返回平局
            if not valid_moves:
                return 0.5  # 平局
                
            # 随机选择一个走法
            row, col = random.choice(valid_moves)
            board[row][col] = current_player
            
            # 切换玩家
            current_player = 3 - current_player  # 1->2, 2->1
    
    def _check_win(self, board, row, col):
        """检查指定位置的棋子是否形成五连"""
        player = board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、右斜、左斜
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 向两个方向检查
            for direction in [1, -1]:
                for step in range(1, 5):  # 最多检查4步
                    x, y = row + dx * step * direction, col + dy * step * direction
                    if (0 <= x < len(board) and 
                        0 <= y < len(board[0]) and 
                        board[x][y] == player):
                        count += 1
                    else:
                        break
            
            if count >= 5:
                return True
                
        return False
    
    def color_to_player(self):
        """将AI颜色转换为棋盘上的玩家编号"""
        return 1 if self.color == StoneColor.BLACK else 2


class MCTS:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self, model, n_playout=1000, device=None):
        """初始化MCTS搜索器
        
        Args:
            model: 策略价值网络模型
            n_playout: 每次决策的模拟次数
            device: 计算设备
        """
        self.model = model
        self.n_playout = n_playout
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = None
    
    def get_action(self, board_state, temp=1e-3, return_prob=False):
        """获取最佳动作
        
        Args:
            board_state: 当前棋盘状态，二维列表
            temp: 温度参数，控制探索程度（接近0则选择访问次数最多的动作）
            return_prob: 是否返回动作概率分布
            
        Returns:
            如果return_prob为False，仅返回最佳动作(row, col)
            如果return_prob为True，返回(最佳动作, 动作概率分布)
        """
        # 创建根节点
        self.root = MCTSNode(deepcopy(board_state))
        
        # 执行n_playout次模拟
        for _ in range(self.n_playout):
            # 复制当前棋盘状态
            board = deepcopy(board_state)
            # 从根节点开始搜索
            node = self.root
            
            # 选择阶段: 选择最有价值的子节点直到找到叶子节点或未完全扩展的节点
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                if node.move:
                    row, col = node.move
                    board[row][col] = 3 - node.parent.player  # 交替玩家
            
            # 扩展阶段: 如果节点还有未尝试的走法且不是终止状态，选择一个扩展
            if not node.is_terminal() and node.untried_moves:
                move = random.choice(node.untried_moves)
                row, col = move
                board[row][col] = node.player
                child = node.expand()
                node = child
            
            # 模拟阶段: 从当前节点快速随机模拟到游戏结束
            result = self._simulate_game(board, node.player)
            
            # 回溯阶段: 更新路径上所有节点的统计信息
            while node is not None:
                node.update(result)
                node = node.parent
                # 切换结果视角（因为是零和游戏）
                result = 1.0 - result
        
        # 基于访问次数计算动作概率
        actions = [(child.move, child.visits) for child in self.root.children]
        actions.sort(key=lambda x: x[1], reverse=True)
        
        # 使用温度参数调整概率分布
        if temp < 1e-3:  # 如果温度接近0，直接选择访问次数最多的动作
            best_action = actions[0][0] if actions else None
            action_probs = np.zeros(len(board_state) * len(board_state[0]))
            if best_action:
                row, col = best_action
                action_probs[row * len(board_state) + col] = 1.0
        else:  # 否则根据访问次数和温度参数计算概率分布
            visits = np.array([x[1] for x in actions])
            visits_temp = np.power(visits, 1.0/temp)
            probs = visits_temp / np.sum(visits_temp)
            
            # 转换为完整的动作概率分布
            action_probs = np.zeros(len(board_state) * len(board_state[0]))
            for i, (action, _) in enumerate(actions):
                row, col = action
                action_probs[row * len(board_state) + col] = probs[i]
            
            # 选择动作（在非确定性模式下可以根据概率分布随机选择）
            best_action = actions[0][0] if actions else None
        
        if return_prob:
            return best_action, action_probs
        else:
            return best_action
    
    def _simulate_game(self, board, player):
        """随机模拟游戏直至结束
        
        Args:
            board: 当前棋盘状态的副本
            player: 当前玩家
            
        Returns:
            模拟结果（从当前玩家视角）：1.0表示获胜，0.0表示失败，0.5表示平局
        """
        current_player = player
        
        while True:
            # 检查游戏是否结束
            for row in range(len(board)):
                for col in range(len(board[0])):
                    if board[row][col] != 0:
                        # 检查该位置是否形成五连
                        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                        for dx, dy in directions:
                            count = 1
                            # 正向检查
                            for step in range(1, 5):
                                x, y = row + dx * step, col + dy * step
                                if (0 <= x < len(board) and 0 <= y < len(board[0]) and 
                                    board[x][y] == board[row][col]):
                                    count += 1
                                else:
                                    break
                            # 反向检查
                            for step in range(1, 5):
                                x, y = row - dx * step, col - dy * step
                                if (0 <= x < len(board) and 0 <= y < len(board[0]) and 
                                    board[x][y] == board[row][col]):
                                    count += 1
                                else:
                                    break
                            
                            # 如果有一方获胜，返回结果
                            if count >= 5:
                                return 1.0 if board[row][col] == player else 0.0
            
            # 检查是否平局（棋盘已满）
            if all(board[r][c] != 0 for r in range(len(board)) for c in range(len(board[0]))):
                return 0.5
                
            # 获取所有可能的走法
            valid_moves = []
            for r in range(len(board)):
                for c in range(len(board[0])):
                    if board[r][c] == 0:
                        valid_moves.append((r, c))
            
            # 如果没有合法走法，返回平局
            if not valid_moves:
                return 0.5
                
            # 随机选择一个走法
            move = random.choice(valid_moves)
            r, c = move
            board[r][c] = current_player
            
            # 切换玩家
            current_player = 3 - current_player  # 1->2, 2->1
    
    # 兼容旧模块：将 search 暴露为 get_action
    get_action = search
