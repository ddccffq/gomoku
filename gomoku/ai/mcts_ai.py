# coding:utf-8
import random
import math
import numpy as np
from .base_ai import BaseAIPlayer, AILevel, StoneColor

class MCTSNode:
    """蒙特卡罗树搜索节点"""
    
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move  # 到达该节点的移动，(x, y)
        self.children = {}  # 子节点
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数
        self.untried_moves = []  # 未尝试的移动
        self.player = 0  # 当前玩家，1或2
    
    def select_child(self, c_param=1.4):
        """根据UCB公式选择子节点"""
        s = sorted(
            self.children.items(),
            key=lambda x: x[1].wins / x[1].visits + c_param * math.sqrt(2 * math.log(self.visits) / x[1].visits)
        )
        return s[-1][1]  # 返回最佳子节点
    
    def add_child(self, move, player):
        """添加子节点"""
        child = MCTSNode(parent=self, move=move)
        child.player = player
        self.untried_moves.remove(move)
        self.children[move] = child
        return child
    
    def update(self, result):
        """更新节点统计信息"""
        self.visits += 1
        self.wins += result

class MCTSPlayer(BaseAIPlayer):
    """基于MCTS的AI玩家"""
    
    def __init__(self, level=AILevel.HARD):  # 将MEDIUM更改为HARD
        super().__init__(level)
        # 根据难度设置MCTS参数
        if level == AILevel.EASY:
            self.num_simulations = 500
            self.c_puct = 4.0
        elif level == AILevel.HARD:  # 这里也需要更改
            self.num_simulations = 1000
            self.c_puct = 5.0
        else:  # EXPERT
            self.num_simulations = 2000
            self.c_puct = 5.0
    
    def think(self, board, color):
        """
        使用MCTS算法思考下一步
        
        Args:
            board (list): 棋盘状态，二维数组
            color (StoneColor): 己方棋子颜色
            
        Returns:
            tuple: 下一步落子位置 (x, y)
        """
        # 将枚举转换为数字
        player = color.value if isinstance(color, StoneColor) else color
        board_size = len(board)
        
        # 创建根节点
        root = MCTSNode()
        root.player = player
        
        # 获取所有可用的移动
        root.untried_moves = [
            (x, y) for y in range(board_size) for x in range(board_size)
            if board[y][x] == 0
        ]
        
        # 如果没有可用的移动，返回None
        if not root.untried_moves:
            return None
        
        # 如果只有一个可用的移动，直接返回
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        # 执行MCTS模拟
        for _ in range(self.num_simulations):
            # 复制棋盘
            board_copy = [row[:] for row in board]
            node = root
            
            # 选择阶段
            while not node.untried_moves and node.children:
                node = node.select_child()
                x, y = node.move
                board_copy[y][x] = node.player
            
            # 扩展阶段
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                x, y = move
                next_player = 3 - node.player  # 切换玩家
                board_copy[y][x] = node.player
                node = node.add_child(move, next_player)
            
            # 模拟阶段
            current_player = node.player
            while True:
                # 检查是否有胜者
                winner = self._check_winner(board_copy)
                if winner:
                    break
                    
                # 检查是否平局
                if all(board_copy[y][x] != 0 for y in range(board_size) for x in range(board_size)):
                    winner = 0  # 平局
                    break
                
                # 随机选择下一步
                available_moves = [
                    (x, y) for y in range(board_size) for x in range(board_size)
                    if board_copy[y][x] == 0
                ]
                
                if not available_moves:
                    winner = 0  # 平局
                    break
                
                x, y = random.choice(available_moves)
                board_copy[y][x] = current_player
                current_player = 3 - current_player  # 切换玩家
            
            # 回溯阶段
            while node:
                result = 0
                if winner == player:
                    result = 1  # 胜利
                elif winner != 0:
                    result = -1  # 失败
                    
                node.update(result)
                node = node.parent
        
        # 选择访问次数最多的子节点
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        
        return best_move
    
    def _check_winner(self, board):
        """检查是否有胜者"""
        board_size = len(board)
        
        # 检查行
        for y in range(board_size):
            for x in range(board_size - 4):
                if board[y][x] != 0:
                    if all(board[y][x + i] == board[y][x] for i in range(5)):
                        return board[y][x]
        
        # 检查列
        for x in range(board_size):
            for y in range(board_size - 4):
                if board[y][x] != 0:
                    if all(board[y + i][x] == board[y][x] for i in range(5)):
                        return board[y][x]
        
        # 检查主对角线
        for y in range(board_size - 4):
            for x in range(board_size - 4):
                if board[y][x] != 0:
                    if all(board[y + i][x + i] == board[y][x] for i in range(5)):
                        return board[y][x]
        
        # 检查副对角线
        for y in range(board_size - 4):
            for x in range(4, board_size):
                if board[y][x] != 0:
                    if all(board[y + i][x - i] == board[y][x] for i in range(5)):
                        return board[y][x]
        
        return 0  # 没有胜者
