# coding:utf-8

from typing import List, Tuple, Dict, Optional
import random
from copy import deepcopy
from ai.base_ai import BaseAI, StoneColor, AILevel
from ai.hard_ai import HardAI


class ExpertAI(HardAI):
    """
    专家AI实现 - 使用MinMax搜索算法
    """
    
    def __init__(self, color, level=AILevel.EXPERT):
        super().__init__(color, level)
        self.max_depth = 3  # 搜索深度
    
    def get_move(self, board_state):
        """
        使用MinMax搜索选择最优走法
        
        Args:
            board_state: 棋盘状态，二维列表
            
        Returns:
            位置元组(row, col)
        """
        player = self.color_to_player()
        opponent = 3 - player
        
        # 首先尝试使用HardAI的逻辑寻找必胜和必防位置
        hard_move = super().get_move(board_state)
        
        # 如果找到必胜位置，直接返回
        board_copy = deepcopy(board_state)
        if hard_move:
            x, y = hard_move
            board_copy[x][y] = player
            if self._check_win(board_copy, x, y, player):
                return hard_move
                
        # 使用MinMax搜索
        best_value = -float('inf')
        best_move = None
        alpha = -float('inf')  # Alpha-Beta剪枝的alpha值
        beta = float('inf')    # Alpha-Beta剪枝的beta值
        
        # 获取合理的候选位置
        candidates = self._get_candidate_positions(board_state)
        
        # 遍历所有候选位置
        for move in candidates:
            row, col = move
            # 模拟落子
            board_copy = deepcopy(board_state)
            board_copy[row][col] = player
            
            # 评估这一步的得分
            value = self._minimax(board_copy, self.max_depth - 1, False, alpha, beta, player, opponent)
            
            # 更新最佳走法
            if value > best_value:
                best_value = value
                best_move = move
            
            # 更新Alpha值
            alpha = max(alpha, best_value)
        
        # 如果找到有效走法则返回，否则使用HardAI的策略
        return best_move if best_move else hard_move
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta, player, opponent):
        """
        MinMax搜索算法实现，带Alpha-Beta剪枝
        
        Args:
            board: 棋盘状态
            depth: 剩余搜索深度
            is_maximizing: 是否是最大化节点(true为玩家回合，false为对手回合)
            alpha, beta: Alpha-Beta剪枝的边界值
            player: 己方玩家编号
            opponent: 对手编号
            
        Returns:
            当前局面的评估分数
        """
        # 检查终止条件：达到叶子节点或游戏结束
        if depth == 0:
            return self._evaluate_board(board, player, opponent)
        
        # 获取候选位置
        candidates = self._get_candidate_positions(board)
        
        if is_maximizing:  # 最大化节点(玩家回合)
            value = -float('inf')
            for move in candidates:
                row, col = move
                # 检查位置是否为空
                if board[row][col] == 0:
                    # 模拟落子
                    board[row][col] = player
                    
                    # 检查是否获胜
                    if self._check_win(board, row, col, player):
                        board[row][col] = 0  # 恢复棋盘
                        return 10000 + depth  # 加上深度奖励尽快获胜
                    
                    # 递归评估子节点
                    child_value = self._minimax(board, depth - 1, False, alpha, beta, player, opponent)
                    board[row][col] = 0  # 恢复棋盘
                    
                    value = max(value, child_value)
                    alpha = max(alpha, value)
                    
                    # Alpha-Beta剪枝
                    if alpha >= beta:
                        break
                        
            return value
        else:  # 最小化节点(对手回合)
            value = float('inf')
            for move in candidates:
                row, col = move
                # 检查位置是否为空
                if board[row][col] == 0:
                    # 模拟落子
                    board[row][col] = opponent
                    
                    # 检查是否获胜
                    if self._check_win(board, row, col, opponent):
                        board[row][col] = 0  # 恢复棋盘
                        return -10000 - depth  # 加上深度惩罚尽快防守
                    
                    # 递归评估子节点
                    child_value = self._minimax(board, depth - 1, True, alpha, beta, player, opponent)
                    board[row][col] = 0  # 恢复棋盘
                    
                    value = min(value, child_value)
                    beta = min(beta, value)
                    
                    # Alpha-Beta剪枝
                    if alpha >= beta:
                        break
                        
            return value
    
    def _evaluate_board(self, board, player, opponent):
        """
        评估整个棋盘的分数，考虑双方的形势
        
        Args:
            board: 棋盘状态
            player: 己方玩家编号
            opponent: 对手编号
            
        Returns:
            棋盘评估分数
        """
        player_score = 0
        opponent_score = 0
        board_size = len(board)
        
        # 统计所有空位的得分
        for row in range(board_size):
            for col in range(board_size):
                if board[row][col] == 0:
                    # 模拟己方落子，评估得分
                    board[row][col] = player
                    player_score += self._evaluate_move(board, row, col, player, opponent)
                    
                    # 模拟对手落子，评估得分
                    board[row][col] = opponent
                    opponent_score += self._evaluate_move(board, row, col, opponent, player)
                    
                    # 恢复棋盘
                    board[row][col] = 0
        
        # 最终评分为己方得分减去对手得分
        return player_score - opponent_score * 0.8  # 稍微降低对防守的重视
    
    def _get_candidate_positions(self, board):
        """
        获取有价值的候选落子位置，减少搜索空间
        
        Args:
            board: 棋盘状态
            
        Returns:
            候选位置列表
        """
        board_size = len(board)
        candidates = []
        
        # 在已有棋子周围2格范围内寻找候选位置
        for row in range(board_size):
            for col in range(board_size):
                if board[row][col] != 0:  # 有棋子的位置
                    # 检查周围2格内的空位
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            r, c = row + dr, col + dc
                            # 检查位置是否在棋盘内且为空
                            if (0 <= r < board_size and 0 <= c < board_size and
                                board[r][c] == 0 and (r, c) not in candidates):
                                candidates.append((r, c))
        
        # 如果没有找到候选位置(例如空棋盘)，则返回棋盘中心位置
        if not candidates:
            center = board_size // 2
            candidates.append((center, center))
        
        return candidates
    
    def _check_win(self, board, row, col, player):
        """
        检查指定位置是否获胜(五子连珠)
        
        Args:
            board: 棋盘状态
            row, col: 位置坐标
            player: 玩家编号
            
        Returns:
            是否获胜
        """
        # 方向: 水平、垂直、主对角线、副对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 当前位置算1个
            
            # 向两个方向查找连子
            for sign in [-1, 1]:
                for step in range(1, 5):  # 最多看4步
                    x, y = row + sign * dx * step, col + sign * dy * step
                    
                    # 检查边界和玩家
                    if (0 <= x < len(board) and 0 <= y < len(board) and 
                        board[x][y] == player):
                        count += 1
                    else:
                        break
            
            # 检查是否达到五连
            if count >= 5:
                return True
        
        return False
