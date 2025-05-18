# coding:utf-8

import random
from .base_ai import BaseAI, StoneColor, AILevel

class EasyAI(BaseAI):
    """
    简单AI实现 - 使用随机策略
    """
    
    def __init__(self, color, level=AILevel.EASY):
        super().__init__(color, level)
    
    def get_move(self, board_state):
        """
        随机选择一个空位落子
        
        Args:
            board_state: 棋盘状态，二维列表
            
        Returns:
            位置元组(row, col)
        """
        empty_positions = []
        
        # 收集所有空位
        for row in range(len(board_state)):
            for col in range(len(board_state[row])):
                if board_state[row][col] == 0:  # 空位
                    empty_positions.append((row, col))
        
        # 随机选择一个空位
        if empty_positions:
            return random.choice(empty_positions)
        
        # 棋盘已满，返回None表示无法下棋
        return None
