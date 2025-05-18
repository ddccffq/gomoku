# coding:utf-8
import random
import numpy as np
from .base_ai import BaseAIPlayer, AILevel, StoneColor

class RandomAIPlayer(BaseAIPlayer):
    """随机AI玩家，随机选择一个合法的落子位置"""
    
    def __init__(self, level=AILevel.EASY):
        super().__init__(level)
    
    def think(self, board, color):
        """
        随机选择一个空位
        
        Args:
            board (list): 棋盘状态，二维数组
            color (StoneColor): 己方棋子颜色
            
        Returns:
            tuple: 下一步落子位置 (x, y)
        """
        # 查找所有空位
        empty_cells = []
        for y in range(len(board)):
            for x in range(len(board[y])):
                if board[y][x] == 0:
                    empty_cells.append((x, y))
        
        # 随机选择一个空位
        if empty_cells:
            return random.choice(empty_cells)
        
        # 如果棋盘已满，返回None
        return None
