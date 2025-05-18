# coding:utf-8
from enum import Enum, auto
from typing import List, Tuple, Optional

class StoneColor(Enum):
    """棋子颜色枚举"""
    BLACK = 1
    WHITE = 2

class AILevel(Enum):
    """AI难度级别枚举"""
    EASY = auto()
    HARD = auto()
    EXPERT = auto()
    # MEDIUM 不存在，需要确保所有代码使用上面三个级别

class BaseAIPlayer:
    """AI玩家基类"""
    
    def __init__(self, color=StoneColor.BLACK, level=AILevel.EXPERT):
        self.color = color
        self.level = level
    
    def think(self, board, color):
        """思考下一步棋的位置"""
        raise NotImplementedError("子类必须实现think方法")

# 为了向后兼容，添加BaseAI别名
BaseAI = BaseAIPlayer
