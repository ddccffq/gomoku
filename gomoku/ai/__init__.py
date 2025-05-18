# 将AI目录标记为Python包
"""
五子棋AI模块，包含不同类型的AI玩家
"""

# 导入常用组件
from ai.base_ai import AILevel, StoneColor
from ai.ai_factory import AIFactory

__all__ = ['AILevel', 'StoneColor', 'AIFactory']
