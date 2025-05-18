from typing import Optional, Dict, Type
import random

from ai.base_ai import BaseAI, AILevel, StoneColor
from ai.easy_ai import EasyAI
from ai.hard_ai import HardAI
from ai.expert_ai import ExpertAI
from ai.adaptive_ai import AdaptiveAI
from ai.difficulty_manager import AIStyle


class EnhancedAIFactory:
    """增强版AI工厂
    
    支持创建各种类型和风格的AI
    """
    
    # 各难度的AI类映射
    AI_CLASSES = {
        AILevel.EASY: EasyAI,
        AILevel.HARD: HardAI,
        AILevel.EXPERT: ExpertAI,
        # 特殊类型
        "ADAPTIVE": AdaptiveAI
    }
    
    @classmethod
    def create_ai(cls, color: StoneColor, style: AIStyle = None) -> Optional[BaseAI]:
        """创建AI实例
        
        Args:
            color: AI执棋颜色
            style: AI风格(可选)
            
        Returns:
            创建的AI实例，如果创建失败则返回None
        """
        try:
            # 固定使用一个默认难度
            ai_class = cls.AI_CLASSES[AILevel.HARD]  # 默认使用HARD难度
            
            # 创建AI实例
            ai_instance = ai_class(color)
            
            # 如果指定了风格，设置AI风格
            if style and hasattr(ai_instance, 'style'):
                ai_instance.style = style
                
            return ai_instance
        except Exception as e:
            print(f"创建AI失败: {str(e)}")
            return None
    
    @classmethod
    def create_adaptive_ai(cls, color: StoneColor) -> Optional[AdaptiveAI]:
        """创建适应性AI
        
        Args:
            color: AI执棋颜色
            
        Returns:
            适应性AI实例
        """
        try:
            return AdaptiveAI(color)
        except Exception as e:
            print(f"创建适应性AI失败: {e}")
            return None
    
    @classmethod
    def create_styled_ai(cls, level: AILevel, color: StoneColor, style: AIStyle) -> Optional[BaseAI]:
        """创建指定风格的AI
        
        Args:
            level: AI难度级别
            color: AI执棋颜色
            style: 指定AI风格
            
        Returns:
            具有特定风格的AI实例
        """
        return cls.create_ai(color, style)
    
    @classmethod
    def create_random_styled_ai(cls, level: AILevel, color: StoneColor) -> Optional[BaseAI]:
        """创建随机风格的AI
        
        Args:
            level: AI难度级别
            color: AI执棋颜色
            
        Returns:
            随机风格的AI实例
        """
        style = random.choice(list(AIStyle))
        return cls.create_ai(color, style)
