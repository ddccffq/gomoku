from enum import Enum
import random
from typing import Dict, Any, List, Tuple, Optional

from ai.base_ai import AILevel, StoneColor


class AIStyle(Enum):
    """AI风格枚举"""
    BALANCED = 0   # 平衡型
    AGGRESSIVE = 1 # 进攻型
    DEFENSIVE = 2  # 防守型


class DifficultyManager:
    """AI难度管理器
    
    管理不同难度级别AI的具体参数和行为特性
    """
    
    # 单例模式
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化难度管理器"""
        # 搜索深度配置 - 不同难度级别使用不同的搜索深度
        self.search_depths = {
            AILevel.EASY: {
                "initial_depth": 1,
                "max_depth": 2,
                "time_limit": 1.0
            },
            AILevel.HARD: {
                "initial_depth": 2,
                "max_depth": 4,
                "time_limit": 2.0
            },
            AILevel.EXPERT: {
                "initial_depth": 3,
                "max_depth": 6, 
                "time_limit": 5.0
            }
        }
        
        # 评估函数精度 - 简单AI使用简化评估，专家AI使用完整精确评估
        self.evaluation_precision = {
            AILevel.EASY: 0.5,    # 简化评估，降低50%精度
            AILevel.HARD: 0.8,    # 略微降低精度
            AILevel.EXPERT: 1.0   # 完整精度
        }
        
        # 错误机制 - 简单AI有概率做出次优选择
        self.mistake_probability = {
            AILevel.EASY: 0.3,    # 30%概率犯错
            AILevel.HARD: 0.1,    # 10%概率犯错
            AILevel.EXPERT: 0.0   # 不犯错
        }
        
        # 风格倾向 - 不同难度有不同风格倾向
        self.style_preference = {
            AILevel.EASY: {
                AIStyle.BALANCED: 0.7,
                AIStyle.AGGRESSIVE: 0.2,
                AIStyle.DEFENSIVE: 0.1
            },
            AILevel.HARD: {
                AIStyle.BALANCED: 0.4,
                AIStyle.AGGRESSIVE: 0.3,
                AIStyle.DEFENSIVE: 0.3
            },
            AILevel.EXPERT: {
                # 专家级AI根据局势动态调整，默认平衡
                AIStyle.BALANCED: 0.34,
                AIStyle.AGGRESSIVE: 0.33, 
                AIStyle.DEFENSIVE: 0.33
            }
        }
        
        # 分支因子 - 控制搜索广度，影响每个深度考虑的候选走法数量
        self.branching_factor = {
            AILevel.EASY: 6,     # 每层最多考虑6个走法
            AILevel.HARD: 12,    # 每层最多考虑12个走法
            AILevel.EXPERT: 18   # 每层最多考虑18个走法
        }
        
        # 开局库和定式库使用率 - 控制AI使用预设策略的频率
        self.pattern_usage = {
            AILevel.EASY: 0.3,   # 30%使用率
            AILevel.HARD: 0.7,   # 70%使用率
            AILevel.EXPERT: 0.95 # 95%使用率
        }
        
        # 威胁感知级别 - 决定AI对威胁的识别和响应敏感度
        self.threat_awareness = {
            AILevel.EASY: 0.4,   # 低敏感度
            AILevel.HARD: 0.8,   # 中敏感度
            AILevel.EXPERT: 1.0  # 高敏感度
        }
    
    def get_search_params(self, level: AILevel) -> Dict[str, Any]:
        """获取指定难度的搜索参数
        
        Args:
            level: AI难度级别
            
        Returns:
            搜索参数字典
        """
        return self.search_depths.get(level, self.search_depths[AILevel.HARD])
    
    def should_make_mistake(self, level: AILevel) -> bool:
        """判断是否应该做出次优选择
        
        Args:
            level: AI难度级别
            
        Returns:
            是否应该犯错
        """
        prob = self.mistake_probability.get(level, 0)
        return random.random() < prob
    
    def get_evaluation_precision(self, level: AILevel) -> float:
        """获取评估函数精度系数
        
        Args:
            level: AI难度级别
            
        Returns:
            评估函数精度系数(0.0-1.0)
        """
        return self.evaluation_precision.get(level, 1.0)
    
    def get_style(self, level: AILevel, board_state=None) -> AIStyle:
        """根据难度级别和当前局势确定AI风格
        
        Args:
            level: AI难度级别
            board_state: 当前棋盘状态，专家级AI会根据局势动态调整风格
            
        Returns:
            AI风格
        """
        # 获取当前难度的风格偏好
        preferences = self.style_preference.get(level, self.style_preference[AILevel.BALANCED])
        
        # 专家级根据局势动态调整风格
        if level == AILevel.EXPERT and board_state:
            # 分析局势判断最佳风格
            # 这里简单实现，实际可以基于局势做更复杂的分析
            piece_count = sum(1 for row in board_state for cell in row if cell != 0)
            
            if piece_count < 20:  # 开局阶段，偏向激进
                return AIStyle.AGGRESSIVE
            elif piece_count < 50:  # 中盘阶段，平衡发展
                return AIStyle.BALANCED
            else:  # 残局阶段，防守为主
                return AIStyle.DEFENSIVE
        
        # 根据概率随机选择风格
        styles = list(preferences.keys())
        weights = list(preferences.values())
        return random.choices(styles, weights=weights, k=1)[0]
    
    def get_candidate_move_count(self, level: AILevel) -> int:
        """获取候选走法数量
        
        Args:
            level: AI难度级别
            
        Returns:
            需要考虑的候选走法数量
        """
        return self.branching_factor.get(level, 10)
    
    def use_pattern_library(self, level: AILevel) -> bool:
        """判断是否使用模式库(开局库/定式库)
        
        Args:
            level: AI难度级别
            
        Returns:
            是否使用模式库
        """
        prob = self.pattern_usage.get(level, 0.5)
        return random.random() < prob
    
    def get_threat_awareness(self, level: AILevel) -> float:
        """获取威胁感知系数
        
        Args:
            level: AI难度级别
            
        Returns:
            威胁感知系数(0.0-1.0)
        """
        return self.threat_awareness.get(level, 0.7)
    
    def introduce_randomness(self, scores: List[Tuple[Any, float]], level: AILevel) -> List[Tuple[Any, float]]:
        """根据难度级别为评分引入随机性
        
        Args:
            scores: 走法评分列表，每个元素为(走法,评分)元组
            level: AI难度级别
            
        Returns:
            处理后的评分列表
        """
        if not scores:
            return scores
            
        # 获取随机因子
        random_factor = 0
        if level == AILevel.EASY:
            random_factor = 0.3  # 30%随机波动
        elif level == AILevel.HARD:
            random_factor = 0.1  # 10%随机波动
        
        # 没有随机因子则直接返回
        if random_factor <= 0:
            return scores
            
        # 添加随机波动
        result = []
        max_score = max(score for _, score in scores)
        for move, score in scores:
            # 分数越高，波动越小，确保好棋不会因为随机因素变成坏棋
            actual_factor = random_factor * (1 - score / max_score) if max_score > 0 else random_factor
            random_score = score * (1 + (random.random() * 2 - 1) * actual_factor)
            result.append((move, random_score))
            
        return result
