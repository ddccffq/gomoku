# coding:utf-8
import os
import random
from .base_ai import BaseAIPlayer, AILevel, StoneColor

# 导入不同的AI实现类
from .mcts_ai import MCTSPlayer
from .random_ai import RandomAIPlayer
from .rule_based_ai import RuleBasedAIPlayer
try:
    from .mcts_nn import NeuralMCTSPlayer  # 新增导入
except ImportError:
    NeuralMCTSPlayer = None  # 如果导入失败，设置为None

def load_model(model_path, model_type="default"):
    """加载模型并输出模型信息
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型标识
        
    Returns:
        加载的模型对象
    """
    try:
        from ai.models import create_gomoku_model
        import torch
        import os
        
        # 输出模型基本信息
        print(f"\n===== 加载模型: {model_type} =====")
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 错误: 模型文件不存在: {model_path}")
            return None
        
        # 获取模型文件大小
        file_size = os.path.getsize(model_path)
        size_mb = file_size / (1024 * 1024)
        print(f"文件大小: {size_mb:.2f} MB")
        
        # 检测设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 尝试先加载状态字典以分析结构
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # 如果是状态字典而不是模型对象
            if isinstance(state_dict, dict):
                # 计算参数总数
                total_params = sum(p.numel() for p in state_dict.values())
                print(f"模型总参数: {total_params:,}")
                
                # 尝试从键名发现滤波器数量
                for key, value in state_dict.items():
                    if 'conv' in key.lower() and '.weight' in key:
                        if len(value.shape) == 4:  # 卷积层权重通常是4维的
                            filters = value.shape[0]
                            print(f"检测到滤波器数量: {filters}")
                            break
        except Exception as e:
            print(f"分析状态字典失败: {str(e)}")
        
        # 创建模型结构
        model = create_gomoku_model(board_size=15, device=device)
        
        # 分析模型结构
        num_filters = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and not num_filters:
                num_filters = module.out_channels
                print(f"检测到滤波器数量: {num_filters}")
                break
        
        # 估计模型大小
        if num_filters <= 32:
            model_size_category = "tiny"
        elif num_filters <= 64:
            model_size_category = "small"
        elif num_filters <= 128:
            model_size_category = "medium"
        else:
            model_size_category = "large"
        
        print(f"检测到模型大小: {model_size_category}")
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ 加载模型时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

class AIFactory:
    """AI工厂类，用于创建不同类型的AI"""
    
    @staticmethod
    def create_ai(ai_type, level=AILevel.HARD, model_path=None):  # 将MEDIUM改为HARD
        """
        创建指定类型的AI
        
        Args:
            ai_type (str): AI类型，'random', 'rule', 'mcts', 'neural'
            level (AILevel): AI难度级别
            model_path (str): 神经网络模型路径，仅对'neural'类型有效
            
        Returns:
            BaseAIPlayer: AI实例
        """
        if ai_type == 'random':
            return RandomAIPlayer(level)
        elif ai_type == 'rule':
            return RuleBasedAIPlayer(level)
        elif ai_type == 'mcts':
            return MCTSPlayer(level)
        elif ai_type == 'neural':
            if NeuralMCTSPlayer is not None:
                return NeuralMCTSPlayer(level, model_path)
            else:
                print("神经网络AI不可用，使用MCTS AI代替")
                return MCTSPlayer(level)
        else:
            # 默认返回规则型AI
            return RuleBasedAIPlayer(level)
    
    @staticmethod
    def create_ai_by_level(level):
        """根据难度级别创建不同类型的AI"""
        if level == AILevel.EASY:
            return RandomAIPlayer(level)
        elif level == AILevel.HARD:  # 将MEDIUM改为HARD
            return RuleBasedAIPlayer(level)
        else:  # EXPERT
            if NeuralMCTSPlayer is not None:
                return NeuralMCTSPlayer(level)
            else:
                return MCTSPlayer(level)

    @classmethod
    def create_ai(cls, level=AILevel.HARD, color=StoneColor.BLACK):  # 将MEDIUM改为HARD
        """创建默认AI实例"""
        return RuleBasedAIPlayer(level)
