# coding:utf-8
import threading
import traceback
import os
import torch
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from qfluentwidgets import InfoBar, InfoBarPosition

# AI相关导入
from ai.ai_factory import AIFactory
from ai.base_ai import AILevel, StoneColor


class AIController(QObject):
    """AI控制器类，处理AI相关的交互和逻辑"""
    
    # 定义信号
    aiMoveReady = pyqtSignal(tuple)  # AI走法准备好的信号，参数为(row, col)坐标
    aiThinkingChanged = pyqtSignal(bool)  # AI思考状态变化信号
    aiError = pyqtSignal(str)  # AI错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ai = None
        self.ai_level = AILevel.EASY  # 默认简单难度
        self.ai_color = StoneColor.WHITE  # 默认AI执白
        self.is_thinking = False
        self.board_data = None
        
    def set_difficulty(self, level):
        """设置AI难度"""
        if level in [AILevel.EASY, AILevel.HARD, AILevel.EXPERT]:
            self.ai_level = level
            # 如果AI已经创建，则重新创建以应用新的难度
            if self.ai is not None:
                self.create_ai()
            return True
        return False
    
    def set_color(self, color):
        """设置AI执棋颜色"""
        if color in [StoneColor.BLACK, StoneColor.WHITE]:
            self.ai_color = color
            # 如果AI已经创建，则重新创建以应用新的颜色
            if self.ai is not None:
                self.create_ai()
            return True
        return False
    
    def create_ai(self):
        """创建AI实例"""
        try:
            print(f"尝试创建AI: 颜色={self.ai_color}, 难度={self.ai_level}")
            self.ai = AIFactory.create_ai(self.ai_level, self.ai_color)
            
            if self.ai is None:
                print("警告: AIFactory.create_ai返回了None")
                self.aiError.emit("无法创建AI，请检查AI模块")
                return False
            
            # 获取并输出模型信息
            self._log_ai_model_info()
            
            print(f"创建AI成功: {self.ai.__class__.__name__}")
            return True
        except Exception as e:
            print(f"创建AI失败: {str(e)}")
            traceback.print_exc()
            self.aiError.emit(f"创建AI失败: {str(e)}")
            return False
    
    def _log_ai_model_info(self):
        """记录AI模型的详细信息"""
        try:
            if hasattr(self.ai, 'model') and self.ai.model is not None:
                # 获取模型总参数数量
                model = self.ai.model
                total_params = sum(p.numel() for p in model.parameters())
                print(f"AI模型总参数: {total_params:,}")
                
                # 检测滤波器数量
                filter_count = 0
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        out_channels = module.out_channels
                        if filter_count == 0:  # 第一层卷积
                            filter_count = out_channels
                            print(f"检测到滤波器数量: {filter_count}")
                
                # 估计模型大小
                if filter_count <= 32:
                    model_size = "tiny"
                elif filter_count <= 64:
                    model_size = "small"
                elif filter_count <= 128:
                    model_size = "medium"
                else:
                    model_size = "large"
                
                print(f"检测到模型大小: {model_size}")
                
                # 如果有模型路径信息，则输出
                if hasattr(self.ai, 'model_path') and self.ai.model_path:
                    model_path = self.ai.model_path
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path)
                        size_mb = file_size / (1024 * 1024)
                        print(f"模型文件大小: {size_mb:.2f} MB")
                        print(f"模型文件路径: {model_path}")
        except Exception as e:
            print(f"获取AI模型信息时出错: {str(e)}")
    
    def calculate_move(self, board_data):
        """计算AI的下一步走法"""
        # 如果AI尚未创建，先尝试创建
        if self.ai is None:
            if not self.create_ai():
                self.aiError.emit("AI未初始化，无法计算走法")
                return
        
        # 保存当前棋盘状态
        self.board_data = [row[:] for row in board_data]
        
        # 设置思考状态
        self.is_thinking = True
        self.aiThinkingChanged.emit(True)
        
        # 在线程中执行AI计算
        threading.Thread(target=self._think).start()
    
    def _think(self):
        """AI思考过程，在单独线程中执行"""
        try:
            print("请求AI计算走法...")
            move = self.ai.get_move(self.board_data)
            print(f"AI计算出的走法: {move}")
            
            # 计算完成，发出信号
            self.aiMoveReady.emit(move)
        except Exception as e:
            print(f"AI思考出错: {str(e)}")
            traceback.print_exc()
            self.aiError.emit(str(e))
        finally:
            # 无论成功失败，都重置思考状态
            self.is_thinking = False
            self.aiThinkingChanged.emit(False)
    
    def check_ai_modules(self):
        """检查AI模块是否可用"""
        try:
            # 尝试导入必要的AI模块
            import ai.board_evaluator
            import ai.move_generator
            import ai.easy_ai
            import ai.hard_ai
            import ai.expert_ai
            
            # 尝试创建不同类型的AI实例以验证可用性
            for level in [AILevel.EASY, AILevel.HARD, AILevel.EXPERT]:
                for color in [StoneColor.BLACK, StoneColor.WHITE]:
                    ai = AIFactory.create_ai(level, color)
                    if ai is None:
                        print(f"无法创建AI: level={level}, color={color}")
                        return False
            return True
        except Exception as e:
            error_msg = f"AI模块初始化失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.aiError.emit(error_msg)
            return False
