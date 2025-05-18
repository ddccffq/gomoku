import os
import sys
import json
import torch
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.model_training import GomokuTrainer
from ai.self_play import SelfPlay


class TrainingWorker(QThread):
    """训练工作线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int, dict)  # 进度百分比, 额外数据
    game_completed = pyqtSignal(int, dict)  # 游戏索引, 游戏数据
    training_epoch_completed = pyqtSignal(str, int, dict)  # 模型名称, epoch, 指标
    training_completed = pyqtSignal(dict)  # 训练结果统计
    log_message = pyqtSignal(str)  # 日志消息
    error_occurred = pyqtSignal(str)  # 错误消息
    board_updated = pyqtSignal(list, list, int)  # 棋盘数据, 历史记录, 当前玩家
    
    def __init__(self, config=None):
        super().__init__()
        
        # 默认配置
        self.default_config = {
            # 模型配置
            'num_filters': 32,
            'num_residual_blocks': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 5,
            'optimizer': 'Adam',
            
            # 自我对弈配置
            'num_games': 100,
            'save_interval': 10,
            
            # 路径配置
            'model1_path': '',
            'model2_path': '',
            'output_dir': './trained_models',
            'selfplay_dir': './self_play_games'
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 控制标志
        self.is_running = False
        self.is_paused = False
    
    def run(self):
        """执行训练过程"""
        self.is_running = True
        self.is_paused = False
        
        try:
            # 创建输出目录
            os.makedirs(self.config['output_dir'], exist_ok=True)
            os.makedirs(self.config['selfplay_dir'], exist_ok=True)
            
            # 创建图像保存目录
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'game_images')
            os.makedirs(save_dir, exist_ok=True)
            self.log_message.emit(f"棋盘图像将保存至: {save_dir}")
            
            # 创建训练器
            self.log_message.emit("初始化训练器...")
            trainer_config = {
                'num_filters': self.config['num_filters'],
                'num_residual_blocks': self.config['num_residual_blocks'],
                'learning_rate': self.config['learning_rate'],
                'batch_size': self.config['batch_size'],
                'num_epochs': self.config['num_epochs'],
                'optimizer': self.config['optimizer'],
                'model1_path': self.config['model1_path'],
                'model2_path': self.config['model2_path'],
                'output_dir': self.config['output_dir']
            }
            
            trainer = GomokuTrainer(trainer_config)
            self.log_message.emit("训练器初始化完成")
            
            # 开始自我对弈
            self.log_message.emit(f"开始自我对弈过程，计划进行 {self.config['num_games']} 局对弈...")
            
            selfplay_config = {
                'num_games': self.config['num_games'],
                'save_interval': self.config['save_interval'],
                'output_dir': self.config['selfplay_dir'],
                'image_dir': save_dir
            }
            
            selfplay = SelfPlay(trainer, selfplay_config)
            
            # 执行自我对弈时传入中断检查函数
            games = selfplay.start(callback=self._game_callback, check_interrupt=lambda: not self.is_running)
            
            # 如果过程被中断，结束训练
            if not self.is_running:
                self.log_message.emit("训练过程被用户中断")
                return
            
            # 开始训练模型
            self.log_message.emit("开始训练模型...")
            for epoch in range(self.config['num_epochs']):
                trainer.train_on_games(games, callback=self._epoch_callback)
                if (epoch + 1) % self.config['save_interval'] == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 添加模型大小信息
                    model_size = self._get_model_size_name()
                    model1_path, model2_path, _ = trainer.save_models(timestamp, model_size)
                    self.log_message.emit(f"模型1已保存: {model1_path}")
                    self.log_message.emit(f"模型2已保存: {model2_path}")
            
            # 如果过程被中断，结束训练
            if not self.is_running:
                self.log_message.emit("训练过程被用户中断")
                return
            
            # 保存最终模型
            self.log_message.emit("保存最终模型...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 添加模型大小信息
            model_size = self._get_model_size_name()
            model1_path, model2_path, history_path = trainer.save_models(timestamp, model_size, is_final=True)
            
            # 发送训练完成信号
            training_results = {
                'model1_path': model1_path,
                'model2_path': model2_path,
                'history_path': history_path,
                'num_games': len(games),
                'timestamp': timestamp
            }
            self.training_completed.emit(training_results)
            
            self.log_message.emit("训练过程已完成!")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_message = f"训练过程出错: {str(e)}\n{tb}"
            self.log_message.emit(error_message)
            self.error_occurred.emit(str(e))
            
        finally:
            self.is_running = False
    
    def _get_model_size_name(self):
        """根据模型配置参数确定模型大小名称"""
        num_filters = self.config.get('num_filters', 32)
        num_blocks = self.config.get('num_residual_blocks', 3)
        
        if num_filters <= 32 and num_blocks <= 3:
            return "tiny"
        elif num_filters <= 64 and num_blocks <= 5:
            return "small"
        elif num_filters <= 128 and num_blocks <= 10:
            return "medium"
        else:
            return "large"
    
    def _game_callback(self, game_idx, game_data):
        """对弈回调
        
        Args:
            game_idx: 游戏索引（从1开始）
            game_data: 游戏数据
        """
        # 处理暂停
        while self.is_paused and self.is_running:
            self.msleep(100)
        
        # 如果不再运行，直接返回
        if not self.is_running:
            return False  # 返回False表示中断
        
        # 计算进度
        progress = min(100, int(100 * game_idx / self.config['num_games']))
        
        # 统计数据
        stats = {
            'current_game': game_idx,
            'total_games': self.config['num_games'],
            'black_wins': sum(1 for g in game_data.get('move_history', []) if g[2] == 1),
            'white_wins': sum(1 for g in game_data.get('move_history', []) if g[2] == 2),
            'draws': 1 if game_data.get('winner', 0) == 0 else 0
        }
        
        # 发射进度与统计信息，供绘图使用
        self.progress_updated.emit(int(100 * game_idx / self.config['num_games']), stats)
        
        # 发送进度信号
        self.progress_updated.emit(progress, stats)
        
        # 发送游戏完成信号
        self.game_completed.emit(game_idx, game_data)
        
        # 记录日志
        winner_str = '黑棋' if game_data.get('winner') == 1 else '白棋' if game_data.get('winner') == 2 else '和棋'
        self.log_message.emit(f"对弈 {game_idx}/{self.config['num_games']} 完成，胜者: {winner_str}，棋盘图像已保存")
        
        return True  # 返回True表示继续
    
    def update_board_state(self, board, moves, current_player):
        """更新棋盘状态，发送信号到界面"""
        import numpy as np
        
        # 将NumPy数组转换为Python列表
        if isinstance(board, np.ndarray):
            board = board.tolist()
        
        # 确保moves也是Python列表
        if isinstance(moves, np.ndarray):
            moves = moves.tolist()
        
        # 创建唯一的状态标识
        current_state = (len(moves), moves[-1] if moves else None)
        
        # 检查是否与上一步相同，避免重复处理
        if hasattr(self, '_last_board_state') and self._last_board_state == current_state:
            return
        
        # 保存当前状态
        self._last_board_state = current_state
        
        # 发送棋盘状态更新信号
        self.board_updated.emit(board, moves, current_player)
        
        # 记录日志，确保格式正确
        if current_player > 0:  # 只在游戏进行中记录
            player_name = "黑棋" if current_player == 1 else "白棋"
            move_info = ""
            
            if moves and len(moves) > 0:
                last_move = moves[-1]
                if isinstance(last_move, (list, tuple)) and len(last_move) >= 2:
                    row, col = last_move[0], last_move[1]
                    move_info = f"落子坐标：({row}, {col})"
            
            self.log_message.emit(f"棋盘状态已更新：回合 {len(moves)}。{player_name}{move_info}。")
    
    def _epoch_callback(self, model_name, epoch, metrics):
        """训练轮次回调
        
        Args:
            model_name: 模型名称
            epoch: 当前epoch
            metrics: 训练指标
        """
        # 处理暂停
        while self.is_paused and self.is_running:
            self.msleep(100)
        
        # 如果不再运行，直接返回
        if not self.is_running:
            return
        
        # 发送epoch完成信号
        self.training_epoch_completed.emit(model_name, epoch, metrics)
        
        # 记录日志
        self.log_message.emit(
            f"{model_name} Epoch {epoch}/{self.config['num_epochs']} - "
            f"训练损失: {metrics['train_loss']:.4f}, 验证损失: {metrics['val_loss']:.4f}"
        )
    
    def stop(self):
        """停止训练"""
        self.is_running = False
    
    def pause(self):
        """暂停训练"""
        self.is_paused = True
        self.log_message.emit("训练已暂停")
    
    def resume(self):
        """恢复训练"""
        self.is_paused = False
        self.log_message.emit("训练已恢复")

class TrainingController(QObject):
    """训练控制器，负责协调训练界面和训练线程"""
    
    # 转发训练工作者的信号
    progress_updated = pyqtSignal(int, dict)
    game_completed = pyqtSignal(int, dict)
    training_epoch_completed = pyqtSignal(str, int, dict)
    training_completed = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    board_updated = pyqtSignal(list, list, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
    
    def start_training(self, config):
        """启动训练过程
        
        Args:
            config: 训练配置
        """
        # 如果已有工作线程，先停止
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # 创建新的工作线程
        self.worker = TrainingWorker(config)
        
        # 连接信号
        self.worker.progress_updated.connect(self.progress_updated)
        self.worker.game_completed.connect(self.game_completed)
        self.worker.training_epoch_completed.connect(self.training_epoch_completed)
        self.worker.training_completed.connect(self.training_completed)
        self.worker.log_message.connect(self.log_message)
        self.worker.error_occurred.connect(self.error_occurred)
        
        # 确保连接board_updated信号(如果存在)
        if hasattr(self.worker, 'board_updated'):
            self.worker.board_updated.connect(self.board_updated)
        
        # 启动线程
        self.worker.start()
    
    def stop_training(self):
        """停止训练"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
    
    def toggle_pause(self):
        """切换暂停/恢复状态"""
        if not self.worker or not self.worker.isRunning():
            return False
        
        if self.worker.is_paused:
            self.worker.resume()
            return False  # 不再暂停
        else:
            self.worker.pause()
            return True  # 现在已暂停
    
    def is_training(self):
        """检查是否正在训练"""
        return self.worker and self.worker.isRunning()
    
    def is_paused(self):
        """检查训练是否已暂停"""
        return self.worker and self.worker.is_paused
