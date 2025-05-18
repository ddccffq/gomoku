import tkinter as tk
from tkinter import ttk
import os

class TrainingWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("五子棋训练")
        self.geometry("800x600")
        
        # 添加一个标签用于显示数据统计
        self.data_stats_label = tk.Label(self, text="训练数据统计: 加载中...")
        self.data_stats_label.pack(pady=5)
        
        self.training_log = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED)
        self.training_log.pack(expand=True, fill=tk.BOTH)
        
        self.game_state = None  # 游戏状态对象
        
        # 加载数据统计
        self.load_data_stats()

    def append_to_training_log(self, message):
        """向训练日志中追加信息"""
        self.training_log.config(state=tk.NORMAL)
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.config(state=tk.DISABLED)

    def log_move(self, player_name, row, col):
        """记录落子信息到训练日志"""
        log_message = f"落子: {player_name} 在坐标 ({row}, {col}) 落子"
        self.append_to_training_log(log_message)
    
    def update_game_state(self, game_state):
        """更新游戏状态并记录日志"""
        self.game_state = game_state
        
        # 获取最后一步落子
        if hasattr(game_state, 'last_move') and game_state.last_move:
            row, col = game_state.last_move
            player_name = "黑棋" if game_state.current_player == 2 else "白棋"
            self.log_move(player_name, row, col)

    def load_data_stats(self):
        """加载训练数据统计信息"""
        try:
            # 获取训练数据目录
            training_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'trained_models', 'training_data'
            )
            
            if not os.path.exists(training_data_dir):
                self.data_stats_label.config(text="训练数据统计: 未找到数据目录")
                return
                
            # 统计用户贡献和自我对弈的数据
            user_sessions = 0
            selfplay_sessions = 0
            total_games = 0
            
            for session_name in os.listdir(training_data_dir):
                session_path = os.path.join(training_data_dir, session_name)
                if not os.path.isdir(session_path):
                    continue
                    
                # 统计游戏数量
                games = len([name for name in os.listdir(session_path) 
                             if os.path.isdir(os.path.join(session_path, name))])
                
                if 'user_session' in session_name:
                    user_sessions += 1
                    total_games += games
                else:
                    selfplay_sessions += 1
                    total_games += games
            
            # 更新标签
            stats_text = f"训练数据统计: 共 {total_games} 局游戏 "
            stats_text += f"(自我对弈: {selfplay_sessions} 组, 用户贡献: {user_sessions} 组)"
            
            self.data_stats_label.config(text=stats_text)
            
            self.append_to_training_log(f"数据统计: {stats_text}")
            
        except Exception as e:
            print(f"加载训练数据统计失败: {str(e)}")
            self.data_stats_label.config(text=f"训练数据统计: 加载失败 ({str(e)})")