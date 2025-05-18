import tkinter as tk
from tkinter import ttk
import os

class TrainingWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("������ѵ��")
        self.geometry("800x600")
        
        # ���һ����ǩ������ʾ����ͳ��
        self.data_stats_label = tk.Label(self, text="ѵ������ͳ��: ������...")
        self.data_stats_label.pack(pady=5)
        
        self.training_log = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED)
        self.training_log.pack(expand=True, fill=tk.BOTH)
        
        self.game_state = None  # ��Ϸ״̬����
        
        # ��������ͳ��
        self.load_data_stats()

    def append_to_training_log(self, message):
        """��ѵ����־��׷����Ϣ"""
        self.training_log.config(state=tk.NORMAL)
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.config(state=tk.DISABLED)

    def log_move(self, player_name, row, col):
        """��¼������Ϣ��ѵ����־"""
        log_message = f"����: {player_name} ������ ({row}, {col}) ����"
        self.append_to_training_log(log_message)
    
    def update_game_state(self, game_state):
        """������Ϸ״̬����¼��־"""
        self.game_state = game_state
        
        # ��ȡ���һ������
        if hasattr(game_state, 'last_move') and game_state.last_move:
            row, col = game_state.last_move
            player_name = "����" if game_state.current_player == 2 else "����"
            self.log_move(player_name, row, col)

    def load_data_stats(self):
        """����ѵ������ͳ����Ϣ"""
        try:
            # ��ȡѵ������Ŀ¼
            training_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'trained_models', 'training_data'
            )
            
            if not os.path.exists(training_data_dir):
                self.data_stats_label.config(text="ѵ������ͳ��: δ�ҵ�����Ŀ¼")
                return
                
            # ͳ���û����׺����Ҷ��ĵ�����
            user_sessions = 0
            selfplay_sessions = 0
            total_games = 0
            
            for session_name in os.listdir(training_data_dir):
                session_path = os.path.join(training_data_dir, session_name)
                if not os.path.isdir(session_path):
                    continue
                    
                # ͳ����Ϸ����
                games = len([name for name in os.listdir(session_path) 
                             if os.path.isdir(os.path.join(session_path, name))])
                
                if 'user_session' in session_name:
                    user_sessions += 1
                    total_games += games
                else:
                    selfplay_sessions += 1
                    total_games += games
            
            # ���±�ǩ
            stats_text = f"ѵ������ͳ��: �� {total_games} ����Ϸ "
            stats_text += f"(���Ҷ���: {selfplay_sessions} ��, �û�����: {user_sessions} ��)"
            
            self.data_stats_label.config(text=stats_text)
            
            self.append_to_training_log(f"����ͳ��: {stats_text}")
            
        except Exception as e:
            print(f"����ѵ������ͳ��ʧ��: {str(e)}")
            self.data_stats_label.config(text=f"ѵ������ͳ��: ����ʧ�� ({str(e)})")