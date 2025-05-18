import os
import time
import json
import numpy as np
import random
from datetime import datetime
from ai.model_training import GomokuTrainer

class SelfPlay:
    """五子棋自我对弈"""
    
    def __init__(self, trainer, config=None):
        """初始化自我对弈
        
        Args:
            trainer: GomokuTrainer实例
            config: 自我对弈配置
        """
        # 默认配置
        self.default_config = {
            'board_size': 15,
            'num_games': 100,
            'exploration_temp': 1.0,
            'save_interval': 10,
            'output_dir': './self_play_games'
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 存储GomokuTrainer
        self.trainer = trainer
        
        # 确保输出目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 游戏记录
        self.games = []
    
    def start(self, callback=None):
        """开始自我对弈
        
        Args:
            callback: 游戏完成时的回调函数
        """
        print(f"开始自我对弈，计划对局数: {self.config['num_games']}")
        
        for game_idx in range(self.config['num_games']):
            # 初始化新游戏
            board = np.zeros((self.config['board_size'], self.config['board_size']), dtype=np.int8)
            current_player = 1  # 黑棋先行
            move_history = []
            game_over = False
            winner = 0
            
            # 进行对弈
            while not game_over:
                # 当前玩家模型
                model = 1 if current_player == 1 else 2
                
                # 获取当前玩家的走法
                row, col = self.trainer.get_move(board, model, current_player)
                
                # 如果没有有效走法，游戏结束
                if row is None or col is None:
                    game_over = True
                    winner = 0  # 和棋
                    break
                
                # 记录走法
                move_history.append((row, col, current_player))
                
                # 执行走法
                board[row][col] = current_player
                
                # 检查是否获胜
                if self._check_win(board, row, col, current_player):
                    game_over = True
                    winner = current_player
                    break
                
                # 检查是否和棋（棋盘已满）
                if len(move_history) >= self.config['board_size'] ** 2:
                    game_over = True
                    winner = 0  # 和棋
                    break
                
                # 切换玩家
                current_player = 3 - current_player  # 1->2, 2->1
            
            # 游戏结束，记录结果
            game_record = {
                'board_size': self.config['board_size'],
                'move_history': move_history,
                'winner': winner,
                'timestamp': datetime.now().isoformat()
            }
            
            self.games.append(game_record)
            
            # 保存记录
            if (game_idx + 1) % self.config['save_interval'] == 0 or game_idx == self.config['num_games'] - 1:
                self._save_games(f"games_{game_idx+1}.json")
            
            # 回调
            if callback:
                callback(game_idx + 1, game_record)
                
            print(f"完成对局 {game_idx+1}/{self.config['num_games']}, 胜者: {'黑棋' if winner == 1 else '白棋' if winner == 2 else '和棋'}")
        
        print(f"自我对弈完成，总共 {len(self.games)} 局")
        return self.games
    
    def _check_win(self, board, row, col, player):
        """检查指定位置是否形成胜利
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 玩家ID
            
        Returns:
            True如果该玩家获胜，否则False
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        board_size = self.config['board_size']
        
        for dx, dy in directions:
            count = 1  # 当前位置算一个
            
            # 向两个方向检查
            for direction in [1, -1]:
                for step in range(1, 5):
                    r, c = row + direction * step * dx, col + direction * step * dy
                    if 0 <= r < board_size and 0 <= c < board_size and board[r][c] == player:
                        count += 1
                    else:
                        break
            
            # 如果达到5子连珠则获胜
            if count >= 5:
                return True
        
        return False
    
    def _save_games(self, filename):
        """保存游戏记录
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.config['output_dir'], filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'games': self.games,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"游戏记录已保存: {filepath}")


# 测试代码
if __name__ == "__main__":
    # 创建训练器
    from ai.model_training import GomokuTrainer
    
    trainer_config = {
        'num_filters': 32,
        'num_residual_blocks': 3,
        'output_dir': './test_models'
    }
    
    trainer = GomokuTrainer(trainer_config)
    
    # 创建自我对弈
    selfplay_config = {
        'num_games': 5,
        'save_interval': 2,
        'output_dir': './test_selfplay'
    }
    
    selfplay = SelfPlay(trainer, selfplay_config)
    
    # 执行自我对弈
    games = selfplay.start()
    
    # 训练模型
    trainer.train_on_games(games)
