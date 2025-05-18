# coding:utf-8
import random
import numpy as np
from .base_ai import BaseAIPlayer, AILevel, StoneColor

class RuleBasedAIPlayer(BaseAIPlayer):
    """基于规则的AI玩家，使用简单的评分规则"""
    
    def __init__(self, level=AILevel.HARD):  # 将MEDIUM更改为HARD
        super().__init__(level)
        
        # 根据难度级别设置参数
        if level == AILevel.EASY:
            self.depth = 1
        elif level == AILevel.HARD:  # 这里也需要更改
            self.depth = 2
        else:  # EXPERT
            self.depth = 3
            
        # 初始化评分表
        self.score_table = self._init_score_table()
    
    def _init_score_table(self):
        """初始化棋型评分表
        
        Returns:
            dict: 不同棋型的评分表
        """
        # 参考PatternScore类的分数定义
        return {
            'five': 100000,          # 连五
            'open_four': 10000,      # 活四
            'four': 1000,            # 冲四
            'open_three': 500,       # 活三
            'three': 100,            # 眠三
            'open_two': 50,          # 活二
            'two': 10,               # 眠二
            
            # 复合棋型
            'four_four': 8000,       # 双冲四
            'four_open_three': 5000, # 冲四活三
            'open_three_open_three': 3000  # 双活三
        }
    
    def think(self, board, color):
        """
        基于简单规则评估棋盘并选择最佳位置
        
        Args:
            board (list): 棋盘状态，二维数组
            color (StoneColor): 己方棋子颜色
            
        Returns:
            tuple: 下一步落子位置 (x, y)
        """
        # 将枚举转换为数字
        player = color.value if isinstance(color, StoneColor) else color
        opponent = 3 - player  # 1->2, 2->1
        
        # 创建评分数组
        board_size = len(board)
        scores = np.zeros((board_size, board_size), dtype=np.float32)
        
        # 查找所有空位
        empty_cells = []
        for y in range(board_size):
            for x in range(board_size):
                if board[y][x] == 0:
                    empty_cells.append((x, y))
                    
                    # 评估该位置的分数
                    score = self._evaluate_position(board, x, y, player, opponent)
                    scores[y][x] = score
        
        # 如果没有空位，返回None
        if not empty_cells:
            return None
        
        # 根据AI难度调整选择策略
        if self.level == AILevel.EASY:
            # 简单难度：有70%几率随机选择，30%几率选择最佳位置
            if random.random() < 0.7:
                return random.choice(empty_cells)
            
        # 找到得分最高的位置
        best_score = np.max(scores)
        best_positions = []
        
        for y in range(board_size):
            for x in range(board_size):
                if board[y][x] == 0 and scores[y][x] == best_score:
                    best_positions.append((x, y))
        
        # 随机选择一个最佳位置
        return random.choice(best_positions)
    
    def _evaluate_position(self, board, x, y, player, opponent):
        """评估某一位置的分数"""
        board_size = len(board)
        score = 0
        
        # 定义八个方向
        directions = [
            (1, 0), (0, 1), (1, 1), (1, -1),
            (-1, 0), (0, -1), (-1, -1), (-1, 1)
        ]
        
        # 检查每个方向
        for dx, dy in directions:
            # 检查我方连子
            my_line = 0
            tx, ty = x, y
            for _ in range(4):  # 最多检查4步
                tx += dx
                ty += dy
                if 0 <= tx < board_size and 0 <= ty < board_size and board[ty][tx] == player:
                    my_line += 1
                else:
                    break
            
            # 检查对方连子
            opp_line = 0
            tx, ty = x, y
            for _ in range(4):
                tx += dx
                ty += dy
                if 0 <= tx < board_size and 0 <= ty < board_size and board[ty][tx] == opponent:
                    opp_line += 1
                else:
                    break
            
            # 评分策略
            if my_line >= 4:
                score += self.score_table['five']  # 我方可以五连
            elif my_line == 3:
                score += self.score_table['open_four']   # 我方可以四连
            elif my_line == 2:
                score += self.score_table['open_three']  # 我方可以三连
            elif my_line == 1:
                score += self.score_table['open_two']    # 我方可以两连
            
            if opp_line >= 4:
                score += self.score_table['five'] / 2  # 阻止对方五连
            elif opp_line == 3:
                score += self.score_table['open_four'] / 2  # 阻止对方四连
            elif opp_line == 2:
                score += self.score_table['open_three'] / 2  # 阻止对方三连
        
        return score
    
    def get_move(self, board_state):
        """与think方法兼容的接口，将board_state转换为(x,y)坐标返回
        
        Args:
            board_state: 棋盘状态，二维列表
            
        Returns:
            tuple: (row, col) 位置坐标
        """
        # 确定当前玩家颜色
        player_id = self.color_to_player()
        color = StoneColor.BLACK if player_id == 1 else StoneColor.WHITE
        
        # 调用think方法获取坐标
        move = self.think(board_state, color)
        
        # 如果think返回的是(x,y)格式，转换为(row,col)格式
        if move:
            x, y = move
            return (y, x)  # 交换坐标顺序为(row, col)
        
        return None
    
    def color_to_player(self):
        """将AI颜色转换为棋盘上的玩家编号"""
        return 1 if self.color == StoneColor.BLACK else 2
