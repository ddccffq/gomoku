# coding:utf-8

import random
from .base_ai import BaseAI, StoneColor, AILevel

class HardAI(BaseAI):
    """
    困难AI实现 - 使用评分策略
    """
    
    def __init__(self, color, level=AILevel.HARD):
        super().__init__(color, level)
    
    def get_move(self, board_state):
        """
        使用评分策略选择最优走法
        
        Args:
            board_state: 棋盘状态，二维列表
            
        Returns:
            位置元组(row, col)
        """
        best_score = -float('inf')
        best_moves = []
        board_size = len(board_state)
        
        # 玩家编号 (1为黑棋，2为白棋)
        player = self.color_to_player()
        opponent = 3 - player  # 对手编号
        
        # 搜索所有空位，找出评分最高的走法
        for row in range(board_size):
            for col in range(board_size):
                if board_state[row][col] == 0:  # 空位
                    # 模拟落子
                    board_state[row][col] = player
                    
                    # 计算该位置的评分
                    score = self._evaluate_move(board_state, row, col, player, opponent)
                    
                    # 恢复棋盘
                    board_state[row][col] = 0
                    
                    # 更新最佳走法
                    if score > best_score:
                        best_score = score
                        best_moves = [(row, col)]
                    elif score == best_score:
                        best_moves.append((row, col))
        
        # 如果有多个评分相同的最佳走法，随机选择一个
        if best_moves:
            return random.choice(best_moves)
        
        # 棋盘已满或出错，返回None
        return None
    
    def _evaluate_move(self, board, row, col, player, opponent):
        """
        评估一个位置的分数，考虑进攻和防守
        
        Args:
            board: 棋盘状态
            row, col: 位置坐标
            player: 玩家编号
            opponent: 对手编号
            
        Returns:
            位置评分
        """
        # 方向: 水平、垂直、主对角线、副对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        # 自己的连子评分
        attack_score = self._evaluate_direction(board, row, col, directions, player)
        
        # 阻止对手连子的评分
        defense_score = self._evaluate_direction(board, row, col, directions, opponent)
        
        # 总分为进攻分数和防守分数的加权和
        return attack_score * 1.1 + defense_score
    
    def _evaluate_direction(self, board, row, col, directions, player_id):
        """
        评估一个位置在所有方向上的分数
        
        Args:
            board: 棋盘状态
            row, col: 位置坐标
            directions: 方向列表，如 [(1,0), (0,1), (1,1), (1,-1)]
            player_id: 玩家编号
            
        Returns:
            所有方向上的综合评分
        """
        board_size = len(board)
        total_score = 0
        
        for dx, dy in directions:
            # 左右或上下方向的连子数和空位数
            consecutive = 1  # 当前位置算1个
            space_before = 0
            space_after = 0
            
            # 向一个方向查找连子
            for sign in [-1, 1]:
                for step in range(1, 5):  # 最多看4步
                    x, y = row + sign * dx * step, col + sign * dy * step
                    
                    # 检查边界
                    if not (0 <= x < board_size and 0 <= y < board_size):
                        break
                    
                    # 计算连子和空位
                    if board[x][y] == player_id:
                        if step == 1 and sign == -1:
                            consecutive += 1
                        elif space_before == 0 and sign == -1:
                            consecutive += 1
                        elif space_after == 0 and sign == 1:
                            consecutive += 1
                        else:
                            break
                    elif board[x][y] == 0:
                        if sign == -1:
                            space_before += 1
                        else:
                            space_after += 1
                    else:
                        break
                    
                    # 如果超过一个空位就停止
                    if (sign == -1 and space_before > 1) or (sign == 1 and space_after > 1):
                        break
            
            # 计算这个方向的分数
            direction_score = 0
            
            # 基于连子数和空位数评分
            if consecutive >= 5:
                direction_score = 100000  # 五连胜利
            elif consecutive == 4:
                if space_before > 0 and space_after > 0:
                    direction_score = 10000  # 活四
                elif space_before > 0 or space_after > 0:
                    direction_score = 1000   # 冲四
            elif consecutive == 3:
                if space_before > 0 and space_after > 0:
                    direction_score = 500    # 活三
                elif space_before > 0 or space_after > 0:
                    direction_score = 100    # 眠三
            elif consecutive == 2:
                if space_before > 0 and space_after > 0:
                    direction_score = 50     # 活二
                elif space_before > 0 or space_after > 0:
                    direction_score = 10     # 眠二
            else:
                direction_score = 1          # 单子
            
            total_score += direction_score
        
        return total_score
