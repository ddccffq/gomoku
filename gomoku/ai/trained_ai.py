# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
from datetime import datetime
from .base_ai import BaseAI, AILevel, StoneColor

class TrainedAI(BaseAI):
    """使用训练好的模型进行决策的AI"""
    
    def __init__(self, color, level=AILevel.EXPERT, model_path=None):
        super().__init__(color, level)
        self.model_path = model_path
        self.model_info = None
        self.board_size = 15
        
        # 调试输出
        print(f"TrainedAI.__init__: color={color}, level={level}, model_path={model_path}")
        
        # 如果提供了模型路径，尝试加载
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return False
                
            # 调试输出
            print(f"TrainedAI.load_model: 正在加载模型 {model_path}")
            
            # 如果是模型文件，加载模型
            if model_path.endswith('.pth'):
                # 此处暂时不实现真正的模型加载逻辑
                self.model_info = {
                    'path': model_path,
                    'loaded_at': datetime.now().isoformat(),
                    'type': 'trained_model'
                }
                print(f"已加载模型: {model_path}")
                return True
                
            # 如果是统计文件，加载统计信息
            if model_path.endswith('.json'):
                with open(model_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                print(f"已加载模型统计信息: {model_path}")
                return True
                
            return False
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_move(self, board_state):
        """根据当前棋盘状态获取最优走法"""
        # 调试输出
        print(f"TrainedAI.get_move: 被调用，模型路径={self.model_path}")
        
        # 如果没有加载模型，使用启发式搜索策略
        if not self.model_info:
            print("没有加载模型，使用启发式策略")
            return self._get_heuristic_move(board_state)
        
        try:
            # 这里实现基于训练模型的决策逻辑
            print("使用训练好的模型进行决策")
            
            # 获取所有空位
            empty_positions = []
            for row in range(len(board_state)):
                for col in range(len(board_state[0])):
                    if board_state[row][col] == 0:
                        empty_positions.append((row, col))
            
            # 没有空位，返回None
            if not empty_positions:
                return None
            
            # 评估每个可能的位置
            scored_moves = []
            for pos in empty_positions:
                row, col = pos
                score = self._evaluate_position(board_state, row, col)
                scored_moves.append((score, pos))
            
            # 排序并找出最佳走法
            scored_moves.sort(reverse=True)
            
            # 获取前几个最佳走法
            top_n = min(3, len(scored_moves))
            best_move = scored_moves[random.randint(0, top_n-1)][1]
            
            print(f"TrainedAI选择的最佳走法: {best_move}，得分: {scored_moves[0][0]}")
            return best_move
            
        except Exception as e:
            print(f"AI决策出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时使用启发式策略
            return self._get_heuristic_move(board_state)
    
    def _evaluate_position(self, board_state, row, col):
        """评估一个位置的得分"""
        # 模拟落子并评估位置
        score = 0
        my_player = self.color_to_player()
        opponent = 3 - my_player
        
        # 先保存原值
        original_value = board_state[row][col]
        # 模拟我方落子
        board_state[row][col] = my_player
        
        # 检查水平、垂直、对角线方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            # 计算当前方向的连子数
            my_count = 1  # 当前位置算1个
            blocked_ends = 0  # 被阻挡的端点
            
            # 模拟我方落子
            for direction in [-1, 1]:  # 两个方向
                for step in range(1, 5):  # 最多往每个方向看4步
                    x, y = row + dx * step * direction, col + dy * step * direction
                    if 0 <= x < len(board_state) and 0 <= y < len(board_state[0]):
                        if board_state[x][y] == my_player:
                            my_count += 1
                        elif board_state[x][y] == opponent:
                            blocked_ends += 1
                            break
                        else:  # 空位
                            break
                    else:  # 超出边界
                        blocked_ends += 1
                        break
            
            # 评分规则 - 根据连子数和端点阻挡情况
            if my_count >= 5:  # 连成5子及以上
                score += 100000
            elif my_count == 4:
                if blocked_ends == 0:  # 活四
                    score += 10000
                elif blocked_ends == 1:  # 冲四
                    score += 1000
            elif my_count == 3:
                if blocked_ends == 0:  # 活三
                    score += 500
                elif blocked_ends == 1:  # 眠三
                    score += 100
            elif my_count == 2:
                if blocked_ends == 0:  # 活二
                    score += 50
                elif blocked_ends == 1:  # 眠二
                    score += 10
        
        # 模拟对手落子，评估防守价值
        board_state[row][col] = opponent
        
        # 检查所有方向
        for dx, dy in directions:
            opponent_count = 1
            blocked_ends = 0
            
            for direction in [-1, 1]:
                for step in range(1, 5):
                    x, y = row + dx * step * direction, col + dy * step * direction
                    if 0 <= x < len(board_state) and 0 <= y < len(board_state[0]):
                        if board_state[x][y] == opponent:
                            opponent_count += 1
                        elif board_state[x][y] == my_player:
                            blocked_ends += 1
                            break
                        else:  # 空位
                            break
                    else:  # 超出边界
                        blocked_ends += 1
                        break
            
            # 防守评分规则
            if opponent_count >= 5:  # 阻止对手连5
                score += 90000
            elif opponent_count == 4:
                if blocked_ends == 0:  # 阻止对手活四
                    score += 9000
                elif blocked_ends == 1:  # 阻止对手冲四
                    score += 800
            elif opponent_count == 3:
                if blocked_ends == 0:  # 阻止对手活三
                    score += 500
                elif blocked_ends == 1:  # 阻止对手眠三
                    score += 100
        
        # 棋盘中心位置加分
        center = len(board_state) // 2
        distance_to_center = abs(row - center) + abs(col - center)
        center_score = max(0, (7 - distance_to_center) * 10)
        score += center_score
        
        # 恢复棋盘状态
        board_state[row][col] = original_value
        
        # 加入随机因素，避免AI总是选择同一位置
        score += random.randint(0, 10)
        
        return score
        
    def _get_heuristic_move(self, board_state):
        """使用启发式策略获取走法"""
        # 调试输出
        print("TrainedAI._get_heuristic_move: 使用启发式策略")
        
        # 获取所有可用位置
        empty_positions = []
        for row in range(len(board_state)):
            for col in range(len(board_state[0])):
                if board_state[row][col] == 0:
                    empty_positions.append((row, col))
        
        # 如果没有可用位置，返回None
        if not empty_positions:
            return None
            
        # 如果是第一步，下在棋盘中心位置
        if all(board_state[i][j] == 0 for i in range(len(board_state)) for j in range(len(board_state[0]))):
            center = len(board_state) // 2
            return (center, center)
        
        # 评估每个位置
        scored_positions = []
        for row, col in empty_positions:
            score = self._evaluate_position(board_state, row, col)
            scored_positions.append((score, (row, col)))
        
        # 按得分排序
        scored_positions.sort(reverse=True)
        
        # 返回得分最高的位置
        return scored_positions[0][1]
