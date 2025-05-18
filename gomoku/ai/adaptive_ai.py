import random
import time
from typing import List, Tuple, Dict, Optional

from ai.base_ai import BaseAI, StoneColor, AILevel
from ai.board_evaluator import BoardEvaluator
from ai.search_algorithms import AlphaBetaSearch
from ai.difficulty_manager import DifficultyManager, AIStyle
from ai.easy_ai import EasyAI
from ai.hard_ai import HardAI
from ai.expert_ai import ExpertAI


class AdaptiveAI(BaseAI):
    """适应性AI
    
    特点:
    1. 动态调整难度以适应玩家水平
    2. 会根据玩家的表现自动提高或降低难度
    3. 能够识别和学习玩家的风格
    4. 综合使用不同难度级别的AI策略
    """
    
    def __init__(self, color=StoneColor.BLACK):
        super().__init__(color)
        self.difficulty_manager = DifficultyManager.instance()
        self.evaluator = BoardEvaluator()
        
        # 初始难度
        self.current_level = AILevel.HARD
        self.skill_rating = 0.5  # 0.0-1.0范围的技能评级
        
        # 创建不同难度的AI实例
        self.easy_ai = EasyAI(color)
        self.hard_ai = HardAI(color)
        self.expert_ai = ExpertAI(color)
        
        # AI实例映射
        self.ai_instances = {
            AILevel.EASY: self.easy_ai,
            AILevel.HARD: self.hard_ai,
            AILevel.EXPERT: self.expert_ai
        }
        
        # 适应性参数
        self.adaptation_factor = 0.1  # 每局调整的幅度
        self.player_strength = 0.5  # 估计的玩家水平 (0.0-1.0)
        self.consecutive_wins = 0  # 连续赢的局数
        self.consecutive_losses = 0  # 连续输的局数
        
        # 游戏历史记录
        self.game_history = []  # 记录过去的游戏结果
        self.player_moves = []  # 记录玩家的走法
        
        # "让子"机制参数
        self.allow_mistakes = False  # 是否允许有意识的错误
        self.mistake_probability = 0.0  # 错误概率
        
        # 当前游戏状态追踪
        self.player_value = 1 if color == StoneColor.BLACK else 2
        self.move_count = 0
        self.last_board_state = None
    
    def get_move(self, board: List[List[int]]) -> Tuple[int, int]:
        """获取下一步走法
        
        综合考虑难度级别、玩家水平和适应性参数
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            决策的走法坐标
        """
        print(f"AdaptiveAI思考中...")
        start_time = time.time()
        
        # 更新计数器和状态追踪
        self.move_count += 1
        
        # 分析棋盘变化，检测玩家走法
        if self.last_board_state:
            player_move = self._detect_player_move(self.last_board_state, board)
            if player_move:
                self.player_moves.append(player_move)
                # 分析玩家水平，每10步调整一次
                if len(self.player_moves) % 10 == 0:
                    self._update_player_strength()
        
        # 保存当前棋盘状态
        self.last_board_state = [row[:] for row in board]
        
        # 动态难度调整
        self._adjust_difficulty()
        
        # 根据当前难度决定使用哪个AI实例
        if random.random() > self.skill_rating:
            # 使用较低难度的AI
            if self.current_level == AILevel.EXPERT:
                ai_to_use = self.hard_ai
            else:
                ai_to_use = self.easy_ai
        else:
            # 使用当前难度的AI
            ai_to_use = self.ai_instances[self.current_level]
        
        # 获取AI走法
        move = ai_to_use.get_move(board)
        
        # "让子"机制 - 有概率选择次优解
        if self.allow_mistakes and random.random() < self.mistake_probability:
            suboptimal_move = self._get_suboptimal_move(board, move)
            if suboptimal_move:
                print(f"AdaptiveAI故意做出次优选择")
                move = suboptimal_move
        
        print(f"AdaptiveAI决策完成，难度级别:{self.current_level.name}，耗时:{time.time()-start_time:.2f}秒")
        return move
    
    def _detect_player_move(self, prev_board: List[List[int]], curr_board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """检测玩家的走法
        
        比较前后棋盘状态，找出玩家新落的子
        
        Args:
            prev_board: 之前的棋盘状态
            curr_board: 当前的棋盘状态
            
        Returns:
            玩家的走法，如果无法检测则返回None
        """
        opponent_value = 3 - self.player_value
        
        for r in range(len(curr_board)):
            for c in range(len(curr_board[0])):
                if prev_board[r][c] == 0 and curr_board[r][c] == opponent_value:
                    return (r, c)
        
        return None
    
    def _update_player_strength(self):
        """根据玩家的走法评估其水平"""
        if not self.player_moves:
            return
        
        # 分析玩家走法的质量
        quality_scores = []
        
        # 模拟一个棋盘用于分析
        sim_board = [[0 for _ in range(15)] for _ in range(15)]
        opponent_value = 3 - self.player_value
        
        for i, move in enumerate(self.player_moves[-10:]):  # 只分析最近10步
            row, col = move
            
            # 用专家AI评估这步棋
            expert_scores = []
            # 生成候选走法
            candidates = self._generate_candidates(sim_board, opponent_value, 5)
            
            if candidates:
                # 评估每个候选走法
                for candidate in candidates:
                    c_row, c_col = candidate
                    sim_board[c_row][c_col] = opponent_value
                    score = self.evaluator.evaluate_move(sim_board, c_row, c_col, 
                                                       StoneColor.BLACK if opponent_value == 1 else StoneColor.WHITE)
                    sim_board[c_row][c_col] = 0  # 恢复棋盘
                    expert_scores.append((candidate, score))
                
                # 排序找出最佳走法
                expert_scores.sort(key=lambda x: x[1], reverse=True)
                best_candidate = expert_scores[0][0] if expert_scores else None
                
                # 计算玩家走法的相对质量
                if best_candidate:
                    best_score = expert_scores[0][1]
                    # 找到玩家走法的评分
                    player_score = next((s for c, s in expert_scores if c == move), 0)
                    
                    # 相对质量 = 玩家评分 / 最佳评分
                    if best_score > 0:
                        relative_quality = min(1.0, player_score / best_score)
                        quality_scores.append(relative_quality)
            
            # 更新模拟棋盘，添加玩家走法
            sim_board[row][col] = opponent_value
        
        # 计算平均质量得分
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # 更新玩家水平估计，缓慢调整
        self.player_strength = 0.8 * self.player_strength + 0.2 * avg_quality
        print(f"更新玩家水平估计: {self.player_strength:.2f}")
    
    def _generate_candidates(self, board: List[List[int]], player: int, top_n: int) -> List[Tuple[int, int]]:
        """生成候选走法
        
        简化实现，仅生成周围空位
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            top_n: 生成的候选数量
            
        Returns:
            候选走法列表
        """
        size = len(board)
        candidates = []
        
        # 如果棋盘为空，返回中心点
        is_empty = all(all(cell == 0 for cell in row) for row in board)
        if is_empty:
            return [(size // 2, size // 2)]
        
        # 找出所有已落子的位置
        pieces = []
        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    pieces.append((r, c))
        
        # 在已有棋子周围找空位
        checked = set()
        for pr, pc in pieces:
            # 检查3x3范围内的空位
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    
                    r, c = pr + dr, pc + dc
                    if 0 <= r < size and 0 <= c < size and board[r][c] == 0:
                        if (r, c) not in checked:
                            candidates.append((r, c))
                            checked.add((r, c))
                            
                            # 如果已经有足够的候选，直接返回
                            if len(candidates) >= top_n:
                                return candidates
        
        # 如果候选不足，添加更多空位
        if len(candidates) < top_n:
            for r in range(size):
                for c in range(size):
                    if board[r][c] == 0 and (r, c) not in checked:
                        candidates.append((r, c))
                        if len(candidates) >= top_n:
                            break
                if len(candidates) >= top_n:
                    break
        
        return candidates
    
    def _adjust_difficulty(self):
        """根据游戏历史和玩家水平调整难度"""
        # 基于玩家水平调整技能评级
        target_skill = self.player_strength
        
        # 调整技能评级，缓慢趋近目标
        adjustment = (target_skill - self.skill_rating) * self.adaptation_factor
        self.skill_rating = max(0.0, min(1.0, self.skill_rating + adjustment))
        
        # 根据技能评级设置当前难度级别
        if self.skill_rating < 0.3:
            self.current_level = AILevel.EASY
        elif self.skill_rating < 0.7:
            self.current_level = AILevel.HARD
        else:
            self.current_level = AILevel.EXPERT
        
        # 调整"让子"机制参数
        self.allow_mistakes = self.skill_rating > 0.6  # 高水平玩家需要更高难度
        self.mistake_probability = max(0.0, (0.7 - self.player_strength) * 0.2)  # 水平越低，错误概率越高
        
        print(f"难度调整: 级别={self.current_level.name}, 技能评级={self.skill_rating:.2f}, 让子概率={self.mistake_probability:.2f}")
    
    def _get_suboptimal_move(self, board: List[List[int]], best_move: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """获取一个次优解走法，用于"让子"机制
        
        Args:
            board: 棋盘状态
            best_move: 最佳走法
            
        Returns:
            次优解走法，如果没有则返回None
        """
        candidates = self._generate_candidates(board, self.player_value, 10)
        
        if best_move in candidates:
            candidates.remove(best_move)  # 移除最佳走法
        
        if not candidates:  # 如果没有其他候选走法
            return None
            
        # 评估候选走法
        scored_moves = []
        for move in candidates:
            row, col = move
            # 模拟落子
            board[row][col] = self.player_value
            
            # 使用评估器评分
            score = self.evaluator.evaluate_move(
                board, row, col, 
                StoneColor.BLACK if self.player_value == 1 else StoneColor.WHITE
            )
            
            # 恢复棋盘
            board[row][col] = 0
            
            scored_moves.append((move, score))
        
        # 排序
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前三名中的一个次优解
        candidates = [move for move, _ in scored_moves[:3]]
        if candidates:
            return random.choice(candidates)
        
        return None
    
    def notify_game_result(self, won: bool):
        """通知游戏结果，用于调整难度
        
        Args:
            won: AI是否获胜
        """
        self.game_history.append(won)
        
        if won:
            # AI赢了，增加连续胜利计数
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # 连续胜利太多，可能需要降低难度
            if self.consecutive_wins >= 3:
                self.skill_rating = max(0.0, self.skill_rating - 0.05)
                print(f"连续获胜{self.consecutive_wins}次，降低技能评级至{self.skill_rating:.2f}")
        else:
            # AI输了，增加连续失败计数
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # 连续失败太多，可能需要提高难度
            if self.consecutive_losses >= 2:
                self.skill_rating = min(1.0, self.skill_rating + 0.05)
                print(f"连续失败{self.consecutive_losses}次，提高技能评级至{self.skill_rating:.2f}")
        
        # 重置游戏状态
        self.move_count = 0
        self.last_board_state = None
        self.player_moves = []
