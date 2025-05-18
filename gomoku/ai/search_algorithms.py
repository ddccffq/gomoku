import time
import math
from typing import List, Tuple, Dict, Set, Optional, Callable

from ai.base_ai import StoneColor
from ai.board_evaluator import BoardEvaluator


class AlphaBetaSearch:
    """Alpha-Beta剪枝搜索算法
    
    实现了带时间控制的Alpha-Beta剪枝搜索算法，包含节点排序和置换表优化
    """
    
    def __init__(self, evaluator: BoardEvaluator, color: StoneColor, max_depth=4, time_limit=5.0):
        """初始化Alpha-Beta搜索对象
        
        Args:
            evaluator: 棋盘评估器
            color: AI的棋子颜色
            max_depth: 最大搜索深度
            time_limit: 搜索时间限制(秒)
        """
        self.evaluator = evaluator
        self.color = color
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = 0
        self.nodes_explored = 0
        self.transposition_table = {}  # 置换表，用于记录已搜索过的局面
        self.history_table = {}  # 历史启发表，记录每个位置的历史成功率
        self.killer_moves = []  # 杀手启发表，记录在相同层级上的好走法
        
        # 初始化每一层的杀手走法列表 (每层保存两个杀手走法)
        for _ in range(max_depth + 1):
            self.killer_moves.append([None, None])
    
    def search(self, board: List[List[int]]) -> Tuple[int, int]:
        """执行搜索并返回最佳走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            最佳的落子位置 (row, col)
        """
        self.start_time = time.time()
        self.nodes_explored = 0
        
        # 初始化最佳走法
        best_move = None
        
        # 清空杀手走法
        for i in range(len(self.killer_moves)):
            self.killer_moves[i] = [None, None]
        
        # 迭代加深搜索
        for depth in range(1, self.max_depth + 1):
            try:
                move, score = self._iterative_search(board, depth)
                best_move = move
                
                # 如果找到了必胜走法或者必败走法，则立即返回
                if score > 90000 or score < -90000:
                    print(f"发现必胜或必败走法: {move}, 分数: {score}")
                    break
                    
                # 检查时间是否用尽
                if time.time() - self.start_time > self.time_limit * 0.8:
                    print(f"搜索达到时间限制，完成深度 {depth}")
                    break
                    
            except TimeoutError:
                print(f"搜索超时，完成深度 {depth-1}")
                break
        
        # 打印统计信息
        elapsed = time.time() - self.start_time
        print(f"搜索完成: 耗时 {elapsed:.2f}秒, 探索节点 {self.nodes_explored}, 每秒节点 {self.nodes_explored/elapsed:.0f}")
        
        return best_move
    
    def _iterative_search(self, board: List[List[int]], depth: int) -> Tuple[Tuple[int, int], float]:
        """执行特定深度的迭代搜索"""
        # 生成所有候选走法
        moves = self._get_candidate_moves(board)
        
        # 排序走法
        self._sort_moves(board, moves)
        
        # 初始化最佳分数和走法
        best_score = float('-inf')
        best_move = moves[0] if moves else None
        
        # 记录alpha值，用于窗口优化
        alpha = float('-inf')
        beta = float('inf')
        
        # 对每个走法进行搜索
        for move in moves:
            # 检查是否超时
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError("搜索超时")
            
            # 模拟落子
            row, col = move
            board[row][col] = 1 if self.color == StoneColor.BLACK else 2
            
            # 对立方视角执行搜索
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, False)
            
            # 撤销落子
            board[row][col] = 0
            
            # 更新最佳分数和走法
            if score > best_score:
                best_score = score
                best_move = move
                
                # 更新alpha值
                alpha = max(alpha, score)
            
            # 打印调试信息
            print(f"深度 {depth}, 走法 {move}, 分数 {score}")
        
        return best_move, best_score
    
    def _alpha_beta(self, board: List[List[int]], depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Alpha-Beta剪枝搜索的主函数
        
        Args:
            board: 当前棋盘状态
            depth: 当前搜索深度
            alpha: Alpha值
            beta: Beta值
            is_maximizing: 是否是最大化层
            
        Returns:
            当前局面的评估分数
        """
        self.nodes_explored += 1
        
        # 检查是否超时
        if self.nodes_explored % 1000 == 0 and time.time() - self.start_time > self.time_limit:
            raise TimeoutError("搜索超时")
        
        # 生成局面哈希值
        board_hash = self._get_board_hash(board)
        
        # 检查置换表
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            stored_depth, stored_score, score_type = entry
            
            # 如果存储的深度大于等于当前深度，可以使用存储的分数
            if stored_depth >= depth:
                if score_type == 'exact':
                    return stored_score
                elif score_type == 'lower_bound' and stored_score >= beta:
                    return stored_score
                elif score_type == 'upper_bound' and stored_score <= alpha:
                    return stored_score
        
        # 检查终止条件: 达到最大深度
        if depth == 0:
            player_color = self.color if is_maximizing else StoneColor.WHITE if self.color == StoneColor.BLACK else StoneColor.BLACK
            return self.evaluator.evaluate_board(board, player_color)
        
        # 检查是否有胜利者
        winner = self._check_winner(board)
        if winner:
            if (winner == 1 and self.color == StoneColor.BLACK) or (winner == 2 and self.color == StoneColor.WHITE):
                return 100000 - 100 * (self.max_depth - depth)  # 尽快获胜
            else:
                return -100000 + 100 * (self.max_depth - depth)  # 尽量拖延失败
        
        # 获取所有候选走法
        moves = self._get_candidate_moves(board)
        
        # 排序走法
        self._sort_moves(board, moves, depth)
        
        # 初始化最佳分数
        best_score = float('-inf') if is_maximizing else float('inf')
        
        # Alpha-Beta搜索
        score_type = 'upper_bound' if is_maximizing else 'lower_bound'
        
        for move in moves:
            row, col = move
            
            # 确定当前玩家的棋子颜色
            if is_maximizing:
                player_value = 1 if self.color == StoneColor.BLACK else 2
            else:
                player_value = 2 if self.color == StoneColor.BLACK else 1
                
            # 模拟落子
            board[row][col] = player_value
            
            # 递归搜索
            score = self._alpha_beta(board, depth - 1, alpha, beta, not is_maximizing)
            
            # 撤销落子
            board[row][col] = 0
            
            # 更新最佳分数
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    score_type = 'exact'
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    score_type = 'exact'
                beta = min(beta, best_score)
            
            # 剪枝
            if beta <= alpha:
                # 记录杀手走法
                if is_maximizing and score >= beta:
                    self._update_killer_moves(move, depth)
                elif not is_maximizing and score <= alpha:
                    self._update_killer_moves(move, depth)
                
                # 记录历史表
                if move in self.history_table:
                    self.history_table[move] += depth * depth
                else:
                    self.history_table[move] = depth * depth
                
                break
        
        # 更新置换表
        self.transposition_table[board_hash] = (depth, best_score, score_type)
        
        return best_score
    
    def _get_candidate_moves(self, board: List[List[int]]) -> List[Tuple[int, int]]:
        """获取候选走法
        
        根据当前棋盘状态，获取所有有意义的候选走法，而不是枚举所有空位
        """
        size = len(board)
        moves = []
        checked = set()
        
        # 扫描整个棋盘，寻找非空位置
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    # 搜索此棋子周围的空位
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue
                            
                            r, c = row + dr, col + dc
                            if 0 <= r < size and 0 <= c < size and board[r][c] == 0:
                                pos = (r, c)
                                if pos not in checked:
                                    moves.append(pos)
                                    checked.add(pos)
        
        # 如果没有找到任何走法（棋盘为空），返回中心点
        if not moves:
            center = size // 2
            return [(center, center)]
        
        return moves
    
    def _sort_moves(self, board: List[List[int]], moves: List[Tuple[int, int]], depth: int = None):
        """对候选走法进行排序
        
        使用多种启发式方法对走法进行排序，以提高剪枝效率
        """
        # 准备走法评分
        move_scores = []
        
        for move in moves:
            score = 0
            
            # 1. 使用历史表启发
            if move in self.history_table:
                score += self.history_table[move]
            
            # 2. 使用杀手走法启发
            if depth is not None:
                if move in self.killer_moves[depth]:
                    score += 10000  # 优先尝试杀手走法
            
            # 3. 使用简单的棋型评估启发（例如，检查是否能五连）
            row, col = move
            player_value = 1 if self.color == StoneColor.BLACK else 2
            
            # 模拟落子
            board[row][col] = player_value
            
            # 检查是否能形成五连
            if self._is_winning_move(board, row, col):
                score += 100000
            
            # 撤销落子
            board[row][col] = 0
            
            # 收集分数
            move_scores.append((move, score))
        
        # 根据分数降序排序
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 更新走法列表
        for i in range(len(moves)):
            if i < len(move_scores):
                moves[i] = move_scores[i][0]
    
    def _update_killer_moves(self, move: Tuple[int, int], depth: int):
        """更新杀手走法表
        
        Args:
            move: 要记录的杀手走法
            depth: 当前搜索深度
        """
        # 确保不重复记录相同的走法
        if move != self.killer_moves[depth][0]:
            # 将第一个杀手走法移到第二位，新的走法放到第一位
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
    
    def _get_board_hash(self, board: List[List[int]]) -> int:
        """计算棋盘状态的哈希值，用于置换表"""
        h = 0
        size = len(board)
        
        for row in range(size):
            for col in range(size):
                piece = board[row][col]
                if piece != 0:
                    # 使用Zobrist哈希
                    h ^= hash((row, col, piece)) & ((1 << 64) - 1)
        
        return h
    
    def _check_winner(self, board: List[List[int]]) -> int:
        """检查棋盘上是否有胜者
        
        Returns:
            胜者 (1=黑, 2=白) 或 0 (无胜者)
        """
        size = len(board)
        
        # 检查水平、垂直和对角线方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    continue
                
                player = board[row][col]
                
                for dr, dc in directions:
                    count = 1
                    
                    # 检查一个方向
                    for i in range(1, 5):
                        r, c = row + dr * i, col + dc * i
                        if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                            count += 1
                        else:
                            break
                    
                    # 检查反方向
                    for i in range(1, 5):
                        r, c = row - dr * i, col - dc * i
                        if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                            count += 1
                        else:
                            break
                    
                    # 检查是否有五个或更多连续棋子
                    if count >= 5:
                        return player
        
        return 0
    
    def _is_winning_move(self, board: List[List[int]], row: int, col: int) -> bool:
        """检查指定位置的棋子是否形成胜利
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            
        Returns:
            是否是获胜的走法
        """
        size = len(board)
        player = board[row][col]
        
        # 检查四个方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # 检查一个方向
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 检查反方向
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        
        return False


class EnhancedAlphaBetaSearch:
    """增强版Alpha-Beta剪枝搜索算法
    
    集成了多种高级优化技术：
    1. 置换表：避免重复计算局面
    2. 历史启发：优先尝试历史上好的走法
    3. 杀手启发：优先尝试在相似局面中有效的关键走法
    4. 候选点筛选：只考虑有意义的落子点，减少搜索空间
    """
    
    def __init__(self, evaluator, color, max_depth=4, time_limit=5.0):
        """初始化增强版Alpha-Beta搜索
        
        Args:
            evaluator: 棋盘评估器
            color: AI的棋子颜色
            max_depth: 最大搜索深度
            time_limit: 搜索时间限制(秒)
        """
        self.evaluator = evaluator
        self.color = color
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        # 初始化计时和统计
        self.start_time = 0
        self.nodes_explored = 0
        
        # 尝试初始化优化组件，如果导入失败则使用基本功能
        try:
            from ai.optimization_techniques import TranspositionTable, MoveOrderer, CandidateMoveGenerator
            self.transposition_table = TranspositionTable()
            self.move_orderer = MoveOrderer()
            self.candidate_generator = CandidateMoveGenerator()
            self.use_optimizations = True
        except ImportError:
            print("警告: 无法导入优化技术模块，将使用基本功能")
            self.use_optimizations = False
        
        # AI玩家值（1=黑, 2=白）
        self.player_value = 1 if color == StoneColor.BLACK else 2
        self.opponent_value = 3 - self.player_value
    
    def search(self, board):
        """执行搜索并返回最佳走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            最佳的落子位置 (row, col)
        """
        self.start_time = time.time()
        self.nodes_explored = 0
        
        # 初始化最佳走法
        best_move = None
        
        # 检查直接胜利走法
        winning_moves = self._check_immediate_win(board, self.player_value)
        if winning_moves:
            print(f"发现直接获胜走法: {winning_moves[0]}")
            return winning_moves[0]
        
        # 检查防守关键点
        opponent = 3 - self.player_value
        critical_moves = self._check_immediate_win(board, opponent)
        if critical_moves:
            print(f"发现关键防守点: {critical_moves[0]}")
            return critical_moves[0]
        
        try:
            # 生成候选走法
            if self.use_optimizations:
                candidate_moves = self.candidate_generator.generate_candidates(board, self.player_value, 20)
            else:
                candidate_moves = self._generate_candidate_moves(board)
            
            # 排序候选走法
            if self.use_optimizations:
                ordered_moves = self.move_orderer.order_moves(candidate_moves, board, self.player_value, 0)
            else:
                ordered_moves = candidate_moves
            
            # 初始化Alpha-Beta值
            alpha = float('-inf')
            beta = float('inf')
            best_score = float('-inf')
            
            # 对每个候选走法进行搜索
            for move in ordered_moves:
                # 检查时间限制
                if time.time() - self.start_time > self.time_limit:
                    break
                
                row, col = move
                # 模拟落子
                board[row][col] = self.player_value
                
                # 计算局面评分
                score = -self._alpha_beta(board, self.max_depth - 1, -beta, -alpha, False)
                
                # 恢复棋盘
                board[row][col] = 0
                
                # 更新最佳走法
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
            
            # 如果没找到最佳走法（可能是搜索超时），返回第一个候选走法
            if best_move is None and ordered_moves:
                best_move = ordered_moves[0]
        except Exception as e:
            print(f"搜索过程出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 如果仍然没有找到走法，选择一个随机有效位置
        if best_move is None:
            best_move = self._select_fallback_move(board)
        
        return best_move
    
    def _alpha_beta(self, board, depth, alpha, beta, is_maximizing):
        """增强版Alpha-Beta剪枝搜索
        
        Args:
            board: 当前棋盘状态
            depth: 当前搜索深度
            alpha: Alpha值
            beta: Beta值
            is_maximizing: 是否是最大化层
            
        Returns:
            当前局面的评估分数
        """
        self.nodes_explored += 1
        
        # 检查是否超时
        if self.nodes_explored % 1000 == 0 and time.time() - self.start_time > self.time_limit:
            return 0  # 超时，返回中性评分
        
        # 使用置换表（如果可用）
        if self.use_optimizations:
            hash_value = self.transposition_table.compute_hash(board)
            score, best_move, found = self.transposition_table.lookup(hash_value, depth, alpha, beta)
            if found:
                return score
        
        # 获取当前玩家
        player = self.player_value if is_maximizing else self.opponent_value
        
        # 基线条件：叶子节点评估
        if depth == 0:
            score = self.evaluator.evaluate_board(board, self.color)
            return score if is_maximizing else -score
        
        # 检查是否有一方获胜
        winner = self._check_winner(board)
        if winner:
            if winner == self.player_value:
                return 100000  # 我方获胜
            else:
                return -100000  # 对方获胜
        
        # 生成候选走法
        if self.use_optimizations:
            candidate_moves = self.candidate_generator.generate_candidates(
                board, player, 15 if depth <= 2 else 10
            )
        else:
            candidate_moves = self._generate_candidate_moves(board)
        
        # 排序走法
        if self.use_optimizations:
            ordered_moves = self.move_orderer.order_moves(candidate_moves, board, player, depth)
        else:
            ordered_moves = candidate_moves
        
        # Alpha-Beta搜索
        best_score = float('-inf') if is_maximizing else float('inf')
        
        for move in ordered_moves:
            row, col = move
            
            # 模拟走棋
            board[row][col] = player
            
            # 递归搜索
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, not is_maximizing)
            
            # 撤销走棋
            board[row][col] = 0
            
            # 更新最佳分数
            if is_maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
            
            # 剪枝
            if beta <= alpha:
                break
        
        # 存储到置换表（如果可用）
        if self.use_optimizations:
            flag = (
                self.transposition_table.EXACT if alpha < best_score < beta
                else self.transposition_table.LOWER_BOUND if best_score >= beta
                else self.transposition_table.UPPER_BOUND
            )
            self.transposition_table.store(hash_value, depth, best_score, flag, None)
        
        return best_score
    
    def _check_immediate_win(self, board, player):
        """检查是否有直接获胜的走法
        
        Args:
            board: 棋盘状态
            player: 玩家ID (1=黑, 2=白)
            
        Returns:
            获胜走法列表，如果没有则返回空列表
        """
        winning_moves = []
        size = len(board)
        
        # 检查所有空位
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue
                    
                # 模拟落子
                board[row][col] = player
                
                # 检查是否获胜
                if self._check_win(board, row, col, player):
                    winning_moves.append((row, col))
                
                # 恢复棋盘
                board[row][col] = 0
                
                # 如果找到获胜走法，立即返回
                if winning_moves:
                    return winning_moves
        
        return winning_moves
    
    def _check_win(self, board, row, col, player):
        """检查指定位置是否形成胜利
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 玩家ID
        
        Returns:
            True如果形成胜利，否则False
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 检查一个方向
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 检查相反方向
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 如果达到五子连珠，则获胜
            if count >= 5:
                return True
        
        return False
    
    def _check_winner(self, board):
        """检查棋盘上是否有一方已经获胜
        
        Args:
            board: 棋盘状态
            
        Returns:
            获胜玩家ID，如果没有则返回None
        """
        size = len(board)
        
        # 检查所有棋子
        for row in range(size):
            for col in range(size):
                player = board[row][col]
                if player != 0:  # 如果有棋子
                    if self._check_win(board, row, col, player):
                        return player
        
        return None
    
    def _generate_candidate_moves(self, board):
        """生成候选走法
        
        默认实现：选择所有有棋子相邻的空位
        
        Args:
            board: 棋盘状态
            
        Returns:
            候选走法列表
        """
        size = len(board)
        candidates = []
        checked = set()
        
        # 检查棋盘是否为空
        is_empty = True
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    is_empty = False
                    break
            if not is_empty:
                break
        
        # 如果棋盘为空，返回中心点
        if is_empty:
            center = size // 2
            return [(center, center)]
        
        # 查找所有已有棋子
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:  # 有棋子
                    # 检查周围3x3范围内的空位
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue
                            
                            r, c = row + dr, col + dc
                            if (0 <= r < size and 0 <= c < size and 
                                board[r][c] == 0 and (r, c) not in checked):
                                candidates.append((r, c))
                                checked.add((r, c))
        
        # 如果没有找到候选走法（理论上不应该发生），返回所有空位
        if not candidates:
            for row in range(size):
                for col in range(size):
                    if board[row][col] == 0:
                        candidates.append((row, col))
        
        return candidates
    
    def _select_fallback_move(self, board):
        """当正常搜索失败时，选择一个备用走法
        
        Args:
            board: 棋盘状态
            
        Returns:
            备用走法
        """
        size = len(board)
        center = size // 2
        
        # 先尝试中心点
        if board[center][center] == 0:
            return (center, center)
        
        # 然后尝试中心点周围的位置
        for d in range(1, size):
            # 以中心为起点，螺旋向外搜索
            for i in range(-d, d+1):
                # 检查上下边界
                if center + i >= 0 and center + i < size:
                    # 检查左边界上的点
                    if center - d >= 0 and board[center + i][center - d] == 0:
                        return (center + i, center - d)
                    
                    # 检查右边界上的点
                    if center + d < size and board[center + i][center + d] == 0:
                        return (center + i, center + d)
                
                # 检查左右边界
                if center + i >= 0 and center + i < size:
                    # 检查上边界上的点
                    if center - d >= 0 and board[center - d][center + i] == 0:
                        return (center - d, center + i)
                    
                    # 检查下边界上的点
                    if center + d < size and board[center + d][center + i] == 0:
                        return (center + d, center + i)
        
        # 最后尝试任何空位
        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    return (row, col)
        
        # 如果没有找到空位（棋盘已满），返回中心点
        # 这种情况理论上不应该发生
        return (center, center)


class IterativeDeepeningSearch:
    """迭代加深搜索算法
    
    逐步增加搜索深度，直到达到最大深度或时间限制
    使用AlphaBeta搜索作为基础搜索算法
    """
    
    def __init__(self, evaluator, color, initial_depth=2, max_depth=5, time_limit=3.0):
        """初始化迭代加深搜索
        
        Args:
            evaluator: 棋盘评估器
            color: AI的棋子颜色
            initial_depth: 初始搜索深度
            max_depth: 最大搜索深度
            time_limit: 搜索时间限制(秒)
        """
        self.evaluator = evaluator
        self.color = color
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        # 创建Alpha-Beta搜索实例
        self.alpha_beta = AlphaBetaSearch(evaluator, color, max_depth, time_limit)
    
    def search(self, board):
        """执行迭代加深搜索
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            最佳走法 (row, col)
        """
        start_time = time.time()
        best_move = None
        
        # 检查直接获胜走法
        winning_moves = self._find_winning_move(board)
        if winning_moves:
            print(f"找到直接获胜走法: {winning_moves[0]}")
            return winning_moves[0]
        
        # 检查防守关键点
        opponent = 3 - (1 if self.color == StoneColor.BLACK else 2)
        critical_moves = self._find_blocking_move(board)
        if critical_moves:
            print(f"找到关键防守点: {critical_moves[0]}")
            return critical_moves[0]
        
        # 从初始深度开始，逐渐增加搜索深度
        for depth in range(self.initial_depth, self.max_depth + 1):
            print(f"开始深度 {depth} 的搜索...")
            
            # 设置Alpha-Beta搜索的深度
            self.alpha_beta.max_depth = depth
            
            # 计算这一深度的时间预算
            elapsed = time.time() - start_time
            remaining_time = self.time_limit - elapsed
            
            # 如果剩余时间不足，停止增加搜索深度
            if remaining_time < 0.1:
                print(f"时间不足，停止搜索")
                break
                
            # 为这一深度设置时间限制
            self.alpha_beta.time_limit = remaining_time
            
            try:
                # 执行这一深度的搜索
                move = self.alpha_beta.search(board)
                
                # 如果搜索成功，更新最佳走法
                if move:
                    best_move = move
                    print(f"深度 {depth} 的最佳走法: {best_move}")
                    
                    # 检查是否还有足够的时间进行更深层次的搜索
                    if time.time() - start_time > self.time_limit * 0.8:
                        print(f"已用时 {time.time() - start_time:.2f}秒，停止增加深度")
                        break
            except Exception as e:
                print(f"深度 {depth} 搜索出错: {e}")
                break
        
        # 如果没有找到走法，使用一个备用策略
        if not best_move:
            best_move = self._get_fallback_move(board)
            
        return best_move
    
    def _find_winning_move(self, board):
        """寻找直接获胜的走法"""
        player = 1 if self.color == StoneColor.BLACK else 2
        winning_moves = []
        
        # 检查每个空位
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] != 0:
                    continue
                
                # 模拟落子
                board[row][col] = player
                
                # 检查是否获胜
                if self._check_win(board, row, col, player):
                    winning_moves.append((row, col))
                
                # 恢复棋盘
                board[row][col] = 0
                
                # 如果找到获胜走法，直接返回
                if winning_moves:
                    return winning_moves
        
        return None
    
    def _find_blocking_move(self, board):
        """寻找需要阻止对手获胜的关键点"""
        opponent = 3 - (1 if self.color == StoneColor.BLACK else 2)
        critical_moves = []
        
        # 检查每个空位
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] != 0:
                    continue
                
                # 模拟对手落子
                board[row][col] = opponent
                
                # 检查对手是否获胜
                if self._check_win(board, row, col, opponent):
                    critical_moves.append((row, col))
                
                # 恢复棋盘
                board[row][col] = 0
                
                # 如果找到关键点，直接返回
                if critical_moves:
                    return critical_moves
        
        return None
    
    def _check_win(self, board, row, col, player):
        """检查指定位置是否形成胜利"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 检查一个方向
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < len(board) and 0 <= c < len(board) and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 检查反方向
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < len(board) and 0 <= c < len(board) and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 如果达到五子连珠，则获胜
            if count >= 5:
                return True
        
        return False
    
    def _get_fallback_move(self, board):
        """当常规搜索失败时使用的备用策略"""
        # 寻找中心点附近的空位
        center = len(board) // 2
        for d in range(0, len(board)):
            for dx in range(-d, d+1):
                for dy in range(-d, d+1):
                    if dx == 0 and dy == 0:
                        # 先检查中心点
                        if board[center][center] == 0:
                            return (center, center)
                        continue
                    
                    x, y = center + dx, center + dy
                    if 0 <= x < len(board) and 0 <= y < len(board) and board[x][y] == 0:
                        return (x, y)
                        
        # 如果找不到空位(理论上不可能)
        return (center, center)


# 兼容旧版引用：将基本ID搜索别名为“OptimizedIterativeDeepeningSearch”
# 如果已经有一个真正的OptimizedIterativeDeepeningSearch类，可移除此别名
OptimizedIterativeDeepeningSearch = IterativeDeepeningSearch


if __name__ == "__main__":
    # 测试代码
    from ai.board_evaluator import BoardEvaluator
    
    # 创建15x15的空棋盘
    test_board = [[0 for _ in range(15)] for _ in range(15)]
    
    # 放置一些棋子进行测试
    test_board[7][7] = 1  # 黑棋
    test_board[7][8] = 2  # 白棋
    test_board[8][7] = 1  # 黑棋
    
    # 初始化评估器
    evaluator = BoardEvaluator()
    
    # 创建搜索器
    ids = IterativeDeepeningSearch(evaluator, StoneColor.BLACK, initial_depth=2, max_depth=5, time_limit=2.0)
    
    # 执行搜索
    best_move = ids.search(test_board)
    
    print(f"最佳走法: {best_move}")
