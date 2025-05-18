import numpy as np
from typing import Dict, Tuple, List, Set, Optional


class TranspositionTable:
    """
    置换表 - 缓存已计算过的局面评估结果
    
    使用Zobrist哈希来高效地识别等价局面，避免重复计算
    """
    
    # 节点类型
    EXACT = 0       # 精确值
    LOWER_BOUND = 1 # 下界（Alpha截断）
    UPPER_BOUND = 2 # 上界（Beta截断）
    
    def __init__(self, max_size=1000000):
        """初始化置换表
        
        Args:
            max_size: 表的最大条目数，防止内存占用过大
        """
        self.max_size = max_size
        self.table = {}  # 使用字典存储，键为局面哈希，值为(depth, score, flag, best_move)元组
        self._zobrist_keys = None  # 延迟初始化Zobrist键
    
    def initialize_zobrist_keys(self, board_size=15):
        """初始化Zobrist哈希键
        
        为每个位置和每种棋子类型生成唯一的随机数
        
        Args:
            board_size: 棋盘大小
        """
        # 随机数生成器
        rng = np.random.default_rng(42)  # 使用固定种子以确保键的一致性
        
        # 为每个位置的每种棋子类型生成64位随机数
        # [位置][棋子类型]，棋子类型：1=黑，2=白
        self._zobrist_keys = np.zeros((board_size, board_size, 3), dtype=np.int64)
        
        # 填充随机数
        for i in range(board_size):
            for j in range(board_size):
                for k in range(1, 3):  # 跳过0（空位）
                    self._zobrist_keys[i][j][k] = rng.integers(1, 2**64-1)
    
    def compute_hash(self, board):
        """计算棋盘状态的哈希值
        
        Args:
            board: 棋盘状态
            
        Returns:
            64位哈希值
        """
        if self._zobrist_keys is None:
            self.initialize_zobrist_keys(len(board))
            
        hash_value = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                piece = board[i][j]
                if piece != 0:  # 如果有棋子
                    hash_value ^= self._zobrist_keys[i][j][piece]
        
        return hash_value
    
    def lookup(self, hash_value, depth, alpha, beta):
        """查找局面
        
        Args:
            hash_value: 局面哈希值
            depth: 当前搜索深度
            alpha: Alpha值
            beta: Beta值
            
        Returns:
            如果找到匹配项，返回(score, best_move, True)，否则返回(0, None, False)
        """
        if hash_value in self.table:
            stored_depth, score, flag, best_move = self.table[hash_value]
            
            if stored_depth >= depth:  # 存储的结果适用于当前或更深的搜索
                if flag == self.EXACT:  # 精确值
                    return score, best_move, True
                
                if flag == self.LOWER_BOUND and score >= beta:  # 下界截断
                    return beta, best_move, True
                
                if flag == self.UPPER_BOUND and score <= alpha:  # 上界截断
                    return alpha, best_move, True
        
        return 0, None, False  # 未找到有用的条目
    
    def store(self, hash_value, depth, score, flag, best_move):
        """存储局面评估结果
        
        Args:
            hash_value: 局面哈希值
            depth: 搜索深度
            score: 评估分数
            flag: 节点类型（EXACT/LOWER_BOUND/UPPER_BOUND）
            best_move: 最佳走法
        """
        # 如果表已满，移除一些旧条目
        if len(self.table) >= self.max_size:
            # 简单策略：随机删除10%的条目
            keys_to_delete = np.random.choice(
                list(self.table.keys()),
                size=int(self.max_size * 0.1),
                replace=False
            )
            for key in keys_to_delete:
                del self.table[key]
        
        # 存储新条目
        self.table[hash_value] = (depth, score, flag, best_move)
    
    def clear(self):
        """清空置换表"""
        self.table.clear()


class HistoryHeuristic:
    """
    历史启发表 - 跟踪走法在搜索中的成功历史
    
    记录不同走法导致剪枝的频率，用于未来搜索中优先考虑历史上好的走法
    """
    
    def __init__(self, board_size=15):
        """初始化历史启发表
        
        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size
        self.history_table = np.zeros((board_size, board_size), dtype=np.int32)
    
    def add(self, move, depth):
        """增加走法的历史价值
        
        剪枝效果越好的走法，历史价值越高，价值与深度的平方成正比
        
        Args:
            move: 走法坐标(row, col)
            depth: 当前搜索深度
        """
        row, col = move
        # 历史价值增加量与深度的平方成正比
        self.history_table[row][col] += depth * depth
    
    def get(self, move):
        """获取走法的历史价值
        
        Args:
            move: 走法坐标(row, col)
            
        Returns:
            历史价值分数
        """
        row, col = move
        return self.history_table[row][col]
    
    def sort_moves(self, moves):
        """根据历史价值对走法列表排序
        
        Args:
            moves: 走法列表[(row, col), ...]
            
        Returns:
            按历史价值降序排序的走法列表
        """
        return sorted(moves, key=lambda move: self.get(move), reverse=True)
    
    def clear(self):
        """清空历史启发表"""
        self.history_table.fill(0)


class KillerMoveTable:
    """
    杀手走法表 - 存储在同一层搜索中导致剪枝的走法
    
    与历史启发不同，杀手走法专注于当前搜索的特定深度
    """
    
    def __init__(self, max_depth=20):
        """初始化杀手走法表
        
        Args:
            max_depth: 最大搜索深度
        """
        self.max_depth = max_depth
        # 为每一层深度存储两个杀手走法
        self.killer_moves = [[None, None] for _ in range(max_depth + 1)]
    
    def add(self, move, depth):
        """添加杀手走法
        
        Args:
            move: 走法坐标(row, col)
            depth: 当前搜索深度
        """
        if depth > self.max_depth:
            return
            
        # 确保不记录重复的杀手走法
        if move != self.killer_moves[depth][0]:
            # 第二个杀手走法被第一个替换，新走法成为第一个
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
    
    def get(self, depth):
        """获取指定深度的杀手走法
        
        Args:
            depth: 搜索深度
            
        Returns:
            该深度的杀手走法列表
        """
        if depth > self.max_depth:
            return []
            
        # 过滤掉None值
        return [move for move in self.killer_moves[depth] if move is not None]
    
    def clear(self):
        """清空杀手走法表"""
        self.killer_moves = [[None, None] for _ in range(self.max_depth + 1)]


class MoveOrderer:
    """
    走法排序器 - 整合多种启发式方法对走法进行排序
    
    结合MVV-LVA（Most Valuable Victim - Least Valuable Aggressor）、
    历史启发、杀手走法等技术，确定走法的优先级
    """
    
    def __init__(self, board_size=15):
        """初始化走法排序器
        
        Args:
            board_size: 棋盘大小
        """
        self.history = HistoryHeuristic(board_size)
        self.killers = KillerMoveTable()
        self.tt_move = None  # 来自置换表的最佳走法
    
    def set_tt_move(self, move):
        """设置来自置换表的最佳走法
        
        Args:
            move: 走法坐标(row, col)
        """
        self.tt_move = move
    
    def score_move(self, move, board, player, depth):
        """对走法进行评分
        
        Args:
            move: 走法坐标(row, col)
            board: 当前棋盘状态
            player: 当前玩家
            depth: 当前搜索深度
            
        Returns:
            走法的评分
        """
        row, col = move
        score = 0
        
        # 1. 置换表走法有最高优先级
        if self.tt_move == move:
            return 10000000
        
        # 2. 杀手走法有第二优先级
        killer_moves = self.killers.get(depth)
        if move in killer_moves:
            return 1000000 + (1 if move == killer_moves[0] else 0)  # 第一个杀手走法优先级更高
        
        # 3. 历史启发
        score += self.history.get(move)
        
        # 4. 战术评估 - 检查是否能形成连子或阻止对手连子
        # 模拟落子并进行简单评估
        original = board[row][col]
        board[row][col] = player
        
        # 检查是否能获胜
        if self._is_winning_move(board, row, col, player):
            score += 100000
        
        # 检查是否能阻止对手获胜
        opponent = 3 - player
        board[row][col] = opponent
        if self._is_winning_move(board, row, col, opponent):
            score += 50000
        
        # 恢复棋盘
        board[row][col] = original
        
        # 5. 位置评估 - 中心位置更有价值
        center = len(board) // 2
        distance_to_center = abs(row - center) + abs(col - center)
        # 距离越近越好，所以用负距离
        score += (30 - distance_to_center) * 10
        
        return score
    
    def order_moves(self, moves, board, player, depth):
        """对走法列表进行排序
        
        Args:
            moves: 走法列表[(row, col), ...]
            board: 当前棋盘状态
            player: 当前玩家
            depth: 当前搜索深度
            
        Returns:
            按评分降序排序的走法列表
        """
        # 计算每个走法的得分并排序
        move_scores = [(move, self.score_move(move, board, player, depth)) for move in moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, _ in move_scores]
    
    def update_history(self, move, depth):
        """更新走法的历史价值
        
        Args:
            move: 走法坐标(row, col)
            depth: 当前搜索深度
        """
        self.history.add(move, depth)
    
    def update_killer(self, move, depth):
        """更新杀手走法
        
        Args:
            move: 走法坐标(row, col)
            depth: 当前搜索深度
        """
        self.killers.add(move, depth)
    
    def clear(self):
        """清空所有历史数据"""
        self.history.clear()
        self.killers.clear()
        self.tt_move = None
    
    def _is_winning_move(self, board, row, col, player):
        """检查走法是否能获胜
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            走法是否能获胜
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、两个对角线
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 向一个方向计数
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 向反方向计数
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 检查是否达到五连
            if count >= 5:
                return True
        
        return False


class CandidateMoveGenerator:
    """
    候选走法生成器 - 生成和筛选有意义的候选走法
    
    通过各种启发式方法，只考虑最有可能的走法，大幅减小搜索空间
    """
    
    def __init__(self, board_size=15):
        """初始化候选走法生成器
        
        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size
        self.max_candidates = 20  # 最大候选数量
    
    def generate_candidates(self, board, player, top_n=None):
        """生成候选走法列表
        
        Args:
            board: 当前棋盘状态
            player: 当前玩家
            top_n: 返回的候选数量，None表示使用默认值
            
        Returns:
            候选走法列表[(row, col), ...]
        """
        if top_n is None:
            top_n = self.max_candidates
        
        size = len(board)
        
        # 检查棋盘是否为空
        if self._is_empty_board(board):
            center = size // 2
            return [(center, center)]  # 空棋盘只考虑中心点
        
        # 候选走法集合
        candidates = set()
        
        # 1. 寻找能直接获胜的走法
        winning_moves = self._find_winning_moves(board, player)
        if winning_moves:
            return winning_moves  # 找到直接获胜的走法，立即返回
        
        # 2. 寻找需要防守的关键点
        opponent = 3 - player
        defense_moves = self._find_winning_moves(board, opponent)
        for move in defense_moves:
            candidates.add(move)
        
        # 3. 寻找形成潜在威胁的走法
        threat_moves = self._find_threat_moves(board, player)
        for move in threat_moves:
            candidates.add(move)
        
        # 4. 寻找已有棋子周围的空位
        neighbor_moves = self._find_neighbor_moves(board)
        for move in neighbor_moves:
            candidates.add(move)
        
        # 如果候选数量不足，增加一些合理的走法
        if len(candidates) < top_n:
            extra_moves = self._generate_extra_moves(board, list(candidates))
            candidates.update(extra_moves[:top_n - len(candidates)])
        
        # 对候选走法进行评分和排序
        scored_candidates = [(move, self._score_candidate(move, board, player)) for move in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回指定数量的候选
        return [move for move, _ in scored_candidates[:top_n]]
    
    def _is_empty_board(self, board):
        """检查棋盘是否为空
        
        Args:
            board: 棋盘状态
            
        Returns:
            棋盘是否为空
        """
        for row in board:
            for cell in row:
                if cell != 0:
                    return False
        return True
    
    def _find_winning_moves(self, board, player):
        """寻找能直接获胜的走法
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            能直接获胜的走法列表
        """
        winning_moves = []
        size = len(board)
        
        # 遍历所有空位
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue
                
                # 模拟落子
                board[row][col] = player
                
                # 检查是否获胜
                if self._is_winning_move(board, row, col, player):
                    winning_moves.append((row, col))
                
                # 恢复棋盘
                board[row][col] = 0
        
        return winning_moves
    
    def _find_threat_moves(self, board, player):
        """寻找能形成威胁的走法（如形成活三、冲四等）
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            能形成威胁的走法列表
        """
        threat_moves = []
        size = len(board)
        
        # 遍历所有空位
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue
                
                # 模拟落子
                board[row][col] = player
                
                # 检查是否形成冲四或活三
                if (self._is_open_four(board, row, col, player) or 
                    self._is_four(board, row, col, player) or
                    self._is_open_three(board, row, col, player)):
                    threat_moves.append((row, col))
                
                # 恢复棋盘
                board[row][col] = 0
        
        return threat_moves
    
    def _find_neighbor_moves(self, board):
        """寻找已有棋子周围的空位
        
        Args:
            board: 棋盘状态
            
        Returns:
            已有棋子周围的空位列表
        """
        neighbor_moves = []
        size = len(board)
        checked = set()
        
        # 先找出所有已有棋子的位置
        pieces = []
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    pieces.append((row, col))
        
        # 寻找这些棋子周围的空位
        for piece_row, piece_col in pieces:
            for dr in range(-2, 3):  # 扩大到2格范围
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    
                    row, col = piece_row + dr, piece_col + dc
                    if (0 <= row < size and 0 <= col < size and 
                        board[row][col] == 0 and 
                        (row, col) not in checked):
                        neighbor_moves.append((row, col))
                        checked.add((row, col))
        
        return neighbor_moves
    
    def _generate_extra_moves(self, board, existing_moves):
        """生成额外的候选走法
        
        当现有候选数量不足时使用
        
        Args:
            board: 棋盘状态
            existing_moves: 现有的候选走法
            
        Returns:
            额外的候选走法列表
        """
        size = len(board)
        extra_moves = []
        
        # 计算棋盘中心
        center = size // 2
        
        # 生成距离中心由近到远的所有空位
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0 and (r, c) not in existing_moves:
                    # 计算到中心的距离作为权重
                    distance = abs(r - center) + abs(c - center)
                    extra_moves.append(((r, c), distance))
        
        # 按距离排序
        extra_moves.sort(key=lambda x: x[1])
        
        # 只返回走法坐标，不返回距离
        return [move for move, _ in extra_moves]
    
    def _score_candidate(self, move, board, player):
        """评估候选走法的价值
        
        Args:
            move: 走法坐标(row, col)
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            走法的评分
        """
        row, col = move
        score = 0
        size = len(board)
        center = size // 2
        
        # 1. 优先考虑中心区域
        distance_to_center = abs(row - center) + abs(col - center)
        score -= distance_to_center * 10
        
        # 模拟落子
        original = board[row][col]
        
        # 2. 检查进攻价值
        board[row][col] = player
        
        # 直接获胜
        if self._is_winning_move(board, row, col, player):
            score += 10000
        
        # 形成活四
        elif self._is_open_four(board, row, col, player):
            score += 5000
        
        # 形成冲四
        elif self._is_four(board, row, col, player):
            score += 1000
        
        # 形成活三
        elif self._is_open_three(board, row, col, player):
            score += 500
        
        # 3. 检查防守价值
        opponent = 3 - player
        board[row][col] = opponent
        
        # 阻止对手获胜
        if self._is_winning_move(board, row, col, opponent):
            score += 9000
        
        # 阻止对手形成活四
        elif self._is_open_four(board, row, col, opponent):
            score += 4500
        
        # 阻止对手形成冲四
        elif self._is_four(board, row, col, opponent):
            score += 900
        
        # 阻止对手形成活三
        elif self._is_open_three(board, row, col, opponent):
            score += 450
        
        # 恢复棋盘
        board[row][col] = original
        
        return score
    
    def _is_winning_move(self, board, row, col, player):
        """检查走法是否能获胜
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            走法是否能获胜
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、两个对角线
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 向一个方向计数
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 向反方向计数
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 检查是否达到五连
            if count >= 5:
                return True
        
        return False
    
    def _is_open_four(self, board, row, col, player):
        """检查是否形成活四
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            是否形成活四
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、两个对角线
        
        for dx, dy in directions:
            # 检查一个方向
            pattern = []
            for step in range(-4, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    pattern.append(board[r][c])
                else:
                    pattern.append(-1)  # 边界外
            
            # 转换为字符串，方便模式匹配
            pattern_str = ''.join(map(str, pattern)).replace('-1', 'X').replace('0', '_')
            
            # 活四模式：....PPPP....（两端都是空白）
            if f"_{player}{player}{player}{player}_" in pattern_str:
                return True
        
        return False
    
    def _is_four(self, board, row, col, player):
        """检查是否形成冲四
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            是否形成冲四
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        opponent = 3 - player
        
        for dx, dy in directions:
            # 检查一个方向
            pattern = []
            for step in range(-4, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    pattern.append(board[r][c])
                else:
                    pattern.append(-1)  # 边界外
            
            # 转换为字符串，方便模式匹配
            pattern_str = ''.join(map(str, pattern)).replace('-1', 'X').replace('0', '_')
            
            # 冲四模式：一端被挡住
            if (f"X{player}{player}{player}{player}_" in pattern_str or 
                f"_{player}{player}{player}{player}X" in pattern_str or
                f"{opponent}{player}{player}{player}{player}_" in pattern_str or
                f"_{player}{player}{player}{player}{opponent}" in pattern_str):
                return True
        
        return False
    
    def _is_open_three(self, board, row, col, player):
        """检查是否形成活三
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            是否形成活三
        """
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # 检查一个方向
            pattern = []
            for step in range(-4, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    pattern.append(board[r][c])
                else:
                    pattern.append(-1)  # 边界外
            
            # 转换为字符串，方便模式匹配
            pattern_str = ''.join(map(str, pattern)).replace('-1', 'X').replace('0', '_')
            
            # 活三模式：...PPP...（两端都是空白，可以形成活四）
            if (f"__{player}{player}{player}__" in pattern_str or
                f"_{player}_{player}{player}_" in pattern_str or
                f"_{player}{player}_{player}_" in pattern_str):
                return True
        
        return False


# 单元测试代码
if __name__ == "__main__":
    # 创建测试棋盘
    test_board = [[0 for _ in range(15)] for _ in range(15)]
    
    # 放置一些棋子进行测试
    test_board[7][7] = 1
    test_board[7][8] = 1
    test_board[7][9] = 1
    test_board[8][6] = 2
    test_board[9][5] = 2
    
    # 测试置换表
    tt = TranspositionTable()
    hash_val = tt.compute_hash(test_board)
    print(f"棋盘哈希值: {hash_val}")
    
    # 测试候选走法生成
    cg = CandidateMoveGenerator()
    candidates = cg.generate_candidates(test_board, 1)
    print(f"黑方候选走法: {candidates}")
    
    # 测试走法排序
    mo = MoveOrderer()
    ordered_moves = mo.order_moves(candidates, test_board, 1, 3)
    print(f"排序后的走法: {ordered_moves}")
