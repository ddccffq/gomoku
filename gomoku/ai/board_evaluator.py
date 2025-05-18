from enum import Enum, auto
import numpy as np
from ai.base_ai import StoneColor


class ChessPattern(Enum):
    """棋型枚举类"""
    FIVE = auto()           # 连五 (五个相连)
    OPEN_FOUR = auto()      # 活四 (两端都空的四连)
    FOUR = auto()           # 冲四 (只有一端开放的四连)
    OPEN_THREE = auto()     # 活三 (两端都空的三连)
    THREE = auto()          # 眠三 (只有一端开放的三连)
    OPEN_TWO = auto()       # 活二 (两端都空的二连)
    TWO = auto()            # 眠二 (只有一端开放的二连)
    NONE = auto()           # 无特殊棋型


class PatternScore:
    """棋型分数"""
    FIVE = 100000           # 连五
    OPEN_FOUR = 10000       # 活四
    FOUR = 1000             # 冲四
    OPEN_THREE = 500        # 活三
    THREE = 100             # 眠三
    OPEN_TWO = 50           # 活二
    TWO = 10                # 眠二
    
    # 复合棋型分数
    FOUR_FOUR = 8000        # 双冲四
    FOUR_OPEN_THREE = 5000  # 冲四活三
    OPEN_THREE_OPEN_THREE = 3000  # 双活三


class GomokuPatternEvaluator:
    """五子棋棋型评估器
    
    识别和评估五子棋中的各种棋型，如连五、活四、冲四、活三等
    """
    
    # 棋型的分数定义
    SCORE_FIVE = 10000  # 连五
    SCORE_OPEN_FOUR = 5000  # 活四
    SCORE_FOUR = 1000  # 冲四
    SCORE_OPEN_THREE = 500  # 活三
    SCORE_THREE = 200  # 眠三
    SCORE_OPEN_TWO = 100  # 活二
    SCORE_TWO = 50  # 眠二
    
    def __init__(self, board_size=15):
        """初始化棋型评估器
        
        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、右斜、左斜
    
    def evaluate_board(self, board, player):
        """评估棋盘对特定玩家的价值
        
        Args:
            board: 棋盘状态
            player: 待评估的玩家
            
        Returns:
            float: 棋盘状态的评分
        """
        score = 0
        opponent = 3 - player  # 对手编号
        
        # 检查是否已经获胜
        is_win, winner = self.check_win(board)
        if is_win:
            if winner == player:
                return self.SCORE_FIVE * 10  # 己方获胜
            else:
                return -self.SCORE_FIVE * 10  # 对手获胜
        
        # 逐个点评估棋型
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[y][x] == player:
                    # 评估己方棋子周围的棋型
                    score += self._evaluate_point(board, y, x, player)
                elif board[y][x] == opponent:
                    # 评估对手棋子周围的棋型，但权重减半（防守价值低于进攻）
                    score -= self._evaluate_point(board, y, x, opponent) * 0.6
        
        return score
    
    def _evaluate_point(self, board, y, x, player):
        """评估特定位置的棋型
        
        Args:
            board: 棋盘状态
            y, x: 位置坐标
            player: 待评估的玩家
            
        Returns:
            float: 该位置的棋型评分
        """
        score = 0
        
        # 检查四个方向
        for dx, dy in self.directions:
            # 记录当前方向上的棋型
            line = self._get_line(board, y, x, dx, dy, player)
            
            # 评估棋型
            if self._is_five(line):
                score += self.SCORE_FIVE
            elif self._is_open_four(line):
                score += self.SCORE_OPEN_FOUR
            elif self._is_four(line):
                score += self.SCORE_FOUR
            elif self._is_open_three(line):
                score += self.SCORE_OPEN_THREE
            elif self._is_three(line):
                score += self.SCORE_THREE
            elif self._is_open_two(line):
                score += self.SCORE_OPEN_TWO
            elif self._is_two(line):
                score += self.SCORE_TWO
        
        return score
    
    def _get_line(self, board, y, x, dx, dy, player):
        """获取某方向上的棋型
        
        Args:
            board: 棋盘状态
            y, x: 位置坐标
            dx, dy: 方向向量
            player: 棋子类型
            
        Returns:
            str: 形如 'XOOOOX' 的字符串表示，X表示空位，O表示己方棋子，B表示边界或对手棋子
        """
        opponent = 3 - player
        line = "O"  # 中心点
        
        # 向一个方向扩展
        for step in range(1, 5):
            ny, nx = y + dy * step, x + dx * step
            if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                if board[ny][nx] == player:
                    line += "O"
                elif board[ny][nx] == 0:
                    line += "X"
                else:
                    line += "B"
                    break
            else:
                line += "B"  # 边界
                break
        
        # 向相反方向扩展
        for step in range(1, 5):
            ny, nx = y - dy * step, x - dx * step
            if 0 <= ny < self.board_size and 0 <= nx < self.board_size:
                if board[ny][nx] == player:
                    line = "O" + line
                elif board[ny][nx] == 0:
                    line = "X" + line
                else:
                    line = "B" + line
                    break
            else:
                line = "B" + line  # 边界
                break
        
        return line
    
    # 以下是各种棋型的判断方法
    
    def _is_five(self, line):
        """连五"""
        return "OOOOO" in line
    
    def _is_open_four(self, line):
        """活四：形如 XOOOOX"""
        return "XOOOOX" in line
    
    def _is_four(self, line):
        """冲四：形如 XOOOO, OOOOX"""
        patterns = ["XOOOO", "OOOOX"]
        return any(pattern in line for pattern in patterns)
    
    def _is_open_three(self, line):
        """活三：形如 XOOOXX, XXOOOX"""
        patterns = ["XOOOXX", "XXOOOX", "XOXOOX", "XOOXOX"]
        return any(pattern in line for pattern in patterns)
    
    def _is_three(self, line):
        """眠三：形如 BOOOX, XOOOB"""
        patterns = ["BOOOX", "XOOOB", "BOOXOX", "XOXOOB"]
        return any(pattern in line for pattern in patterns)
    
    def _is_open_two(self, line):
        """活二：形如 XXOOXX"""
        patterns = ["XXOOXX", "XOXXOX", "XXOXX"]
        return any(pattern in line for pattern in patterns)
    
    def _is_two(self, line):
        """眠二：形如 BOOXX, XXOOB"""
        patterns = ["BOOXX", "XXOOB"]
        return any(pattern in line for pattern in patterns)
    
    def check_win(self, board):
        """检查是否有玩家获胜
        
        Args:
            board: 棋盘状态
            
        Returns:
            tuple: (是否结束, 胜者) 胜者为0表示平局
        """
        # 检查每个玩家是否有五连珠
        for player in [1, 2]:
            # 遍历棋盘上的每个位置
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board[y][x] != player:
                        continue
                    
                    # 检查四个方向
                    for dx, dy in self.directions:
                        count = 1  # 当前位置算一个
                        
                        # 向一个方向检查
                        for step in range(1, 5):
                            ny, nx = y + dy * step, x + dx * step
                            if (0 <= ny < self.board_size and 
                                0 <= nx < self.board_size and 
                                board[ny][nx] == player):
                                count += 1
                            else:
                                break
                        
                        # 如果不到5个，继续检查另一方向
                        if count < 5:
                            continue
                        
                        # 有玩家获胜
                        return True, player
        
        # 检查是否和局（棋盘已满）
        if np.all(board != 0):
            return True, 0
        
        # 游戏继续
        return False, -1
    
    def find_key_defense_points(self, board, player):
        """找出需要防守的关键点
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            list: [(y, x, score)] 防守点列表，按重要性排序
        """
        opponent = 3 - player
        defense_points = []
        
        # 首先检查对手是否有威胁性棋型
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[y][x] != 0:
                    continue  # 只考虑空位
                
                # 模拟对手在此处落子
                board[y][x] = opponent
                threat_score = self._evaluate_point(board, y, x, opponent)
                board[y][x] = 0  # 恢复空位
                
                # 如果是高威胁位置（如对手可以形成活四、冲四或活三）
                if threat_score >= self.SCORE_OPEN_THREE:
                    defense_points.append((y, x, threat_score))
        
        # 按威胁程度降序排序
        defense_points.sort(key=lambda p: p[2], reverse=True)
        
        return defense_points
