# coding:utf-8

class MoveGenerator:
    """走法生成器，用于生成和筛选候选走法"""
    
    @staticmethod
    def generate_moves(board, player, top_n=None):
        """生成有效的候选走法
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            top_n: 保留的最佳走法数量，None表示保留所有
            
        Returns:
            候选走法列表 [(row, col), ...]
        """
        size = len(board)
        valid_moves = []
        
        # 如果棋盘为空，返回中心点
        if MoveGenerator._is_empty_board(board):
            center = size // 2
            return [(center, center)]
        
        # 获取所有非空位置
        non_empty = []
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    non_empty.append((row, col))
        
        # 获取棋子的边界范围，并适当扩大搜索范围
        min_row = max(0, min(row for row, _ in non_empty) - 2)
        max_row = min(size - 1, max(row for row, _ in non_empty) + 2)
        min_col = max(0, min(col for _, col in non_empty) - 2)
        max_col = min(size - 1, max(col for _, col in non_empty) + 2)
        
        # 在棋子周围寻找空位
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if board[row][col] == 0:  # 空位
                    # 检查周围是否有棋子
                    if MoveGenerator._has_neighbor(board, row, col):
                        valid_moves.append((row, col))
        
        # 如果没有找到有效走法，返回所有空位
        if not valid_moves:
            for row in range(size):
                for col in range(size):
                    if board[row][col] == 0:
                        valid_moves.append((row, col))
        
        # 如果指定了top_n，对走法进行评分和筛选
        if top_n is not None and top_n < len(valid_moves):
            scored_moves = []
            
            # 简单评分各个走法
            for move in valid_moves:
                score = MoveGenerator._score_move(board, move[0], move[1], player)
                scored_moves.append((move, score))
            
            # 按分数排序并取前top_n个
            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in scored_moves[:top_n]]
        
        return valid_moves
    
    @staticmethod
    def _is_empty_board(board):
        """检查棋盘是否为空
        
        Args:
            board: 棋盘状态
            
        Returns:
            True如果棋盘为空，否则False
        """
        for row in board:
            for cell in row:
                if cell != 0:
                    return False
        return True
    
    @staticmethod
    def _has_neighbor(board, row, col, distance=2):
        """检查指定位置周围是否有棋子
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            distance: 检查范围
            
        Returns:
            True如果周围有棋子，否则False
        """
        size = len(board)
        
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size and board[r][c] != 0:
                    return True
        
        return False
    
    @staticmethod
    def _score_move(board, row, col, player):
        """使用简单启发式规则评分走法
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 当前玩家
            
        Returns:
            该走法的评分
        """
        size = len(board)
        score = 0
        opponent = 3 - player
        
        # 检查是否能直接获胜
        board[row][col] = player
        if MoveGenerator._check_win(board, row, col, player):
            board[row][col] = 0
            return 10000  # 直接获胜是最高优先级
        board[row][col] = 0
        
        # 检查是否需要防守（对手下在这里能否获胜）
        board[row][col] = opponent
        if MoveGenerator._check_win(board, row, col, opponent):
            board[row][col] = 0
            return 9000  # 防守是次高优先级
        board[row][col] = 0
        
        # 计算连子数量
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # 计算己方连子
            board[row][col] = player
            self_count = MoveGenerator._count_consecutive(board, row, col, dx, dy)
            board[row][col] = 0
            
            # 计算对手连子
            board[row][col] = opponent
            opponent_count = MoveGenerator._count_consecutive(board, row, col, dx, dy)
            board[row][col] = 0
            
            # 评分连子
            if self_count >= 4:
                score += 1000  # 四连
            elif self_count == 3:
                score += 100   # 三连
            elif self_count == 2:
                score += 10    # 二连
            
            if opponent_count >= 4:
                score += 900   # 阻止对手四连
            elif opponent_count == 3:
                score += 90    # 阻止对手三连
        
        # 优先选择靠近中心的位置
        center = size // 2
        distance_to_center = abs(row - center) + abs(col - center)
        score -= distance_to_center  # 距离中心越近越好
        
        return score
    
    @staticmethod
    def _check_win(board, row, col, player):
        """检查指定位置是否能形成五连胜利
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            player: 玩家ID
            
        Returns:
            True如果能形成五连，否则False
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        size = len(board)
        
        for dx, dy in directions:
            count = 1  # 当前位置算一个
            
            # 向正方向检查
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 向反方向检查
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        
        return False
    
    @staticmethod
    def _count_consecutive(board, row, col, dx, dy):
        """计算指定方向上的连子数量
        
        Args:
            board: 棋盘状态
            row: 行坐标
            col: 列坐标
            dx: x方向增量
            dy: y方向增量
            
        Returns:
            连子数量
        """
        size = len(board)
        player = board[row][col]
        count = 1  # 当前位置算一个
        
        # 向正方向检查
        for step in range(1, 5):
            r, c = row + dx * step, col + dy * step
            if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
            else:
                break
        
        # 向反方向检查
        for step in range(1, 5):
            r, c = row - dx * step, col - dy * step
            if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
            else:
                break
        
        return count
