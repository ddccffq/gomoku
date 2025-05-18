# coding:utf-8
from typing import List, Tuple

class VCTSearch:
    """VCT(Victory by Threats) 搜索实现"""

    def __init__(self, max_depth=10, time_limit=2.0):
        self.max_depth = max_depth
        self.time_limit = time_limit

    def search(self, board: List[List[int]], player: int) -> List[Tuple[int, int]]:
        """执行VCT搜索"""
        # 搜索逻辑实现
        pass

    def _check_win(self, board: List[List[int]], player: int) -> bool:
        """检查是否形成五连"""
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for row in range(size):
            for col in range(size):
                if board[row][col] != player:
                    continue

                for dx, dy in directions:
                    count = 1
                    for step in range(1, 5):
                        r, c = row + dx * step, col + dy * step
                        if 0 <= r < size and 0 <= c < size and board[r][c] == player:
                            count += 1
                        else:
                            break

                    if count >= 5:
                        return True

        return False

    def _find_defensive_moves(self, board: List[List[int]], player: int, threat_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """寻找所有可能的防守着法"""
        size = len(board)
        defenses = []

        # 首先尝试直接获胜
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue

                # 模拟落子
                board[row][col] = player

                # 检查是否形成获胜
                if self._check_win(board, player):
                    board[row][col] = 0
                    return [(row, col)]  # 找到直接获胜的走法，立即返回

                # 撤销落子
                board[row][col] = 0

        # 寻找阻止对方威胁的走法
        threat_row, threat_col = threat_point
        opponent = 3 - player

        # 威胁位置特殊处理
        board[threat_row][threat_col] = opponent

        # 检查对手是否能在下一步获胜
        winning_points = []
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue

                # 模拟落子
                board[row][col] = opponent

                # 检查是否形成获胜
                if self._check_win(board, opponent):
                    winning_points.append((row, col))

                # 撤销落子
                board[row][col] = 0

        # 恢复棋盘
        board[threat_row][threat_col] = 0

        # 如果有多个获胜点，必须全部阻止
        if len(winning_points) > 1:
            return []  # 无法防守

        # 如果只有一个获胜点，则必须在该点防守
        if winning_points:
            defenses.append(winning_points[0])
            return defenses

        # 对于威胁周围的位置进行搜索
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue

                r, c = threat_row + dr, threat_col + dc
                if 0 <= r < size and 0 <= c < size and board[r][c] == 0:
                    defenses.append((r, c))

        return defenses

    def _is_four(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否形成四连(冲四)"""
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # 当前位置计为1
            blocked_ends = 0  # 被阻挡的端点数量

            # 检查正方向
            for step in range(1, 5):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        break
                    else:  # 被对手棋子阻挡
                        blocked_ends += 1
                        break
                else:  # 出界
                    blocked_ends += 1
                    break

            # 检查反方向
            for step in range(1, 5):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        break
                    else:  # 被对手棋子阻挡
                        blocked_ends += 1
                        break
                else:  # 出界
                    blocked_ends += 1
                    break

            # 判断是否为冲四(四连但至少一端被阻挡)
            if count == 4 and blocked_ends <= 1:
                return True

        return False

    def _is_open_three(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否形成活三"""
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # 当前位置计为1
            open_ends = 0  # 开放端点数量

            # 检查正方向
            for step in range(1, 4):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        open_ends += 1
                        break
                    else:  # 被对手棋子阻挡
                        break
                else:  # 出界
                    break

            # 检查反方向
            for step in range(1, 4):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        open_ends += 1
                        break
                    else:  # 被对手棋子阻挡
                        break
                else:  # 出界
                    break

            # 判断是否为活三(三连且两端都开放)
            if count == 3 and open_ends == 2:
                return True

        return False

    def _is_blocked_three(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否形成眠三"""
        size = len(board)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # 当前位置计为1
            open_ends = 0  # 开放端点数量

            # 检查正方向
            for step in range(1, 4):
                r, c = row + dx * step, col + dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        open_ends += 1
                        break
                    else:  # 被对手棋子阻挡
                        break
                else:  # 出界
                    break

            # 检查反方向
            for step in range(1, 4):
                r, c = row - dx * step, col - dy * step
                if 0 <= r < size and 0 <= c < size:
                    if board[r][c] == player:
                        count += 1
                    elif board[r][c] == 0:
                        open_ends += 1
                        break
                    else:  # 被对手棋子阻挡
                        break
                else:  # 出界
                    break

            # 判断是否为眠三(三连但只有一端开放)
            if count == 3 and open_ends == 1:
                return True

        return False


class VCFSearch(VCTSearch):
    """VCF(Victory by Consecutive Fours) 搜索"""

    def __init__(self, max_depth=10, time_limit=2.0):
        super().__init__(max_depth, time_limit)

    def _find_threat_points(self, board: List[List[int]], player: int) -> List[Tuple[Tuple[int, int], int]]:
        """寻找所有威胁点，但只考虑能形成冲四或直接获胜的点"""
        size = len(board)
        threats = []

        # 遍历所有空位
        for row in range(size):
            for col in range(size):
                if board[row][col] != 0:
                    continue

                # 模拟落子
                board[row][col] = player

                # 检查是否形成获胜
                if self._check_win(board, player):
                    threats.append(((row, col), 0))  # 0表示获胜威胁
                    board[row][col] = 0
                    continue

                # 检查是否形成冲四
                if self._is_four(board, row, col, player):
                    threats.append(((row, col), 1))  # 1表示冲四威胁

                # 撤销落子
                board[row][col] = 0

        # 按威胁类型排序
        threats.sort(key=lambda x: x[1])

        return threats


# 单元测试代码
if __name__ == "__main__":
    # 创建测试棋盘
    test_board = [[0 for _ in range(15)] for _ in range(15)]

    # 放置一些棋子来测试
    test_board[7][7] = 1  # 黑
    test_board[7][8] = 1  # 黑
    test_board[7][10] = 1  # 黑
    test_board[7][11] = 1  # 黑

    test_board[8][9] = 2  # 白

    # 创建VCF搜索器
    vcf = VCFSearch(max_depth=9, time_limit=3.0)

    # 执行搜索
    winning_sequence = vcf.search(test_board, 1)  # 黑棋

    if winning_sequence:
        print(f"找到必胜序列: {winning_sequence}")
    else:
        print("没有找到必胜序列")