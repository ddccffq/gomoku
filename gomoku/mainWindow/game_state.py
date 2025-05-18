class GameState:
    def __init__(self):
        self.board = [[0 for _ in range(15)] for _ in range(15)]  # 初始化一个15x15的棋盘
        self.current_player = 1  # 当前玩家，1表示黑棋，2表示白棋
        self.last_move = None  # 记录最后一步落子位置

    def make_move(self, row, col, player):
        """执行一步落子"""
        if self.board[row][col] == 0:  # 检查该位置是否为空
            self.board[row][col] = player
            self.last_move = (row, col)
            self.current_player = 3 - player  # 切换玩家
        else:
            raise ValueError("Invalid move: Position already occupied.")