class GameState:
    def __init__(self):
        self.board = [[0 for _ in range(15)] for _ in range(15)]  # ��ʼ��һ��15x15������
        self.current_player = 1  # ��ǰ��ң�1��ʾ���壬2��ʾ����
        self.last_move = None  # ��¼���һ������λ��

    def make_move(self, row, col, player):
        """ִ��һ������"""
        if self.board[row][col] == 0:  # ����λ���Ƿ�Ϊ��
            self.board[row][col] = player
            self.last_move = (row, col)
            self.current_player = 3 - player  # �л����
        else:
            raise ValueError("Invalid move: Position already occupied.")