# coding:utf-8

class TranspositionTable:
    """置换表，用于缓存已搜索过的棋盘状态评分，避免重复搜索"""

    # 节点类型
    EXACT = 0    # 精确值
    LOWERBOUND = 1  # 下界值
    UPPERBOUND = 2  # 上界值
    
    def __init__(self, max_size=1000000):
        """初始化置换表
        
        Args:
            max_size: 置换表的最大条目数，防止内存溢出
        """
        self.table = {}  # 使用字典存储置换表条目
        self.max_size = max_size
    
    def store(self, zobrist_key, depth, score, node_type, best_move=None):
        """存储搜索结果
        
        Args:
            zobrist_key: Zobrist哈希值，作为棋盘状态的唯一标识
            depth: 搜索深度
            score: 评分
            node_type: 节点类型（精确值、下界值、上界值）
            best_move: 最佳走法，可选
        """
        # 如果表已满，执行简单的替换策略（这里可以用更复杂的策略）
        if len(self.table) >= self.max_size:
            self._cleanup()
        
        self.table[zobrist_key] = {
            'depth': depth,
            'score': score,
            'type': node_type,
            'best_move': best_move,
        }
    
    def lookup(self, zobrist_key):
        """查找棋盘状态
        
        Args:
            zobrist_key: Zobrist哈希值
        
        Returns:
            如果找到匹配项，返回存储的条目；否则返回None
        """
        return self.table.get(zobrist_key)
    
    def _cleanup(self):
        """当表满时清理旧条目"""
        # 简单策略：删除一半的条目（优先保留深度较大的条目）
        entries = list(self.table.items())
        # 按搜索深度排序
        entries.sort(key=lambda item: item[1]['depth'])
        # 删除深度较小的一半条目
        for key, _ in entries[:len(entries)//2]:
            del self.table[key]


class ZobristHashing:
    """Zobrist哈希，用于高效地为棋盘状态生成唯一哈希值"""
    
    def __init__(self, board_size=15, num_players=2):
        """初始化Zobrist哈希表
        
        Args:
            board_size: 棋盘大小
            num_players: 玩家数量
        """
        import random
        self.board_size = board_size
        self.num_players = num_players
        # 为每个位置、每种棋子生成随机数
        self.zobrist_table = [
            [
                [random.getrandbits(64) for _ in range(num_players + 1)]  # +1 是为了包含空位
                for _ in range(board_size)
            ]
            for _ in range(board_size)
        ]
        
        # 记录当前轮到哪个玩家的随机数
        self.player_hash = [random.getrandbits(64) for _ in range(num_players + 1)]
    
    def compute_hash(self, board, current_player=None):
        """计算棋盘状态的哈希值
        
        Args:
            board: 当前棋盘状态，二维list
            current_player: 当前玩家，1为黑棋，2为白棋
        
        Returns:
            int: 64位哈希值
        """
        h = 0
        # 计算棋盘状态的哈希值
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != 0:
                    # 异或操作，使得哈希值可逆
                    h ^= self.zobrist_table[i][j][board[i][j]]
        
        # 如果提供了当前玩家，将其纳入哈希值计算
        if current_player is not None:
            h ^= self.player_hash[current_player]
        
        return h
    
    def update_hash(self, prev_hash, row, col, player_id, prev_player_id=None, next_player_id=None):
        """增量更新哈希值
        
        Args:
            prev_hash: 之前的哈希值
            row, col: 落子位置
            player_id: 落子的玩家ID
            prev_player_id: 之前轮到的玩家ID
            next_player_id: 接下来轮到的玩家ID
        
        Returns:
            int: 更新后的哈希值
        """
        h = prev_hash
        
        # 如果之前这个位置不是空的，需要先异或掉原来的值
        if prev_player_id is not None:
            h ^= self.player_hash[prev_player_id]
        
        # 异或新的棋子
        h ^= self.zobrist_table[row][col][player_id]
        
        # 添加新的当前玩家
        if next_player_id is not None:
            h ^= self.player_hash[next_player_id]
        
        return h
