def undo_move(self):
    """悔棋功能 - 确保悔棋后总是用户的回合"""
    if not self.move_history:
        return False  # 无法悔棋，没有历史记录
        
    # 记录当前局面位置，悔棋可能需要撤销多步
    original_position = len(self.move_history)
    
    # 首先撤销一步
    self._undo_last_move()
    
    # 检查是否轮到AI(玩家2)走子，如果是则继续悔棋
    # 假设玩家1是用户，玩家2是AI
    while self.current_player == 2 and self.move_history:
        self._undo_last_move()
        
    # 确保悔棋后是用户的回合
    if self.current_player != 1 and self.move_history:
        # 如果不是用户回合但还有历史记录，再悔一步
        self._undo_last_move()
        
    # 记录撤销了多少步
    steps_undone = original_position - len(self.move_history)
    
    # 更新UI显示
    self.update_status_message(f"已悔棋 {steps_undone} 步 - 轮到{'黑棋' if self.current_player == 1 else '白棋'}")
    
    # 刷新棋盘显示
    self.update()
    return True

def _undo_last_move(self):
    """撤销最后一步走子的内部实现"""
    if not self.move_history:
        return
        
    # 获取最后一步
    last_move = self.move_history.pop()
    row, col = last_move
    
    # 清除棋子
    self.board[row][col] = 0
    
    # 切换玩家
    self.current_player = 3 - self.current_player  # 1→2或2→1
    
    # 如果有悔棋事件，触发它
    if hasattr(self, 'move_undone'):
        self.move_undone.emit(row, col, self.current_player)
