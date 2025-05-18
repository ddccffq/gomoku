# coding:utf-8
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QRectF
from PyQt5.QtGui import QIcon, QFont, QPainter, QPen, QBrush, QColor, QPaintEvent
from PyQt5.QtWidgets import QWidget, QSizePolicy
import datetime
import os
import json

from qfluentwidgets import InfoBar, InfoBarPosition

# 修改导入历史记录管理器
from mainWindow.game_history_manager import GameHistoryManager


class GoBoardWidget(QWidget):
    """15x15的五子棋棋盘组件"""
    
    # 添加玩家变更信号
    playerChanged = pyqtSignal(int)  # 当前玩家变更信号，参数为玩家ID(1为黑棋，2为白棋)
    gameStatusChanged = pyqtSignal(bool, int)  # 游戏状态变更信号(是否结束，胜者ID)
    
    # 棋盘样式 - 背景颜色
    BOARD_STYLES = {
        "经典木色": {"background": QColor("#E8B473"), "line": QColor("#000000")},
        "淡雅青色": {"background": QColor("#B5D8CC"), "line": QColor("#000000")},
        "复古黄褐": {"background": QColor("#D4B483"), "line": QColor("#000000")},
        "冷酷灰色": {"background": QColor("#CCCCCC"), "line": QColor("#000000")},
        "暗黑模式": {"background": QColor("#2D2D2D"), "line": QColor("#FFFFFF")}
    }
    
    def __init__(self, parent=None, style_index=0):
        super().__init__(parent)
        
        # 棋盘属性 - 增加基础尺寸
        self.board_size = 15  # 15x15的棋盘
        self.base_cell_size = 40  # 基础格子大小，实际大小会根据组件尺寸自动计算
        self.base_padding = 25  # 基础边距，实际边距会根据组件尺寸自动计算
        self.base_stone_size = 36  # 基础棋子大小，实际大小会根据组件尺寸自动计算
        
        # 获取样式名称列表
        style_names = list(self.BOARD_STYLES.keys())
        # 确保style_index在有效范围内
        self.current_style = style_names[min(style_index, len(style_names)-1)]
        
        # 棋盘数据 - 0表示空，1表示黑棋，2表示白棋
        self.board_data = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # 棋步记录
        self.move_history = []
        
        # 添加禁手位置列表
        self.forbidden_positions = []
        
        # 设置组件最小大小
        min_board_width = self.board_size * self.base_cell_size + 2 * self.base_padding
        self.setMinimumSize(min_board_width, min_board_width)
        
        # 设置大小策略为扩展，允许组件随窗口调整而放大
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 当前轮到谁下棋 - 1表示黑棋，2表示白棋
        self.current_player = 1
        
        # 游戏状态标志 - 只有游戏开始后才能下棋
        self.game_started = False
        self.game_over = False
        self.winner = 0  # 0表示无胜者，1表示黑棋胜，2表示白棋胜
        
        # 设置组件接受鼠标点击
        self.setMouseTracking(True)
    
    def set_style(self, style_index):
        """设置棋盘风格"""
        style_names = list(self.BOARD_STYLES.keys())
        if 0 <= style_index < len(style_names):
            self.current_style = style_names[style_index]
            self.update()  # 重绘棋盘
            return True
        return False
    
    def get_style_names(self):
        """获取所有棋盘风格名称"""
        return list(self.BOARD_STYLES.keys())
    
    def reset_game(self, start_immediately=True):
        """重置游戏状态"""
        self.board_data = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1  # 黑棋先行
        # 发出玩家变更信号
        self.playerChanged.emit(self.current_player)
        print(f"重置游戏时发出玩家变更信号：当前玩家 -> {self.current_player}")
        self.game_started = start_immediately  # 根据参数决定游戏是否立即开始
        self.move_history = []  # 清空历史记录
        self.game_over = False  # 游戏未结束
        self.winner = 0  # 无胜者
        
        # 清空禁手位置
        self.forbidden_positions = []
        
        # 如果开始是黑棋回合，检测禁手
        if start_immediately and self.current_player == 1:
            self.update_forbidden_positions()
            
        self.update()
    
    def update_forbidden_positions(self):
        """更新所有禁手位置"""
        self.forbidden_positions = []
        # 只在黑棋回合且游戏进行中检测禁手
        if self.current_player != 1 or not self.game_started or self.game_over:
            return
            
        # 检查所有空位
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board_data[row][col] == 0:  # 空位
                    if self.is_forbidden_move(row, col):
                        self.forbidden_positions.append((row, col))
    
    def undo_move(self):
        """悔棋 - 撤销最后一步"""
        # 修改：移除游戏结束时的限制，只要有历史记录就可以悔棋
        if not self.move_history:
            return False
        
        # 获取最后一步
        last_move = self.move_history.pop()
        
        # 清除该位置的棋子
        self.board_data[last_move[0]][last_move[1]] = 0
        
        # 切换回前一个玩家
        previous_player = self.current_player
        self.current_player = 3 - self.current_player
        
        # 发出玩家变更信号
        self.playerChanged.emit(self.current_player)
        print(f"悔棋时发出玩家变更信号：{previous_player} -> {self.current_player}")
        
        # 如果游戏已结束，则恢复为未结束状态
        if self.game_over:
            self.game_over = False
            self.winner = 0  # 清除胜者信息
            
        # 如果悔棋后是黑棋回合，更新禁手位置
        if self.current_player == 1:
            self.update_forbidden_positions()
            
        self.update()
        return True
    
    def surrender(self):
        """投降操作"""
        if self.game_started and not self.game_over:
            self.game_over = True
            self.winner = 3 - self.current_player  # 对方获胜
            self.update()
            return True
        return False
    
    def save_game(self, filename=None):
        """保存游戏状态"""
        # 获取当前时间戳
        timestamp = datetime.datetime.now().isoformat()
        
        game_data = {
            "board_data": self.board_data,
            "current_player": self.current_player,
            "game_started": self.game_started,
            "game_over": self.game_over,
            "move_history": self.move_history,
            "winner": self.winner,
            "timestamp": timestamp,  # 添加时间戳
            "style_index": list(self.BOARD_STYLES.keys()).index(self.current_style),  # 添加棋盘风格索引
            "player_info": {
                "player1": "玩家",  # 玩家1是人类
                "player2": "AI"     # 玩家2是AI
            }
        }
        
        # 导入历史记录管理器
        from mainWindow.game_history_manager import GameHistoryManager
        history_manager = GameHistoryManager()
        
        if filename is None:
            # 生成默认文件名
            winner_str = "黑胜" if self.winner == 1 else "白胜" if self.winner == 2 else "未结束"
            default_filename = f"对战_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{winner_str}.json"
            return default_filename, game_data
        
        # 保存到指定文件
        try:
            saved_path = history_manager.save_game(game_data, os.path.basename(filename))
            if saved_path:
                return saved_path, game_data
            return None, game_data
        except Exception as e:
            print(f"保存游戏失败: {str(e)}")
            return None, game_data
    
    def paintEvent(self, event: QPaintEvent):
        """绘制棋盘和棋子"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        
        # 计算当前实际的格子大小和边距
        # 使用较小的维度确保棋盘是正方形的
        size = min(self.width(), self.height())
        
        # 计算实际可用空间
        available_space = size - 2 * self.base_padding
        
        # 计算格子大小，确保能够整除
        cell_size = available_space / (self.board_size - 1)  # 修改：减1确保边缘线贴合
        
        # 计算实际棋盘的边距 - 居中对齐
        padding_x = (self.width() - (self.board_size - 1) * cell_size) / 2
        padding_y = (self.height() - (self.board_size - 1) * cell_size) / 2
        
        # 计算棋子大小
        stone_size = cell_size * 0.9  # 棋子大小为格子大小的90%
        
        # 获取当前风格
        style = self.BOARD_STYLES[self.current_style]
        
        # 计算棋盘线条区域的大小
        grid_size = (self.board_size - 1) * cell_size
        
        # 绘制棋盘背景 - 修正为只覆盖棋盘线条区域
        background_rect = QRectF(
            padding_x,
            padding_y,
            grid_size,
            grid_size
        )
        painter.fillRect(background_rect, QBrush(style["background"]))
        
        # 设置线条颜色和宽度
        line_width = max(1, int(cell_size / 15))
        painter.setPen(QPen(style["line"], line_width))
        
        # 绘制网格线 - 修改绘制方式确保线条贴合边界
        # 在每个循环中计算具体坐标，而不是使用公式
        
        # 绘制横线
        for i in range(self.board_size):
            y = int(padding_y + i * cell_size)
            # 横线从最左边到最右边
            painter.drawLine(
                int(padding_x), y,
                int(padding_x + (self.board_size - 1) * cell_size), y
            )
        
        # 绘制竖线
        for i in range(self.board_size):
            x = int(padding_x + i * cell_size)
            # 竖线从最上边到最下边
            painter.drawLine(
                x, int(padding_y),
                x, int(padding_y + (self.board_size - 1) * cell_size)
            )
        
        # 绘制天元和星位
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        star_size = max(4, int(cell_size / 8))  # 星位点大小随格子大小缩放
        
        for x, y in star_points:
            painter.setBrush(QBrush(style["line"]))
            painter.drawEllipse(
                int(padding_x + x * cell_size - star_size / 2),
                int(padding_y + y * cell_size - star_size / 2),
                star_size, star_size
            )
        
        # 绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board_data[row][col] != 0:
                    # 计算棋子位置 - 使用padding_x和padding_y
                    x = int(padding_x + col * cell_size - stone_size / 2)
                    y = int(padding_y + row * cell_size - stone_size / 2)
                    
                    # 设置棋子颜色 - 1是黑棋，2是白棋
                    if self.board_data[row][col] == 1:
                        painter.setBrush(QBrush(Qt.black))
                        text_color = Qt.white  # 黑棋上使用白色文字
                    else:
                        painter.setBrush(QBrush(Qt.white))
                        text_color = Qt.black  # 白棋上使用黑色文字
                    
                    # 绘制棋子边框，使用同样的线宽参数
                    painter.setPen(QPen(Qt.black if self.board_data[row][col] == 2 else Qt.gray, line_width))
                    
                    # 绘制棋子
                    painter.drawEllipse(x, y, int(stone_size), int(stone_size))
                    
                    # 查找该棋子的序号
                    move_number = self.find_move_number(row, col)
                    if move_number > 0:
                        # 设置序号文本字体和颜色
                        number_font = painter.font()
                        number_font.setPointSize(int(stone_size / 3))  # 字体大小约为棋子大小的1/3
                        number_font.setBold(True)
                        painter.setFont(number_font)
                        painter.setPen(QPen(text_color))
                        
                        # 绘制序号文本
                        number_rect = QRect(
                            x, y, int(stone_size), int(stone_size)
                        )  
                        painter.drawText(number_rect, Qt.AlignCenter, str(move_number))
        
        # 绘制禁手标记（如果有）
        if self.current_player == 1 and self.game_started and not self.game_over:
            for row, col in self.forbidden_positions:
                # 计算禁手标记位置
                x = int(padding_x + col * cell_size)
                y = int(padding_y + row * cell_size)
                
                # 设置禁手标记样式 - 红色粗线
                mark_size = int(cell_size * 0.4)  # 标记大小为格子大小的40%
                mark_pen = QPen(QColor(255, 0, 0), line_width * 1.5)
                painter.setPen(mark_pen)
                
                # 绘制X形标记
                painter.drawLine(
                    x - mark_size, y - mark_size,
                    x + mark_size, y + mark_size
                )
                painter.drawLine(
                    x + mark_size, y - mark_size,
                    x - mark_size, y + mark_size
                )
        
        # 如果游戏未开始，绘制提示
        if not self.game_started:
            font = painter.font()
            font.setPointSize(int(14 * cell_size / self.base_cell_size))  # 字体大小根据棋盘大小调整
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 0, 0, 180)))
            
            # 创建半透明背景
            rect = QRect(
                int(self.width() / 4),
                int(self.height() / 2 - size / 20),
                int(self.width() / 2),
                int(size / 10)
            )
            painter.fillRect(rect, QBrush(QColor(0, 0, 0, 120)))
            
            painter.drawText(rect, Qt.AlignCenter, "请点击「开始对局」按钮")
    
    def find_move_number(self, row, col):
        """查找指定位置棋子的序号"""
        for i, move in enumerate(self.move_history):
            if move[0] == row and move[1] == col:
                return i + 1  # 序号从1开始
        return 0  # 如果没找到（棋盘被直接设置而不是通过下棋），返回0

    def is_forbidden_move(self, row, col):
        """完整的黑棋禁手检测
        包括：三三禁手、四四禁手、长连禁手"""
        # 先在棋盘上模拟落子以便后续检测
        original_value = self.board_data[row][col]
        self.board_data[row][col] = 1  # 假设是黑子
        
        # 首先检查是否形成五连胜利
        is_winning_move = self.check_win_without_length_limit(row, col)
        
        # 长连禁手(超过5子连珠)
        long_connect = self.check_long_connect(row, col)
        
        # 三三禁手 - 检查是否形成两个以上的活三
        three_three = self.check_three_three(row, col)
        
        # 四四禁手 - 检查是否形成两个以上的四(活四或冲四)
        four_four = self.check_four_four(row, col)
        
        # 恢复棋盘状态
        self.board_data[row][col] = original_value
        
        # 如果是五连同时又是长连，优先判定为胜局而非禁手
        if is_winning_move and long_connect:
            return False
            
        # 其他情况下，如果触发任何禁手规则，则判定为禁手
        return long_connect or three_three or four_four

    def check_win_without_length_limit(self, row, col):
        """检查是否形成五连(不考虑长度限制)"""
        player = self.board_data[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、斜、反斜四个方向
        
        for dx, dy in directions:
            count = 1  # 当前落子点计为1
            
            # 沿着正方向检查连子
            for step in range(1, 5):  # 最多检查4步，加上当前位置刚好5子
                x, y = row + dx * step, col + dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == player:
                    count += 1
                else:
                    break
                    
            # 沿着反方向检查连子
            for step in range(1, 5):
                x, y = row - dx * step, col - dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == player:
                    count += 1
                else:
                    break
            
            # 正好5子连线则获胜
            if count >= 5:
                return True
                
        return False

    def check_three_three(self, row, col):
        """检查三三禁手(同时形成两个以上的活三)"""
        active_threes = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、正斜、反斜四个方向
        
        for dx, dy in directions:
            # 检查当前方向是否形成活三
            if self.is_active_three(row, col, dx, dy):
                active_threes += 1
                # 如果已经找到两个活三，可以提前返回结果
                if active_threes >= 2:
                    return True
        
        # 未形成两个及以上活三，不构成三三禁手
        return False

    def is_active_three(self, row, col, dx, dy):
        """检查指定方向是否形成活三
        活三：在一条线上有三个相连的棋子，并且两端都是空位，可以形成活四的情况"""
        pattern = self.get_line_pattern(row, col, dx, dy)
        
        # 活三的模式(. 表示空位，1表示黑棋，2表示白棋)
        active_three_patterns = [
            '...111..',   # 空空空黑黑黑空空 - 典型活三
            '..1.11..',   # 空空黑空黑黑空空 - 间隔活三
            '..11.1..',   # 空空黑黑空黑空空 - 间隔活三
        ]
        
        for p in active_three_patterns:
            if p in pattern:
                return True
        
        return False

    def check_four_four(self, row, col):
        """检查四四禁手(同时形成两个以上的四，包括活四和冲四)"""
        four_count = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、正斜、反斜四个方向
        
        # 检查每个方向上是否形成四(活四或冲四)
        for dx, dy in directions:
            is_active = self.is_active_four(row, col, dx, dy)
            is_blocked = self.is_blocked_four(row, col, dx, dy)
            
            if is_active or is_blocked:
                four_count += 1
                # 如果已经找到两个四，可以提前返回结果
                if four_count >= 2:
                    return True
        
        # 未形成两个及以上的四，不构成四四禁手
        return False

    def is_active_four(self, row, col, dx, dy):
        """检查指定方向是否形成活四
        活四：在一条线上有四个相连的棋子，一端是空位，下一步可以成五连胜利"""
        pattern = self.get_line_pattern(row, col, dx, dy)
        
        # 活四的模式
        active_four_patterns = [
            '..1111.',   # 空空黑黑黑黑空 - 活四
            '.1111..',   # 空黑黑黑黑空空 - 活四
        ]
        
        for p in active_four_patterns:
            if p in pattern:
                return True
        
        return False

    def is_blocked_four(self, row, col, dx, dy):
        """检查指定方向是否形成冲四
        冲四：在一条线上有四个相连的棋子，但被对方棋子或边界阻挡一端，只有一个方向可以成五连"""
        pattern = self.get_line_pattern(row, col, dx, dy)
        
        # 冲四的模式(2表示白棋或边界阻挡，.表示空位)
        blocked_four_patterns = [
            '.11112',    # 空黑黑黑黑白(或边界)
            '2.1111',    # 白(或边界)空黑黑黑黑
            '.1.111',    # 空黑空黑黑黑
            '.11.11',    # 空黑黑空黑黑
            '.111.1',    # 空黑黑黑空黑
            '1.111.',    # 黑空黑黑黑空
            '11.11.',    # 黑黑空黑黑空
            '111.1.',    # 黑黑黑空黑空
        ]
        
        for p in blocked_four_patterns:
            if p in pattern:
                return True
        
        return False

    def check_long_connect(self, row, col):
        """检查长连禁手(超过5子连珠)"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、正斜、反斜四个方向
        
        for dx, dy in directions:
            count = 1  # 当前位置计为1
            
            # 沿正方向检查连子
            for step in range(1, 6):
                x, y = row + dx * step, col + dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == 1:
                    count += 1
                else:
                    break
            
            # 沿反方向检查连子
            for step in range(1, 6):
                x, y = row - dx * step, col - dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == 1:
                    count += 1
                else:
                    break
            
            # 超过5个连子属于长连禁手
            if count > 5:
                return True
        
        return False

    def get_line_pattern(self, row, col, dx, dy):
        """获取指定方向的棋型模式
        返回一个字符串，'0'表示空位，'1'表示黑子，'2'表示白棋，'.'表示棋盘外"""
        pattern = []
        
        # 获取当前方向上的11格棋型(中心点+两边各5格)
        for step in range(-5, 6):
            x, y = row + dx * step, col + dy * step
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                pattern.append(str(self.board_data[x][y]))
            else:
                pattern.append('.')  # 棋盘外用.表示
        
        return ''.join(pattern)

    def check_win(self, row, col):
        """检查当前玩家是否获胜"""
        player = self.board_data[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、斜、反斜四个方向
        
        for dx, dy in directions:
            count = 1  # 当前落子点计为1
            
            # 沿着正方向检查连子
            for step in range(1, 5):  # 最多检查4步，加上当前位置刚好5子
                x, y = row + dx * step, col + dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == player:
                    count += 1
                else:
                    break
                    
            # 沿着反方向检查连子
            for step in range(1, 5):
                x, y = row - dx * step, col - dy * step
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board_data[x][y] == player:
                    count += 1
                else:
                    break
            
            # 正好5子连线则获胜，超过5子对白棋也算获胜，对黑棋则是禁手
            if count == 5 or (count > 5 and player == 2):
                return True
                
        return False

    def mousePressEvent(self, event):
        """处理鼠标点击事件，放置棋子"""
        if not self.game_started or self.game_over:
            return
        if event.button() != Qt.LeftButton:
            return
        
        # 添加调试输出，显示点击事件被触发
        print("棋盘点击事件被触发")
        
        # 计算格子大小和边距
        size = min(self.width(), self.height())
        available = size - 2 * self.base_padding
        cell_size = available / (self.board_size - 1)
        padding_x = (self.width() - (self.board_size - 1) * cell_size) / 2
        padding_y = (self.height() - (self.board_size - 1) * cell_size) / 2
        
        # 计算落子行列
        col = round((event.x() - padding_x) / cell_size)
        row = round((event.y() - padding_y) / cell_size)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return
        if self.board_data[row][col] != 0:
            return
        
        # 黑棋禁手检测
        if self.current_player == 1 and self.is_forbidden_move(row, col):
            InfoBar.warning(
                title='禁手',
                content='黑棋禁手，不允许落子',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 放置棋子并记录
        self.board_data[row][col] = self.current_player
        self.move_history.append((row, col))
        
        # 检查胜负
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            
            # 确保立即重绘棋盘，显示最后一步棋子
            self.repaint()
            
            # 通知父组件更新玩家信息 - 先于弹窗更新
            parent = self.parent()
            if parent and hasattr(parent, 'update_player_info'):
                parent.update_player_info()
                parent.repaint()
            
            # 显示胜利消息
            winner_text = "黑棋" if self.current_player == 1 else "白棋"
            InfoBar.success(
                title=f'{winner_text}胜利!',
                content=f"{winner_text}已经获胜！您可以悔棋或开始新游戏。",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            
            # 发出游戏状态变更信号
            self.gameStatusChanged.emit(True, self.winner)
            
            return
        
        # 切换玩家
        previous_player = self.current_player
        self.current_player = 3 - self.current_player
        
        # 发出玩家变更信号
        self.playerChanged.emit(self.current_player)
        print(f"发出玩家变更信号：{previous_player} -> {self.current_player}")
        
        # 如果轮到黑棋，更新禁手位置
        if self.current_player == 1:
            self.update_forbidden_positions()
        else:
            self.forbidden_positions = []  # 白棋回合清空禁手标记
        
        # 强制打印日志，确认每次下棋都会触发玩家信息更新
        print(f"落子成功，切换到玩家 {self.current_player}，开始更新玩家信息")
        
        # 重绘棋盘
        self.repaint()
