# coding:utf-8
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import QIcon, QFont, QPainter, QPen, QBrush, QColor, QPaintEvent, QMouseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication, QSizePolicy, QFrame, QFileDialog, QMessageBox, QScrollArea
import sys
import os
import json
import datetime
import threading
import time
import torch  # 添加torch导入

from qfluentwidgets import FluentIcon as FIF, PushButton, ComboBox, isDarkTheme, InfoBar, InfoBarPosition, MessageBox, LineEdit
from qframelesswindow import FramelessWindow

from mainWindow.game_history_manager import GameHistoryManager
from ai.ai_factory import AIFactory
from ai.base_ai import AILevel, StoneColor

class GoBoardWidget(QWidget):
    """15x15的五子棋棋盘组件"""
    
    playerChanged = pyqtSignal(int)  # 当前玩家变更信号
    gameStatusChanged = pyqtSignal(bool, int)  # 游戏状态变更信号(是否结束，胜者ID)
    
    BOARD_STYLES = {
        "经典木色": {"background": QColor("#E8B473"), "line": QColor("#000000")},
        "淡雅青色": {"background": QColor("#B5D8CC"), "line": QColor("#000000")},
        "复古黄褐": {"background": QColor("#D4B483"), "line": QColor("#000000")},
        "冷酷灰色": {"background": QColor("#CCCCCC"), "line": QColor("#000000")},
        "暗黑模式": {"background": QColor("#2D2D2D"), "line": QColor("#FFFFFF")}
    }
    
    def __init__(self, parent=None, style_index=0):
        super().__init__(parent)
        self._init_board_properties(style_index)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
    
    def _init_board_properties(self, style_index):
        """初始化所有棋盘属性"""
        self.board_size = 15
        self.base_cell_size = 40
        self.base_padding = 25
        self.base_stone_size = 36
        
        style_names = list(self.BOARD_STYLES.keys())
        self.current_style = style_names[min(style_index, len(style_names)-1)]
        
        self.board_data = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.move_history = []
        self.forbidden_positions = []
        
        min_board_width = self.board_size * self.base_cell_size + 2 * self.base_padding
        self.setMinimumSize(min_board_width, min_board_width)
        
        self.current_player = 1
        self.game_started = False
        self.game_over = False
        self.winner = 0
        self.is_human_turn = True
        self.ai_thinking = False  # 添加AI思考标志
    
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
    
    def paintEvent(self, event: QPaintEvent):
        """绘制棋盘和棋子"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        
        # 计算当前棋盘规格
        w, h = self.width(), self.height()
        min_size = min(w, h)
        
        # 调整单元格尺寸和边距
        cell_size = (min_size - 2 * self.base_padding) / (self.board_size - 1)
        padding = self.base_padding
        stone_size = min(cell_size * 0.9, self.base_stone_size)
        
        # 获取当前风格
        style = self.BOARD_STYLES.get(self.current_style)
        bg_color = style["background"] if style else QColor("#E8B473")
        line_color = style["line"] if style else QColor("#000000")
        
        # 计算棋盘线条区域的实际大小
        grid_size = (self.board_size - 1) * cell_size
        
        # 绘制棋盘背景 - 只绘制棋盘线条覆盖的区域
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        
        # 创建精确匹配棋盘线条区域的背景矩形
        board_rect = QRectF(
            padding, 
            padding,
            grid_size,
            grid_size
        )
        painter.drawRect(board_rect)
        
        # 绘制棋盘线条
        painter.setPen(QPen(line_color, 1))
        
        # 绘制横线，确保使用整数坐标
        for i in range(self.board_size):
            y = int(padding + i * cell_size)  # 转换为整数
            painter.drawLine(
                QPoint(int(padding), y), 
                QPoint(int(padding + (self.board_size - 1) * cell_size), y)
            )
        
        # 绘制竖线，确保使用整数坐标
        for i in range(self.board_size):
            x = int(padding + i * cell_size)  # 转换为整数
            painter.drawLine(
                QPoint(x, int(padding)), 
                QPoint(x, int(padding + (self.board_size - 1) * cell_size))
            )
        
        # 绘制棋盘中央和星位标记点，确保使用整数坐标
        painter.setBrush(QBrush(line_color))
        star_points = [
            (3, 3), (3, 11), (7, 7),  # 左上, 左下, 中央
            (11, 3), (11, 11)         # 右上, 右下
        ]
        
        for x, y in star_points:
            point_x = int(padding + x * cell_size)  # 转换为整数
            point_y = int(padding + y * cell_size)  # 转换为整数
            painter.drawEllipse(QPoint(point_x, point_y), 3, 3)
        
        # 绘制棋子
        for y in range(self.board_size):
            for x in range(self.board_size):
                stone = self.board_data[y][x]
                if stone == 0:
                    continue
                    
                # 棋子坐标计算，确保使用整数坐标
                pos_x = int(padding + x * cell_size)  # 转换为整数
                pos_y = int(padding + y * cell_size)  # 转换为整数
                
                # 设置棋子颜色和边框
                if stone == 1:  # 黑棋
                    painter.setBrush(QBrush(Qt.black))
                    painter.setPen(QPen(QColor(60, 60, 60), 1))
                else:  # 白棋
                    painter.setBrush(QBrush(Qt.white))
                    painter.setPen(QPen(QColor(180, 180, 180), 1))
                
                # 绘制棋子
                stone_radius = int(stone_size / 2)  # 确保半径也是整数
                painter.drawEllipse(QPoint(pos_x, pos_y), stone_radius, stone_radius)
                
                # 标记最后一手棋
                if self.move_history and (x, y) == self.move_history[-1][1:]:
                    painter.setPen(QPen(Qt.red, 2))
                    marker_size = int(stone_size / 4)  # 转换为整数
                    painter.drawLine(
                        QPoint(pos_x - marker_size, pos_y - marker_size),
                        QPoint(pos_x + marker_size, pos_y + marker_size)
                    )
                    painter.drawLine(
                        QPoint(pos_x - marker_size, pos_y + marker_size),
                        QPoint(pos_x + marker_size, pos_y - marker_size)
                    )
    
    def mousePressEvent(self, event: QMouseEvent):
        """处理鼠标点击事件"""
        if not self.game_started or self.game_over:
            return
        
        # 增强检查：明确验证AI思考状态
        if not self.is_human_turn or hasattr(self, 'ai_thinking') and self.ai_thinking:
            # 在AI回合点击棋盘时显示提示，并直接返回，不进行任何操作
            from qfluentwidgets import InfoBar, InfoBarPosition
            InfoBar.warning(
                title='AI回合',
                content="现在是AI思考时间，请等待AI落子",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self.window()
            )
            return
        
        # 计算棋盘参数
        w, h = self.width(), self.height()
        min_size = min(w, h)
        cell_size = (min_size - 2 * self.base_padding) / (self.board_size - 1)
        padding = self.base_padding
        
        # 获取鼠标位置
        pos = event.pos()
        
        # 转换为棋盘坐标
        board_x = round((pos.x() - padding) / cell_size)
        board_y = round((pos.y() - padding) / cell_size)
        
        # 检查是否在棋盘内
        if not (0 <= board_x < self.board_size and 0 <= board_y < self.board_size):
            return
            
        # 检查是否已经有棋子
        if self.board_data[board_y][board_x] != 0:
            return
            
        # 放置棋子
        self.place_stone(board_x, board_y)
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if not self.game_started or self.game_over:
            return
            
        # 检查当前是否是人类玩家回合，如果不是则拒绝落子
        if not self.is_human_turn:
            # 显示提示
            InfoBar.warning(
                title='提示',
                content='现在是AI思考时间，请等待AI落子',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
    
    def place_stone(self, x, y):
        """在指定位置放置棋子"""
        # 确保在有效范围内
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
            
        # 确保该位置为空
        if self.board_data[y][x] != 0:
            return False
            
        # 额外检查是否是人类回合 - 防止快速点击漏检
        if not self.is_human_turn and hasattr(self, 'ai_thinking') and self.ai_thinking:
            print("防止快速点击：检测到AI思考中，人类不能落子")
            from qfluentwidgets import InfoBar, InfoBarPosition
            InfoBar.warning(
                title='AI回合',
                content="AI正在思考，请等待AI落子",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=1500,
                parent=self.window()
            )
            return False
            
        # 放置棋子
        self.board_data[y][x] = self.current_player
        
        # 添加到历史记录
        self.move_history.append((self.current_player, x, y))
        
        # 检查胜负
        if self.check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
            self.gameStatusChanged.emit(True, self.winner)
        else:
            # 切换玩家
            self.current_player = 3 - self.current_player  # 1->2, 2->1
            self.playerChanged.emit(self.current_player)
            
        # 重绘棋盘
        self.update()
        return True
    
    def check_win(self, x, y):
        """检查是否获胜"""
        # 检查方向：横向、纵向、左上到右下、右上到左下
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.current_player
        
        for dx, dy in directions:
            count = 1  # 当前位置已有一个棋子
            
            # 正向检查
            tx, ty = x + dx, y + dy
            while 0 <= tx < self.board_size and 0 <= ty < self.board_size and self.board_data[ty][tx] == player:
                count += 1
                tx += dx
                ty += dy
            
            # 反向检查
            tx, ty = x - dx, y - dy
            while 0 <= tx < self.board_size and 0 <= ty < self.board_size and self.board_data[ty][tx] == player:
                count += 1
                tx -= dx
                ty -= dy
                
            # 判断是否连成5子
            if count >= 5:
                return True
                
        return False

class BoardWidget(QWidget):
    """五子棋游戏组件，可嵌入到应用界面中"""
    def __init__(self, parent=None, style_index=0):
        super().__init__(parent)
        self._init_model_components()  # 先创建模型选择布局，确保 model_layout 存在
        self._init_ui_components(style_index)
        self._init_ai_components()
        self._init_signals()
        self._init_game_state()
    
    def _init_ui_components(self, style_index):
        """初始化UI组件"""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.board = GoBoardWidget(self, style_index)
        self.left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_layout.addWidget(self.board, 1)
        
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)  # 创建right_layout
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        self.right_layout.setSpacing(15)
        self.right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.right_panel.setFixedWidth(400)
        
        # 先添加容器到主布局，然后再创建右侧面板内容
        self.main_layout.addWidget(self.left_container, 3)
        self.main_layout.addWidget(self.right_panel, 0)
        
        # 创建右侧面板内容
        self._create_right_panel_ui()
        
        self.setObjectName('App-Interface')
    
    def _create_right_panel_ui(self):
        """创建右侧面板上的UI元素"""
        self.title_label = QLabel("五子棋游戏")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = self.title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        
        self.style_label = QLabel("棋盘风格：")
        self.style_combo = ComboBox(self)
        self.style_combo.addItems(self.board.get_style_names())
        self.style_layout = QHBoxLayout()
        self.style_layout.addWidget(self.style_label)
        self.style_layout.addWidget(self.style_combo)
        self.style_layout.addStretch(1)
        
        self.side_label = QLabel("玩家执棋：", self)
        self.side_combo = ComboBox(self)
        self.side_combo.addItems(["执黑", "执白"])
        self.side_layout = QHBoxLayout()
        self.side_layout.addWidget(self.side_label)
        self.side_layout.addWidget(self.side_combo)
        self.side_layout.addStretch(1)
        
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        
        self.player_info = QLabel("当前玩家：黑棋")
        self.player_info.setAlignment(Qt.AlignCenter)
        info_font = self.player_info.font()
        info_font.setPointSize(16)
        self.player_info.setFont(info_font)
        
        self.game_instructions = self._create_game_instructions()
        
        self.button_layout = QVBoxLayout()
        self.start_button = PushButton("开始对局")
        self.undo_button = PushButton("悔棋")
        self.end_game_button = PushButton("结束游戏")
        
        for btn in [self.start_button, self.undo_button, self.end_game_button]:
            btn.setFixedHeight(40)
            self.button_layout.addWidget(btn)
            self.button_layout.addSpacing(10)
        
        self.right_layout.addWidget(self.title_label)
        self.right_layout.addSpacing(10)
        self.right_layout.addLayout(self.style_layout)
        self.right_layout.addLayout(self.side_layout)
        self.right_layout.addLayout(self.model_layout)
        self.right_layout.addWidget(self.separator)
        self.right_layout.addWidget(self.player_info)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.game_instructions)
        self.right_layout.addSpacing(20)
        self.right_layout.addLayout(self.button_layout)
        self.right_layout.addStretch(1)
    
    def _init_ai_components(self):
        """初始化AI相关组件"""
        self.ai = None
        self.ai_enabled = False
        self.ai_level = AILevel.EXPERT  # 默认使用专家级AI
        self.ai_thinking = False
        self.trained_model_path = None
        self.player_side = "black"
        self.is_human_turn = True
        
        # 尝试初始化AI工厂
        try:
            from ai.ai_factory import AIFactory
            self.ai_factory = AIFactory()
            self.ai_modules_available = True
            
        except ImportError as e:
            print(f"无法初始化AI组件: {e}")
            self.ai_modules_available = False
    
    def _init_model_components(self):
        """初始化模型选择组件"""
        self.model_label = QLabel("使用模型：", self)
        self.model_path_edit = LineEdit(self)
        self.model_path_edit.setPlaceholderText("请选择模型文件（.pth/.pt）")
        self.model_browse_btn = PushButton("选择文件", self, FIF.FOLDER)
        self.model_browse_btn.setFixedWidth(100)

        self.model_layout = QHBoxLayout()  # 初始化模型选择布局
        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_path_edit, 1)
        self.model_layout.addWidget(self.model_browse_btn)
        self.model_layout.addStretch(0)
    
    def _init_signals(self):
        """初始化信号连接"""
        self.start_button.clicked.connect(self.onStartGame)
        self.undo_button.clicked.connect(self.onUndoMove)
        self.end_game_button.clicked.connect(self.onEndGame)
        
        self.style_combo.currentIndexChanged.connect(self.change_board_style)
        self.side_combo.currentIndexChanged.connect(self.on_side_changed)
        self.model_browse_btn.clicked.connect(self.on_browse_model_file)
        self.model_path_edit.textChanged.connect(self.on_model_path_changed)
        
        self.board.playerChanged.connect(self.on_player_changed)
        self.board.gameStatusChanged.connect(self.on_game_status_changed)
    
    def _init_game_state(self):
        """初始化游戏状态"""
        self.update_player_info()
        self.board.game_started = False
        self.ai_modules_available = self.check_ai_modules()
        self.update_button_states()
    
    def _create_game_instructions(self):
        """创建游戏说明标签"""
        instructions = QLabel(
            "游戏说明：\n"
            "1. 点击「开始对局」按钮开始游戏\n"
            "2. 黑棋先行，双方轮流下棋\n"
            "3. 先连成五子一线者获胜\n"
            "4. 点击「悔棋」可撤销最后一步\n"
            "5. 点击「结束游戏」可结束当前游戏\n"
            "6. 游戏会自动保存到历史记录"
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignLeft)
        return instructions
    
    def change_board_style(self, index):
        """更改棋盘风格"""
        self.board.set_style(index)
        print(f"已切换棋盘风格为: {self.board.get_style_names()[index]}")
        InfoBar.success(
            title='样式已更改',
            content=f"已切换棋盘风格为: {self.board.get_style_names()[index]}",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def on_side_changed(self, index):
        """更改玩家执棋方"""
        self.player_side = "black" if index == 0 else "white"
        print(f"玩家选择执{self.player_side}棋")
        
        # 如果游戏已经开始，更新玩家回合状态
        if self.board.game_started and not self.board.game_over:
            # 根据当前回合和玩家选择更新是否是人类回合
            self.is_human_turn = (self.board.current_player == 1 and self.player_side == "black") or \
                               (self.board.current_player == 2 and self.player_side == "white")
            self.update_player_info()
            
            # 如果更改后轮到AI，触发AI落子
            if not self.is_human_turn:
                self.make_ai_move()
        
        InfoBar.success(
            title='执棋方已更改',
            content=f"您现在执{('黑棋' if self.player_side == "black" else "白棋")}",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def on_browse_model_file(self):
        """弹出文件选择对话框选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pth *.pt *.bin);;所有文件 (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.trained_model_path = file_path
            print(f"选择模型文件: {file_path}")

    def on_model_path_changed(self, path):
        """模型路径变更时的处理"""
        self.trained_model_path = path if path else None
        print(f"当前模型路径: {self.trained_model_path or '默认AI'}")
    
    def onStartGame(self):
        """开始游戏"""
        # 如果游戏已经开始，询问用户是否保存当前游戏后开始新游戏
        if self.board.game_started and not self.board.game_over:
            reply = QMessageBox.question(
                self, "开始新游戏",
                "当前游戏尚未结束，是否保存当前对局后开始新游戏？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Save:
                self.save_game_history()
        
        # 确保AI已初始化
        if not hasattr(self, 'ai') or self.ai is None:
            print("初始化AI组件...")
            try:
                # 创建AI实例
                if self.ai_modules_available:
                    self.ai = self.ai_factory.create_ai(AILevel.EXPERT)
                    self.ai_enabled = True
                    
                    # 显式加载模型
                    if hasattr(self.ai, 'set_model'):
                        success = self.ai.set_model(self.trained_model_path)
                        print(f"初始AI模型加载结果: {'成功' if success else '失败'}")
                else:
                    print("AI模块不可用，将使用随机落子")
            except Exception as e:
                print(f"初始化AI失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 显示错误提示
                InfoBar.error(
                    title='AI初始化失败',
                    content=f"无法初始化AI: {str(e)[:100]}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
        
        # 重置棋盘数据
        self.board.board_data = [[0 for _ in range(self.board.board_size)] for _ in range(self.board.board_size)]
        self.board.move_history = []
        self.board.current_player = 1  # 黑棋先行
        self.board.game_started = True
        self.board.game_over = False
        self.board.winner = 0
        
        # 更新玩家信息
        self.is_human_turn = True if self.player_side == "black" else False
        self.update_player_info()
        
        # 禁用执棋方和模型选择，但允许更换棋盘风格
        self.side_combo.setEnabled(False)
        self.model_path_edit.setEnabled(False)
        self.model_browse_btn.setEnabled(False)
        self.style_combo.setEnabled(True)
        
        # 更新按钮状态
        self.update_button_states()
        
        # 重绘棋盘
        self.board.update()
        
        # 如果AI先行，则让AI下棋
        if not self.is_human_turn:
            self.make_ai_move()
        
        print("游戏开始")
        InfoBar.success(
            title='游戏开始',
            content="新对局已开始，祝您游戏愉快！",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def onUndoMove(self):
        """悔棋"""
        if not self.board.game_started or self.board.game_over or not self.board.move_history:
            return
        
        # 如果正在等待AI，不能悔棋
        if self.ai_thinking:
            return
        
        # 获取最后一步
        last_move = self.board.move_history.pop()
        player, x, y = last_move
        
        # 清除该位置的棋子
        self.board.board_data[y][x] = 0
        
        # 切换当前玩家
        self.board.current_player = player
        
        # 更新玩家信息
        self.is_human_turn = (self.board.current_player == 1 and self.player_side == "black") or \
                            (self.board.current_player == 2 and self.player_side == "white")
        self.update_player_info()
        
        # 重绘棋盘
        self.board.update()
        
        print(f"已撤销最后一步: 玩家{player}在({x},{y})的落子")
        InfoBar.success(
            title='悔棋成功',
            content="已撤销最后一步棋",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def onEndGame(self):
        """结束游戏"""
        if not self.board.game_started:
            return
        
        # 询问用户是否保存对局 - 移除条件，始终询问保存
        reply = QMessageBox.question(
            self, "结束游戏",
            "是否保存当前对局?",
            QMessageBox.Save | QMessageBox.Discard,
            QMessageBox.Save
        )
        
        if reply == QMessageBox.Save:
            self.save_game_history()
        
        # 结束游戏
        self.board.game_started = False
        self.board.game_over = True
        
        # 启用执棋方和模型选择
        self.side_combo.setEnabled(True)
        self.model_path_edit.setEnabled(True)
        self.model_browse_btn.setEnabled(True)
        
        # 更新按钮状态
        self.update_button_states()
        
        print("游戏已结束")
        InfoBar.success(
            title='游戏结束',
            content="对局已结束",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def on_player_changed(self, player_id):
        """当前玩家变更处理"""
        # 根据当前玩家和执棋方更新是否是人类回合
        self.is_human_turn = (self.board.current_player == 1 and self.player_side == "black") or \
                           (self.board.current_player == 2 and self.player_side == "white")
        
        # 更新玩家信息显示
        self.update_player_info()
        
        # 更新悔棋按钮状态
        self.update_button_states()
        
        # 如果是AI回合，触发AI下棋
        if not self.is_human_turn and self.board.game_started and not self.board.game_over:
            # 立即设置AI思考状态标志
            self.ai_thinking = True
            # 设置棋盘组件的AI思考状态标志
            self.board.ai_thinking = True
            
            # 禁用悔棋按钮，防止AI思考时悔棋
            self.undo_button.setEnabled(False)
            
            # 使用信息提示告知玩家AI正在思考
            InfoBar.info(
                title='AI回合',
                content="AI正在思考中...",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=1000,
                parent=self
            )
            
            # 启动AI思考
            self.make_ai_move()

    def make_ai_move(self):
        """让AI下棋"""
        if not self.board.game_started or self.board.game_over or self.is_human_turn:
            return
            
        # 设置AI思考标志
        self.ai_thinking = True
        self.board.ai_thinking = True
        
        # 禁用所有交互按钮
        self.start_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        
        # 使用QTimer模拟AI思考
        QTimer.singleShot(500, self._process_ai_move)
    
    def _process_ai_move(self):
        """处理AI下棋"""
        try:
            # 如果游戏已结束，则不再进行落子
            if self.board.game_over:
                self.ai_thinking = False
                self.board.ai_thinking = False
                self.update_button_states()
                return
                
            # 确保AI实例已创建
            if not self.ai_enabled or not hasattr(self, 'ai') or self.ai is None:
                print("警告: AI未初始化，回退到随机落子")
                self._random_ai_move()
                return
                
            # 转换当前棋盘状态为AI可用格式
            board_state = self.board.board_data.copy()
            # 确定当前棋色(1=黑, 2=白)
            stone_color = StoneColor.BLACK if self.board.current_player == 1 else StoneColor.WHITE
                
            print(f"AI开始思考: 棋色={'黑' if stone_color == StoneColor.BLACK else '白'}")
            start_time = time.time()
            
            # 让AI思考落子位置
            try:
                # 调用AI接口获取落子位置
                position = self.ai.think(board_state, stone_color)
                
                # 如果AI返回了有效位置
                if position and len(position) == 2:
                    x, y = position
                    print(f"AI决定在({x},{y})落子，思考耗时: {time.time() - start_time:.2f}秒")
                    
                    if 0 <= x < self.board.board_size and 0 <= y < self.board.board_size and board_state[y][x] == 0:
                        # 在棋盘上落子
                        self.board.place_stone(x, y)
                    else:
                        print(f"警告: AI返回的落子位置({x},{y})无效，回退到随机落子")
                        self._random_ai_move()
                else:
                    print(f"警告: AI未返回有效位置: {position}，回退到随机落子")
                    self._random_ai_move()
                    
            except Exception as e:
                print(f"AI思考出错: {str(e)}")
                import traceback
                traceback.print_exc()
                self._random_ai_move()
                    
            # 设置下一回合为人类回合
            self.is_human_turn = True
            self.board.is_human_turn = True
            self.update_player_info()
                    
        except Exception as e:
            print(f"AI走棋处理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self._random_ai_move()
        finally:
            # 不管成功与否，都清除AI思考标志
            self.ai_thinking = False
            self.board.ai_thinking = False
            
            # 恢复按钮状态
            self.update_button_states()
    
    def _random_ai_move(self):
        """执行随机落子（当AI失败时的备选方案）"""
        # 模拟AI落子 - 随机选择一个空位
        import random
        empty_cells = []
        for y in range(self.board.board_size):
            for x in range(self.board.board_size):
                if self.board.board_data[y][x] == 0:
                    empty_cells.append((x, y))
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            print(f"执行随机落子: AI在({x},{y})落子")
            self.board.place_stone(x, y)
    
    def on_game_status_changed(self, is_game_over, winner_id):
        """游戏状态变更处理"""
        if is_game_over:
            # 确定胜者文本
            winner_text = "黑棋胜利" if winner_id == 1 else "白棋胜利" if winner_id == 2 else "平局"
            winner_side = "玩家" if ((winner_id == 1 and self.player_side == "black") or 
                                   (winner_id == 2 and self.player_side == "white")) else "AI"
            print(f"游戏结束: {winner_text}")
            
            # 显示胜利弹窗
            QTimer.singleShot(300, lambda: self.show_victory_dialog(winner_text, winner_side))
        
        # 更新按钮状态
        self.update_button_states()
    
    def show_victory_dialog(self, winner_text, winner_side):
        """显示胜利弹窗"""
        # 创建结果信息
        is_player_victory = winner_side == "玩家"
        title = "恭喜您获得胜利！" if is_player_victory else "很遗憾，您输了"
        
        if "平局" in winner_text:
            title = "棋逢对手，平局收场"
        
        # 创建消息框 - 使用自定义按钮文本和回调函数，而不是尝试添加自定义按钮
        msg_box = MessageBox(
            title,
            f"游戏结果：{winner_text}\n您可以点击\"结束游戏\"按钮并选择是否保存对局记录。\n或者点击下方的\"贡献到训练数据\"按钮将此对局添加到模型训练数据集。",
            self
        )
        
        # 设置按钮文本
        msg_box.yesButton.setText("结束游戏")
        msg_box.cancelButton.setText("继续观看棋盘")
        
        # 在消息文本下方添加贡献按钮区域
        button_widget = QWidget(msg_box)
        button_layout = QHBoxLayout(button_widget)
        
        contribute_button = PushButton("贡献到训练数据", button_widget)
        contribute_button.setIcon(FIF.DOWNLOAD)
        button_layout.addStretch(1)
        button_layout.addWidget(contribute_button)
        button_layout.addStretch(1)
        
        # 将贡献按钮区域添加到消息框的布局中
        main_layout = msg_box.layout()
        if main_layout:
            # 在用户实际可以看到的位置插入贡献按钮
            main_layout.insertWidget(main_layout.count() - 1, button_widget)
        
        # 连接贡献按钮的点击信号
        contribute_button.clicked.connect(lambda: 
            [self.contribute_game_to_training(), msg_box.close()])
        
        # 显示对话框并处理结果
        if msg_box.exec():
            # 用户点击了"结束游戏"按钮
            self.onEndGame()
    
    def contribute_game_to_training(self):
        """将当前对局贡献到训练数据中"""
        try:
            if not self.board.game_over:
                InfoBar.warning(
                    title='无法贡献',
                    content="只能贡献已结束的对局",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
                return
            
            # 确保目录存在 - 使用规范的路径
            app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            training_data_dir = os.path.join(app_root, 'trained_models', 'training_data')
            
            # 创建session目录（基于时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(training_data_dir, f"user_session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            # 创建game目录
            game_dir = os.path.join(session_dir, f"game_user_{timestamp}")
            os.makedirs(game_dir, exist_ok=True)
            
            # 转换游戏数据为训练格式
            states, policies, values = self._convert_game_to_training_data()
            
            # 保存为NumPy文件格式
            import numpy as np
            np.save(os.path.join(game_dir, "states.npy"), np.array(states))
            np.save(os.path.join(game_dir, "policies.npy"), np.array(policies))
            np.save(os.path.join(game_dir, "values.npy"), np.array(values))
            
            # 保存元数据
            metadata = {
                'timestamp': timestamp,
                'game_source': 'user_play',
                'winner': self.board.winner,
                'total_moves': len(self.board.move_history),
                'player_side': self.player_side,
                'states_shape': np.array(states).shape,
                'policies_shape': np.array(policies).shape,
                'values_shape': np.array(values).shape
            }
            
            with open(os.path.join(game_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, indent=2)
                
            # 使用规范化路径输出
            normalized_path = os.path.normpath(game_dir)
            print(f"已贡献对局到训练数据: {normalized_path}")
                
            InfoBar.success(
                title='贡献成功',
                content=f"感谢您的贡献！您的对局已保存到训练数据库",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            
            # 保存历史记录
            #self.save_game_history()
            
            # 结束游戏
            self.onEndGame()
        except Exception as e:
            print(f"贡献对局失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            InfoBar.error(
                title='贡献失败',
                content=f"保存对局数据失败: {str(e)}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
    
    def _convert_game_to_training_data(self):
        """将当前对局转换为训练数据格式
        
        Returns:
            tuple: (states, policies, values)
        """
        from ai.data_handler import board_to_tensor
        
        states = []
        policies = []
        values = []
        
        # 重新模拟整个对局过程
        board_size = 15
        board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        
        # 获取胜者
        winner = self.board.winner
        
        for i, move in enumerate(self.board.move_history):
            player, x, y = move
            
            # 当前状态
            current_board = [row[:] for row in board]  # 深拷贝
            
            # 将当前状态转换为模型输入格式
            state = board_to_tensor(current_board, player)
            
            # 创建策略向量 (one-hot)
            policy = [0.0] * (board_size * board_size)
            policy[y * board_size + x] = 1.0
            
            # 计算价值
            # 如果这个玩家最终赢了，价值为1；如果输了，价值为-1；平局为0
            if winner == 0:  # 平局
                value = 0.0
            elif winner == player:  # 这个玩家赢了
                value = 1.0
            else:  # 这个玩家输了
                value = -1.0
            
            # 添加到训练数据
            states.append(state)
            policies.append(policy)
            values.append(value)
            
            # 更新棋盘
            board[y][x] = player
        
        return states, policies, values
    
    def save_game_history(self):
        """保存游戏记录到历史记录"""
        try:
            # 准备游戏数据
            result = ""
            if self.board.winner == 1:
                result = "黑胜"
            elif self.board.winner == 2:
                result = "白胜"
            else:
                result = "平局"
                
            game_data = {
                'board_data': self.board.board_data,
                'move_history': self.board.move_history,
                'current_player': self.board.current_player,
                'game_over': self.board.game_over,
                'winner': self.board.winner,
                'result': result,
                'player_info': {
                    'player1': '玩家' if self.player_side == "black" else 'AI',
                    'player2': 'AI' if self.player_side == "black" else '玩家'
                },
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # 创建历史记录管理器
            history_manager = GameHistoryManager()
            
            # 生成符合格式的文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}-{result}.json"
            
            # 保存记录
            file_path = history_manager.save_game(game_data, filename)
            if file_path:
                # 转换为规范的路径格式，确保跨平台一致性
                normalized_path = os.path.normpath(file_path)
                print(f"游戏记录已保存: {normalized_path}")
                
                InfoBar.success(
                    title='保存成功',
                    content=f"游戏记录已保存",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            
        except Exception as e:
            print(f"保存游戏记录失败: {str(e)}")
            
            InfoBar.error(
                title='保存失败',
                content=f"保存游戏记录失败: {str(e)}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
    
    def update_player_info(self):
        """更新玩家信息"""
        current_player_text = "黑棋" if self.board.current_player == 1 else "白棋"
        current_side = "玩家" if self.is_human_turn else "AI"
        self.player_info.setText(f"当前：{current_player_text}({current_side})")
    
    def update_button_states(self):
        """更新按钮状态"""
        game_started = self.board.game_started
        game_over = self.board.game_over
        
        # 开始按钮总是可用，但文本会变化
        self.start_button.setEnabled(True)
        self.start_button.setText("重新开始" if game_started and not game_over else "开始对局")
        
        # 悔棋按钮只在游戏进行中且有历史记录时可用
        self.undo_button.setEnabled(game_started and not game_over and len(self.board.move_history) > 0)
        
        # 结束游戏按钮总是在游戏进行中可用，包括游戏结束后
        self.end_game_button.setEnabled(game_started)
    
    def check_ai_modules(self):
        """检查AI模块是否可用"""
        try:
            # 只检查是否可以导入AIFactory，不实际创建实例
            return True
        except ImportError:
            print("AI模块不可用")
            return False
    def load_game_data(self, game_data):
        """加载历史游戏数据"""
        try:
            # 结束当前游戏（如果有）
            self.board.game_started = False
            self.board.game_over = False
            
            # 获取棋盘数据和历史记录
            board_data = game_data.get('board_data', [])
            move_history = game_data.get('move_history', [])
            game_over = game_data.get('game_over', False)
            winner = game_data.get('winner', 0)
            current_player = game_data.get('current_player', 1)
            
            # 确保棋盘数据有效
            if not board_data or not isinstance(board_data, list) or len(board_data) != self.board.board_size:
                raise ValueError("无效的棋盘数据格式")
            
            # 加载棋盘数据
            self.board.board_data = board_data
            self.board.move_history = move_history
            self.board.game_over = game_over
            self.board.winner = winner
            self.board.current_player = current_player
            
            # 设置游戏状态
            self.board.game_started = True
            
            # 如果游戏已结束，则显示结果
            if game_over:
                winner_text = "黑棋胜利" if winner == 1 else "白棋胜利" if winner == 2 else "平局"
                InfoBar.success(
                    title='历史对局',
                    content=f"已加载历史对局，结果: {winner_text}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            else:
                # 如果游戏未结束，设置当前玩家
                player_info = game_data.get('player_info', {})
                if player_info:
                    # 根据历史记录设置玩家方
                    if player_info.get('player1') == '玩家':
                        self.player_side = "black"
                        self.side_combo.setCurrentIndex(0)
                    else:
                        self.player_side = "white"
                        self.side_combo.setCurrentIndex(1)
                
                # 更新是否为玩家回合
                self.is_human_turn = (self.board.current_player == 1 and self.player_side == "black") or \
                                   (self.board.current_player == 2 and self.player_side == "white")
                
                InfoBar.success(
                    title='历史对局',
                    content="已加载历史对局，游戏继续",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            
            # 禁用执棋方和模型选择
            self.side_combo.setEnabled(False)
            self.model_path_edit.setEnabled(False)
            self.model_browse_btn.setEnabled(False)
            
            # 更新玩家信息
            self.update_player_info()
            
            # 更新按钮状态
            self.update_button_states()
            
            # 重绘棋盘
            self.board.update()
            
            # 如果是AI回合且游戏未结束，触发AI落子
            if not self.is_human_turn and not game_over:
                self.make_ai_move()
            
            return True
            
        except Exception as e:
            print(f"加载游戏数据失败: {str(e)}")
            InfoBar.error(
                title='加载失败',
                content=f"无法加载历史对局: {str(e)}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return False


class BoardWindow(FramelessWindow):
    """五子棋游戏窗口"""
    def __init__(self, parent=None, style_index=0):
        super().__init__(parent)
        self.setWindowTitle("五子棋游戏")
        self.setWindowIcon(FIF.GAME.icon())
        self.setWindowFlags(Qt.Window)
        self.resize(1000, 800)
        screen = QApplication.desktop().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        self.central_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.board_widget = BoardWidget(self, style_index)
        self.board_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.board_widget, 1)
        self.button_layout = QHBoxLayout()
        self.close_button = PushButton("关闭窗口")
        self.close_button.setFixedWidth(150)
        self.close_button.clicked.connect(self.close)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.close_button)
        self.main_layout.addLayout(self.button_layout, 0)
        self.setLayout(QVBoxLayout(self))
        self.layout().setContentsMargins(0, 48, 0, 0)
        self.layout().addWidget(self.central_widget)
    
    def closeEvent(self, event):
        """窗口关闭时的清理工作"""
        print("游戏窗口正在关闭，清理资源...")
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoardWindow()
    window.show()
    sys.exit(app.exec_())
