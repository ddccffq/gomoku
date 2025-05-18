# coding:utf-8
import os
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer, QRectF, QPointF
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
                           QSizePolicy, QSplitter, QApplication, QStackedWidget)
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPen

from qfluentwidgets import (ScrollArea, CardWidget, BodyLabel, TitleLabel, SubtitleLabel,
                          PrimaryPushButton, PushButton, FluentIcon as FIF, 
                          TransparentToolButton, TabBar, isDarkTheme)

# 导入绘图库
import matplotlib
matplotlib.use('Qt5Agg')  # 将backend设置为Qt5Agg

# 配置中文字体支持
font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'fonts', 'msyh.ttf')
if os.path.exists(font_path):
    # 设置Matplotlib使用微软雅黑字体
    matplotlib.font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
else:
    # 如果找不到微软雅黑，尝试使用系统中已安装的字体
    try:
        # 查找一个支持中文的字体
        from matplotlib.font_manager import FontProperties
        fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist
                if ('SimSun' in f.name or 'SimHei' in f.name or 'Microsoft YaHei' in f.name)]
        if fonts:
            matplotlib.rcParams['font.family'] = fonts[0]
        print(f"使用系统字体: {fonts[0] if fonts else '无可用中文字体'}")
    except:
        print("WARNING: 未能找到合适的中文字体，图表中的中文可能无法正确显示")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# 从board_view导入棋盘组件
from .board_view import GoBoardWidget


class MatplotlibChart(QWidget):
    """使用Matplotlib的图表组件"""
    
    def __init__(self, parent=None, chart_title="", x_label="", y_label=""):
        super().__init__(parent)
        
        # 设置布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图表标题
        if chart_title:
            self.title_label = QLabel(chart_title, self)
            self.title_label.setAlignment(Qt.AlignCenter)
            title_font = QFont()
            title_font.setPointSize(11)
            title_font.setBold(True)
            self.title_label.setFont(title_font)
            self.layout.addWidget(self.title_label)
        
        # 创建Matplotlib图表
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='none')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")
        
        # 添加子图
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        
        # 设置主网格线
        self.ax.grid(which='major', color='#666666', linestyle='-', alpha=0.2)
        
        # 设置图表
        self.figure.tight_layout(pad=2.0)
        self.layout.addWidget(self.canvas)
        
        # 初始化数据点
        self.x_data = []
        self.y_data = []
        self.line = None
        
        # 设置绘图颜色
        self.update_theme()
    
    def update_theme(self):
        """根据当前主题更新图表颜色"""
        # 获取当前是否是深色主题
        if isDarkTheme():
            text_color = 'white'
            line_color = '#3a7ebf'  # 蓝色在深色主题下
        else:
            text_color = 'black'
            line_color = '#1e70eb'  # 蓝色在浅色主题下
        
        # 设置轴标签和刻度颜色
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.tick_params(axis='x', colors=text_color)
        self.ax.tick_params(axis='y', colors=text_color)
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)
        
        # 更新已有的线条颜色
        if self.line:
            self.line.set_color(line_color)
        
        # 存储颜色设置以供新线条使用
        self.line_color = line_color
        
        # 重绘
        self.canvas.draw_idle()
    
    def update_data(self, x_data, y_data, redraw=True):
        """更新图表数据"""
        self.x_data = x_data
        self.y_data = y_data
        
        # 清除旧图并绘制新图
        self.ax.clear()
        
        if x_data and y_data:
            # 绘制新线条
            self.line, = self.ax.plot(x_data, y_data, '-o', color=self.line_color)
            
            # 更新坐标轴范围，给一定的边界
            y_min = min(y_data) * 0.95 if min(y_data) > 0 else min(y_data) * 1.05
            y_max = max(y_data) * 1.05 if max(y_data) > 0 else max(y_data) * 0.95
            self.ax.set_ylim(y_min, y_max)
            
            # 设置适当的x轴刻度间隔
            if len(x_data) > 10:
                step = len(x_data) // 10
                self.ax.set_xticks(x_data[::step])
        
        # 重新设置网格线
        self.ax.grid(which='major', color='#666666', linestyle='-', alpha=0.2)
        
        if redraw:
            self.canvas.draw_idle()


class BoardMonitor(QWidget):
    """棋盘监控器，用于显示训练过程中的棋盘状态"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 棋盘数据
        self.board_data = [[0 for _ in range(15)] for _ in range(15)]
        self.last_move = (-1, -1)  # 最后一步的位置
        self.current_player = 1
        
        # 设置最小尺寸
        self.setMinimumSize(300, 300)
        
        # 设置鼠标跟踪
        self.setMouseTracking(True)
    
    def setData(self, board_data, moves, current_player):
        """设置棋盘数据"""
        self.board_data = board_data
        self.current_player = current_player
        if moves and len(moves) > 0:
            self.last_move = moves[-1][:2]  # 取最后一步的行列坐标
        
        # 触发重绘
        self.update()
    
    def paintEvent(self, event):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置棋盘大小
        width, height = self.width(), self.height()
        board_size = min(width, height)
        
        # 计算单元格大小
        cell_size = board_size / (len(self.board_data) + 1)
        
        # 计算棋盘起始位置（保持棋盘居中）
        start_x = (width - board_size) / 2 + cell_size
        start_y = (height - board_size) / 2 + cell_size
        
        # 计算棋盘线条区域的实际大小
        grid_width = (len(self.board_data) - 1) * cell_size
        
        # 绘制棋盘背景 - 修正背景矩形区域与棋盘线条区域完全匹配
        background_rect = QRectF(
            start_x, 
            start_y,
            grid_width,
            grid_width
        )
        
        # 根据主题设置背景颜色
        if isDarkTheme():
            painter.setBrush(QBrush(QColor(45, 45, 48)))
        else:
            painter.setBrush(QBrush(QColor(255, 248, 220)))  # 米黄色背景
            
        painter.setPen(Qt.NoPen)
        painter.drawRect(background_rect)
        
        # 设置格子线颜色
        if isDarkTheme():
            painter.setPen(QPen(QColor(100, 100, 100), max(1, cell_size/25)))
        else:
            painter.setPen(QPen(QColor(0, 0, 0), max(1, cell_size/25)))
        
        # 绘制网格线（确保线宽随窗口大小缩放）
        line_width = max(1, int(cell_size / 25))
        grid_pen = QPen(painter.pen())
        grid_pen.setWidth(line_width)
        painter.setPen(grid_pen)
        
        # 绘制横线和竖线
        for i in range(len(self.board_data)):
            # 横线
            painter.drawLine(
                QPointF(start_x, start_y + i * cell_size),
                QPointF(start_x + (len(self.board_data) - 1) * cell_size, start_y + i * cell_size)
            )
            # 竖线
            painter.drawLine(
                QPointF(start_x + i * cell_size, start_y),
                QPointF(start_x + i * cell_size, start_y + (len(self.board_data) - 1) * cell_size)
            )
        
        # 绘制星位点
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for point in star_points:
            x, y = point
            star_size = max(3, int(cell_size / 10))
            painter.setBrush(QBrush(QColor(0, 0, 0) if not isDarkTheme() else QColor(200, 200, 200)))
            painter.drawEllipse(
                QPointF(start_x + x * cell_size, start_y + y * cell_size),
                star_size, star_size
            )
        
        # 绘制棋子
        for i in range(len(self.board_data)):
            for j in range(len(self.board_data[i])):
                if self.board_data[i][j] != 0:
                    # 设置棋子颜色
                    if self.board_data[i][j] == 1:  # 黑棋
                        painter.setBrush(QBrush(QColor(0, 0, 0)))
                    else:  # 白棋
                        painter.setBrush(QBrush(QColor(255, 255, 255)))
                    
                    # 增大棋子大小
                    stone_size = cell_size * 0.45
                    
                    # 绘制棋子阴影效果
                    shadow_size = stone_size * 1.02
                    shadow_offset = stone_size * 0.05
                    
                    # 阴影效果
                    shadow_color = QColor(20, 20, 20, 80)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(shadow_color))
                    painter.drawEllipse(
                        QPointF(start_x + j * cell_size + shadow_offset, start_y + i * cell_size + shadow_offset),
                        shadow_size, shadow_size
                    )
                    
                    # 棋子边框
                    border_size = stone_size * 1.01
                    border_color = QColor(0, 0, 0) if self.board_data[i][j] == 1 else QColor(220, 220, 220)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(border_color))
                    painter.drawEllipse(
                        QPointF(start_x + j * cell_size, start_y + i * cell_size),
                        border_size, border_size
                    )
                    
                    # 棋子本体
                    painter.setPen(Qt.NoPen)
                    if self.board_data[i][j] == 1:  # 黑棋
                        painter.setBrush(QBrush(QColor(0, 0, 0)))
                    else:  # 白棋
                        painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.drawEllipse(
                        QPointF(start_x + j * cell_size, start_y + i * cell_size),
                        stone_size, stone_size
                    )
                    
                    # 绘制高光
                    if self.board_data[i][j] == 2:  # 白棋添加高光
                        highlight_color = QColor(255, 255, 255, 160)
                        painter.setBrush(QBrush(highlight_color))
                        painter.drawEllipse(
                            QPointF(start_x + j * cell_size - stone_size * 0.2, start_y + i * cell_size - stone_size * 0.2),
                            stone_size * 0.4, stone_size * 0.4
                        )
                    
                    # 标记最后一步的位置
                    if i == self.last_move[0] and j == self.last_move[1]:
                        mark_size = stone_size * 0.3
                        # 最后一步标记颜色反转
                        if self.board_data[i][j] == 1:
                            painter.setBrush(QBrush(QColor(255, 255, 255)))
                        else:
                            painter.setBrush(QBrush(QColor(0, 0, 0)))
                        painter.drawEllipse(
                            QPointF(start_x + j * cell_size, start_y + i * cell_size),
                            mark_size, mark_size
                        )


class ParametersInterface(ScrollArea):
    """参数可视化界面"""
    boardSignal = pyqtSignal(list, list, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Parameters-Interface")

        # 滚动容器
        self.scroll_widget = QWidget(self)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 总布局
        self.main_layout = QVBoxLayout(self.scroll_widget)
        self.main_layout.setContentsMargins(20,20,20,20)
        self.main_layout.setSpacing(20)

        # 标题
        self.main_layout.addWidget(TitleLabel("训练参数监控", self))

        # 上半区：棋盘
        top = QHBoxLayout()
        self.main_layout.addLayout(top)

        # 棋盘卡片
        board_card = CardWidget(self)
        bl = QVBoxLayout(board_card)
        bl.addWidget(SubtitleLabel("训练棋局监控", self))
        self.board_widget = BoardMonitor(self)
        self.board_widget.setMinimumSize(600,600)  # 放大棋盘
        bl.addWidget(self.board_widget)
        self.game_info = BodyLabel("当前无对弈",self); bl.addWidget(self.game_info)
        self.game_status = BodyLabel("训练未开始",self); bl.addWidget(self.game_status)
        top.addWidget(board_card,1)  # 全宽展示棋盘

        # 连接信号
        self.boardSignal.connect(self._on_board_update)

    def _on_board_update(self, board, moves, current_player):
        # 更新棋盘
        if isinstance(board, np.ndarray):
            board = board.tolist()
        self.board_widget.setData(board, moves, current_player)
        # 更新文字
        self.game_info.setText(f"当前对弈: 回合 {len(moves)}")
        self.game_status.setText("游戏结束" if current_player==0 else f"当前: {'黑棋' if current_player==1 else '白棋'}")
        if current_player==0:
            QTimer.singleShot(1000, self._reset_board)

    def _reset_board(self):
        size = len(self.board_widget.board_data)
        self.board_widget.setData([[0]*size for _ in range(size)], [], 1)

    def update_parameters(self, x_data, y_data):
        """外部调用更新参数趋势图"""
        self.param_trend_chart.update_data(np.array(x_data), np.array(y_data))
