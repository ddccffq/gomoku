# coding:utf-8
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage

from qfluentwidgets import (InfoBar, InfoBarPosition, FluentIcon as FIF, 
                           PushButton, ScrollArea, SubtitleLabel, CardWidget,
                           BodyLabel, TitleLabel, isDarkTheme)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import os
from datetime import datetime
import traceback


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布，用于绘制训练参数曲线"""
    
    def __init__(self, width=5, height=4, dpi=100):
        # 创建一个Figure对象
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 初始化FigureCanvas
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # 设置画布尺寸策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        # 配置图表样式
        self.update_style()
        
    def update_style(self):
        """更新图表样式以适应当前主题"""
        # 决定基于当前主题的背景和前景色
        if isDarkTheme():
            bg_color = '#2d2d2d'
            text_color = '#ffffff'
            grid_color = '#3a3a3a'
        else:
            bg_color = '#ffffff'
            text_color = '#000000'
            grid_color = '#e0e0e0'
        
        # 设置图表样式
        self.fig.patch.set_facecolor(bg_color)
        self.axes.set_facecolor(bg_color)
        
        # 设置轴标签和标题的颜色
        self.axes.xaxis.label.set_color(text_color)
        self.axes.yaxis.label.set_color(text_color)
        self.axes.title.set_color(text_color)
        
        # 设置刻度标签的颜色
        self.axes.tick_params(axis='x', colors=text_color)
        self.axes.tick_params(axis='y', colors=text_color)
        
        # 设置网格
        self.axes.grid(True, linestyle='--', linewidth=0.5, color=grid_color)
        
        # 设置脊柱的颜色
        for spine in self.axes.spines.values():
            spine.set_color(text_color)
            
        self.draw()


class ParameterVisualizationInterface(ScrollArea):
    """参数可视化界面，用于显示和保存训练过程中的精度和损失曲线"""
    
    # 定义信号，用于接收新的参数数据
    dataReceived = pyqtSignal(list, list, str)
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Parameter-Visualization-Interface")
        
        # 创建内容窗口
        self.view_widget = QWidget(self)
        self.setWidget(self.view_widget)
        self.setWidgetResizable(True)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.view_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        
        # 添加标题
        self.title_label = TitleLabel("参数可视化", self)
        self.main_layout.addWidget(self.title_label)
        
        # 创建卡片布局
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(15)
        
        # 创建精度卡片
        self.accuracy_card = self._create_parameter_card("精度曲线", "训练过程中的模型精度变化")
        self.accuracy_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        self.accuracy_layout.addWidget(self.accuracy_canvas)
        
        # 创建损失卡片
        self.loss_card = self._create_parameter_card("损失曲线", "训练过程中的模型损失变化")
        self.loss_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        self.loss_layout.addWidget(self.loss_canvas)
        
        # 添加卡片到水平布局
        self.cards_layout.addWidget(self.accuracy_card, 1)
        self.cards_layout.addWidget(self.loss_card, 1)
        
        # 添加卡片布局到主布局
        self.main_layout.addLayout(self.cards_layout, 1)
        
        # 创建操作按钮区域
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)
        
        # 添加保存按钮
        self.save_accuracy_button = PushButton("保存精度图", self, FIF.SAVE)
        self.save_loss_button = PushButton("保存损失图", self, FIF.SAVE)
        self.save_both_button = PushButton("保存全部", self, FIF.SAVE_AS)
        
        # 按钮点击处理
        self.save_accuracy_button.clicked.connect(lambda: self._save_figure(self.accuracy_canvas, "精度"))
        self.save_loss_button.clicked.connect(lambda: self._save_figure(self.loss_canvas, "损失"))
        self.save_both_button.clicked.connect(self._save_both_figures)
        
        # 添加按钮到布局
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.save_accuracy_button)
        self.buttons_layout.addWidget(self.save_loss_button)
        self.buttons_layout.addWidget(self.save_both_button)
        
        # 添加按钮布局到主布局
        self.main_layout.addLayout(self.buttons_layout)
        
        # 状态标签
        self.status_label = BodyLabel("等待训练数据...", self)
        self.main_layout.addWidget(self.status_label)
        
        # 存储数据
        self.epochs = []
        self.accuracy_values = []
        self.loss_values = []
        
        # 连接信号
        self.dataReceived.connect(self.update_parameters)
        
        # 定时刷新图表
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_charts)
        self.refresh_timer.start(5000)  # 5秒刷新一次
    
    def _create_parameter_card(self, title, description):
        """创建参数卡片"""
        card = CardWidget(self)
        
        # 创建卡片内部布局
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(15, 15, 15, 15)
        card_layout.setSpacing(10)
        
        # 添加标题
        card_title = SubtitleLabel(title, card)
        card_layout.addWidget(card_title)
        
        # 添加描述
        card_description = BodyLabel(description, card)
        card_layout.addWidget(card_description)
        
        # 添加内容区域 - 直接创建布局而不是使用动态属性
        content_layout = QVBoxLayout()
        card_layout.addLayout(content_layout, 1)
        
        # 根据title设置正确的属性
        if "精度" in title:
            self.accuracy_layout = content_layout
        elif "损失" in title:
            self.loss_layout = content_layout
        
        return card
    
    def update_parameters(self, epochs, values, param_type='accuracy'):
        """更新参数数据"""
        if not epochs:
            return
            
        # 更新数据
        self.epochs = epochs
        
        if param_type == 'accuracy':
            self.accuracy_values = values
        else:  # 'loss'
            self.loss_values = values
        
        # 根据数据类型更新相应图表
        if param_type == 'accuracy':
            self._update_accuracy_chart()
        else:
            self._update_loss_chart()
        
        # 更新状态
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_label.setText(f"上次更新: {last_update} | 当前轮次: {max(epochs) if epochs else 0}")
    
    def _update_accuracy_chart(self):
        """更新精度图表"""
        if not self.epochs or not self.accuracy_values:
            return
            
        # 清除当前图表
        self.accuracy_canvas.axes.clear()
        
        # 绘制新数据
        self.accuracy_canvas.axes.plot(self.epochs, self.accuracy_values, 'o-', color='#2196F3', label='训练精度')
        
        # 更新样式和标签
        self.accuracy_canvas.axes.set_xlabel('训练轮次')
        self.accuracy_canvas.axes.set_ylabel('精度')
        self.accuracy_canvas.axes.set_title('训练精度曲线')
        
        if len(self.epochs) > 1:
            # 使用平滑化趋势线
            try:
                x = np.array(self.epochs)
                y = np.array(self.accuracy_values)
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(x), max(x), 100)
                self.accuracy_canvas.axes.plot(x_smooth, p(x_smooth), '--', color='#FF5722', alpha=0.7, label='趋势')
                self.accuracy_canvas.axes.legend()
            except Exception:
                pass  # 忽略拟合错误
        
        # 刷新画布
        self.accuracy_canvas.update_style()
        self.accuracy_canvas.draw()
    
    def _update_loss_chart(self):
        """更新损失图表"""
        if not self.epochs or not self.loss_values:
            return
            
        # 清除当前图表
        self.loss_canvas.axes.clear()
        
        # 绘制新数据
        self.loss_canvas.axes.plot(self.epochs, self.loss_values, 'o-', color='#F44336', label='训练损失')
        
        # 更新样式和标签
        self.loss_canvas.axes.set_xlabel('训练轮次')
        self.loss_canvas.axes.set_ylabel('损失')
        self.loss_canvas.axes.set_title('训练损失曲线')
        
        if len(self.epochs) > 1:
            # 使用平滑化趋势线
            try:
                x = np.array(self.epochs)
                y = np.array(self.loss_values)
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(x), max(x), 100)
                self.loss_canvas.axes.plot(x_smooth, p(x_smooth), '--', color='#4CAF50', alpha=0.7, label='趋势')
                self.loss_canvas.axes.legend()
            except Exception:
                pass  # 忽略拟合错误
        
        # 刷新画布
        self.loss_canvas.update_style()
        self.loss_canvas.draw()
    
    def refresh_charts(self):
        """刷新所有图表"""
        # 刷新精度图
        if self.epochs and self.accuracy_values:
            self._update_accuracy_chart()
        
        # 刷新损失图
        if self.epochs and self.loss_values:
            self._update_loss_chart()
    
    def _save_figure(self, canvas, figure_type):
        """保存图表为图片"""
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"五子棋AI训练_{figure_type}曲线_{current_date}.png"
        
        # 使用项目目录而不是桌面
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        charts_dir = os.path.join(project_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        default_path = os.path.join(charts_dir, default_name)
        
        # 弹出文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"保存{figure_type}图表", 
            default_path,
            "图片文件 (*.png);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 保存图表
                canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                
                # 显示成功消息
                InfoBar.success(
                    title="保存成功",
                    content=f"{figure_type}图表已保存至 {file_path}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            except Exception as e:
                # 显示错误消息
                InfoBar.error(
                    title="保存失败",
                    content=f"保存图表时出错: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
                traceback.print_exc()
    
    def _save_both_figures(self):
        """保存两个图表"""
        # 创建保存目录
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用项目目录而不是桌面
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        charts_dir = os.path.join(project_dir, "charts")
        default_dir = os.path.join(charts_dir, f"五子棋AI训练参数_{current_date}")
        
        # 弹出目录选择对话框
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录", charts_dir)
        
        if directory:
            try:
                # 确保目录存在
                os.makedirs(os.path.join(directory, "五子棋训练参数"), exist_ok=True)
                save_dir = os.path.join(directory, "五子棋训练参数")
                
                # 保存精度图表
                accuracy_path = os.path.join(save_dir, f"精度曲线_{current_date}.png")
                self.accuracy_canvas.fig.savefig(accuracy_path, dpi=300, bbox_inches='tight')
                
                # 保存损失图表
                loss_path = os.path.join(save_dir, f"损失曲线_{current_date}.png")
                self.loss_canvas.fig.savefig(loss_path, dpi=300, bbox_inches='tight')
                
                # 显示成功消息
                InfoBar.success(
                    title="保存成功",
                    content=f"所有图表已保存至 {save_dir}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            except Exception as e:
                # 显示错误消息
                InfoBar.error(
                    title="保存失败",
                    content=f"保存图表时出错: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
                traceback.print_exc()
    
    def showEvent(self, event):
        """当界面显示时，刷新图表"""
        super().showEvent(event)
        self.refresh_charts()
