# coding:utf-8
from PyQt5.QtWidgets import QWidget, QStackedWidget

# 改为使用相对导入
from .home_interface import HomeInterface
from .history_interface import HistoryInterface
from .board_view import BoardWidget, BoardWindow
from .setting_interface import SettingInterface
from .training_interface import TrainingInterface
from .parameters_interface import ParametersInterface  # 原参数界面（改为实况）
from .parameter_visualization import ParameterVisualizationInterface  # 新参数可视化界面

# 通用Widget类型
Widget = QWidget
StackedWidget = QStackedWidget

# 重新导出组件，使其可以通过interfaces模块访问
__all__ = [
    'Widget', 'StackedWidget',
    'HomeInterface', 'SettingInterface',
    'BoardWidget', 'BoardWindow',
    'HistoryInterface', 'TrainingInterface',
    'ParametersInterface', 'ParameterVisualizationInterface'
]
