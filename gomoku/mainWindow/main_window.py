# coding:utf-8
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import Qt, QUrl, QTimer    # 添加 QTimer
from PyQt5.QtGui import QIcon, QDesktopServices, QColor
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSystemTrayIcon, QMenu, QAction

from qfluentwidgets import (NavigationBar, NavigationItemPosition, MessageBox,
                           isDarkTheme, FluentIcon as FIF, Theme, setTheme)
from qframelesswindow import FramelessWindow, TitleBar

# 改为使用相对导入
from .interfaces import Widget, HomeInterface, SettingInterface, HistoryInterface, BoardWidget, TrainingInterface, ParametersInterface, ParameterVisualizationInterface
from .config import cfg


class CustomTitleBar(TitleBar):
    """ Title bar with icon and title """

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.hBoxLayout.removeWidget(self.minBtn)
        self.hBoxLayout.removeWidget(self.maxBtn)
        self.hBoxLayout.removeWidget(self.closeBtn)

        # add window icon
        self.iconLabel = QLabel(self)
        self.iconLabel.setFixedSize(18, 18)
        self.hBoxLayout.insertSpacing(0, 20)
        self.hBoxLayout.insertWidget(
            1, self.iconLabel, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.window().windowIconChanged.connect(self.setIcon)

        # add title label
        self.titleLabel = QLabel(self)
        self.hBoxLayout.insertWidget(
            2, self.titleLabel, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.titleLabel.setObjectName('titleLabel')
        self.window().windowTitleChanged.connect(self.setTitle)

        # 添加一个伸缩器，将按钮推到右侧
        self.hBoxLayout.addStretch(1)

        # 重新添加窗口控制按钮到右侧
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setSpacing(0)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setAlignment(Qt.AlignTop)
        self.buttonLayout.addWidget(self.minBtn)
        self.buttonLayout.addWidget(self.maxBtn)
        self.buttonLayout.addWidget(self.closeBtn)
        self.hBoxLayout.addLayout(self.buttonLayout)

    def setTitle(self, title):
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setIcon(self, icon):
        self.iconLabel.setPixmap(QIcon(icon).pixmap(18, 18))


class Window(FramelessWindow):
    """主窗口 - 整合初始化逻辑"""
    
    def __init__(self):
        super().__init__()
        self.setTitleBar(CustomTitleBar(self))
        
        # 定义图标路径
        self.icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "211717_circle_icon.png")
        
        # 初始化UI组件
        self._init_ui_components()
        
        # 初始化导航栏
        self._init_navigation()
        
        # 初始化窗口属性
        self._init_window_properties()
        
        # 设置系统托盘
        self._setup_tray_icon()
        
        # 连接信号
        self._connect_signals()
    
    def _init_ui_components(self):
        """初始化UI组件"""
        self.hBoxLayout = QHBoxLayout(self)
        self.navigationBar = NavigationBar(self)
        self.stackWidget = QStackedWidget(self)
        
        # 创建子界面
        self.homeInterface = HomeInterface(self)
        self.appInterface = BoardWidget(self)
        self.historyInterface = HistoryInterface(self)
        self.settingInterface = SettingInterface(self)
        self.trainingInterface = TrainingInterface(self)
        self.parametersInterface = ParametersInterface(self)
        self.paramVisualizationInterface = ParameterVisualizationInterface(self)
        
        # 初始化布局
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, 48, 0, 0)
        self.hBoxLayout.addWidget(self.navigationBar)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setStretchFactor(self.stackWidget, 1)
    
    def _init_navigation(self):
        """初始化导航栏"""
        self.addSubInterface(self.homeInterface, FIF.HOME, '主页', selectedIcon=FIF.HOME_FILL)
        self.addSubInterface(self.appInterface, FIF.GAME, '五子棋游戏')
        self.addSubInterface(self.historyInterface, FIF.HISTORY, '历史对局')

        # 替换原本的库界面为设置界面
        self.addSubInterface(
            self.settingInterface, 
            FIF.SETTING, 
            '设置', 
            NavigationItemPosition.BOTTOM
        )
        
        self.addSubInterface(self.trainingInterface, FIF.ROBOT, '训练', NavigationItemPosition.BOTTOM)
        
        # 使用 PLAY 图标代替不存在的 TELEVISION
        self.addSubInterface(self.parametersInterface, FIF.SYNC, '实况', NavigationItemPosition.BOTTOM)
        
        # 修复: 将不存在的 CHART 图标改为 DATA_USAGE 图标
        self.addSubInterface(self.paramVisualizationInterface, FIF.PLAY, '参数', NavigationItemPosition.BOTTOM)
        
        # 帮助按钮
        self.navigationBar.addItem(
            routeKey='Help',
            icon=FIF.HELP,
            text='帮助',
            onClick=lambda: None,
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
        )

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.navigationBar.setCurrentItem(self.homeInterface.objectName())
    
    def _init_window_properties(self):
        """初始化窗口属性"""
        self.resize(1000, 800)
        # 使用自定义图标
        self.setWindowIcon(QIcon(self.icon_path))
        self.setWindowTitle('五子棋游戏')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        self.setQss()
    
    def _setup_tray_icon(self):
        """设置系统托盘图标 - 整合所有托盘相关功能"""
        # 创建托盘图标
        self.trayIcon = QSystemTrayIcon(self)
        self.trayIcon.setIcon(QIcon(self.icon_path))
        self.trayIcon.setToolTip('五子棋游戏')
        
        # 创建托盘菜单
        trayMenu = QMenu()
        
        # 添加菜单项
        actions = {
            '显示主窗口': self.showNormal,
            '退出': self.quitApplication
        }
        
        for text, slot in actions.items():
            action = QAction(text, self)
            action.triggered.connect(slot)
            trayMenu.addAction(action)
            if text == '显示主窗口':
                trayMenu.addSeparator()
        
        # 设置托盘菜单
        self.trayIcon.setContextMenu(trayMenu)
        
        # 托盘图标双击显示窗口
        self.trayIcon.activated.connect(self.onTrayIconActivated)
        
        # 根据配置决定是否启用托盘图标
        if cfg.get(cfg.minimizeToTray):
            self.trayIcon.show()
    
    def _connect_signals(self):
        """连接信号"""
        cfg.themeChanged.connect(self.onThemeChanged)
        self.settingInterface.minimizeToTrayChanged.connect(self.onMinimizeToTrayChanged)
        self.settingInterface.checkUpdateSig.connect(self.checkUpdate)

    def addSubInterface(self, interface, icon, text: str, position=NavigationItemPosition.TOP, selectedIcon=None):
        self.stackWidget.addWidget(interface)
        self.navigationBar.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            selectedIcon=selectedIcon,
            position=position,
        )

    def setQss(self):
        """加载并应用全局样式表 - 简化资源路径处理"""
        import os
        
        theme_folder = "dark" if isDarkTheme() else "light"
        resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
        
        # 通用样式路径
        common_qss_path = os.path.join(resource_path, f'qss/{theme_folder}/common.qss')
        demo_qss_path = os.path.join(resource_path, f'qss/{theme_folder}/demo.qss')
        
        # 组合样式表
        style_sheets = []
        
        # 尝试加载通用样式
        if os.path.exists(common_qss_path):
            with open(common_qss_path, encoding='utf-8') as f:
                style_sheets.append(f.read())
        
        # 尝试加载特定样式
        if os.path.exists(demo_qss_path):
            with open(demo_qss_path, encoding='utf-8') as f:
                style_sheets.append(f.read())
        
        # 应用样式
        if style_sheets:
            self.setStyleSheet("\n".join(style_sheets))
        else:
            print("警告: 未找到样式表文件")
            self.setStyleSheet("")
        
        # 更新子界面样式
        for interface in [self.homeInterface, self.appInterface, self.historyInterface, self.settingInterface, self.trainingInterface, self.parametersInterface, self.paramVisualizationInterface]:
            if hasattr(interface, 'setStyleSheet') and style_sheets:
                interface.setStyleSheet(style_sheets[0])  # 只应用通用样式

    def switchTo(self, widget):
        self.stackWidget.setCurrentWidget(widget)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationBar.setCurrentItem(widget.objectName())

    def onThemeChanged(self, theme):
        """响应主题变更，更新所有界面样式"""
        # 设置 QFluentWidgets 的主题
        setTheme(theme)
        
        # 更新主窗口样式
        self.setQss()
        
        # 更新各个子界面样式
        for interface in [self.homeInterface, self.appInterface, self.historyInterface, self.settingInterface, self.trainingInterface, self.parametersInterface, self.paramVisualizationInterface]:
            if hasattr(interface, 'updateStyle'):
                interface.updateStyle()
            
        # 刷新全部界面
        self.update()
        
        # 通知用户主题已更改
        from qfluentwidgets import InfoBar, InfoBarPosition
        InfoBar.success(
            title='主题已更改',
            content=f"已切换至{'深色' if isDarkTheme() else '浅色'}主题",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

    def enableAcrylicEffect(self):
        """启用模拟亚克力效果（不使用AcrylicBrush）"""
        try:
            # 根据当前主题选择合适的效果颜色
            if isDarkTheme():
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: rgba(32, 32, 32, 200);
                    }
                """)
            else:
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setStyleSheet("""
                    QMainWindow {
                        background-color: rgba(245, 245, 245, 220);
                    }
                """)
            
        except Exception as e:
            print(f"无法启用半透明效果: {e}")
            
    def disableAcrylicEffect(self):
        """禁用半透明效果"""
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        if isDarkTheme():
            self.setStyleSheet("background-color: rgb(32, 32, 32);")
        else:
            self.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.update()

    def showMessageBox(self):
        w = MessageBox(
            '支持作者🥰',
            '个人开发不易，如果这个项目帮助到了您，可以考虑请作者喝一瓶快乐水🥤。您的支持就是作者开发和维护项目的动力🚀',
            self
        )
        w.yesButton.setText('来啦老弟')
        w.cancelButton.setText('下次一定')

        if w.exec():
            QDesktopServices.openUrl(QUrl("https://afdian.net/a/zhiyiYo"))

    def onMinimizeToTrayChanged(self, enable):
        """响应最小化到托盘设置变更"""
        if enable and self.trayIcon:
            self.trayIcon.show()
        elif self.trayIcon:
            self.trayIcon.hide()

    def onTrayIconActivated(self, reason):
        """响应托盘图标激活事件"""
        if reason == QSystemTrayIcon.DoubleClick:
            self.showNormal()
            self.activateWindow()

    def quitApplication(self):
        """完全退出应用程序"""
        # 隐藏托盘图标，避免图标残留
        if self.trayIcon:
            self.trayIcon.hide()
        # 调用应用程序的quit方法
        QApplication.quit()

    def closeEvent(self, event):
        """重写关闭事件，实现最小化到托盘"""
        from .training_interface import stop_all_training_threads
        stop_all_training_threads()
        
        if cfg.get(cfg.minimizeToTray) and self.trayIcon and self.trayIcon.isVisible():
            event.ignore()  # 忽略关闭事件
            
            # 显示托盘提示
            self.trayIcon.showMessage(
                '五子棋游戏', 
                '程序已最小化到系统托盘，双击托盘图标可再次打开窗口', 
                QSystemTrayIcon.Information, 
                2000
            )
            
            # 隐藏主窗口
            self.hide()
        else:
            # 不使用托盘则正常关闭
            super().closeEvent(event)

    def checkUpdate(self):
        """检查更新"""
        # 这里可以实现真正的检查更新逻辑
        from qfluentwidgets import InfoBar, InfoBarPosition
        
        InfoBar.success(
            title="检查更新",
            content="当前已是最新版本",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

    def stop_training(self):
        """强化的训练停止方法"""
        from .training_interface import stop_all_training_threads
        
        # 停止所有活动的训练线程
        stop_all_training_threads()
        
        # 更新UI状态
        self.update_ui_training_stopped()
    
    def update_ui_training_stopped(self):
        """更新UI以反映训练已停止状态"""
        # 更新相关按钮状态
        if hasattr(self, 'start_training_button'):
            self.start_training_button.setEnabled(True)
        if hasattr(self, 'stop_training_button'):
            self.stop_training_button.setEnabled(False)
        
        # 其他UI更新...