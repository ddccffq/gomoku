# coding:utf-8
import os
from mainWindow.config import cfg, HELP_URL, FEEDBACK_URL, AUTHOR, VERSION, YEAR
from qfluentwidgets import (SettingCardGroup, SwitchSettingCard, 
                            OptionsSettingCard, PushSettingCard,
                            ScrollArea, ComboBoxSettingCard, ExpandLayout, 
                            Theme, InfoBar, InfoBarPosition, setTheme, isDarkTheme,
                            CustomColorSettingCard, HyperlinkCard, PrimaryPushSettingCard,
                            setThemeColor)
from qfluentwidgets import FluentIcon as FIF
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QPalette, QColor, QDesktopServices
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication

from mainWindow.game_history_manager import GameHistoryManager


class SettingInterface(ScrollArea):
    """ 设置界面 """

    minimizeToTrayChanged = pyqtSignal(bool)
    historyDirChanged = pyqtSignal(str)  # 历史目录改变信号
    checkUpdateSig = pyqtSignal()  # 检查更新信号

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # 初始化历史记录管理器
        self.history_manager = GameHistoryManager()

        # 游戏设置组
        self.gameSettingsGroup = SettingCardGroup("游戏设置", self.scrollWidget)
        self.historyDirCard = PushSettingCard(
            "选择文件夹",
            FIF.FOLDER,
            "历史记录保存路径",
            self.history_manager.history_dir,
            self.gameSettingsGroup
        )

        # 个性化组
        self.personalGroup = SettingCardGroup("个性化", self.scrollWidget)
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            "应用主题",
            "调整你的应用外观",
            texts=[
                "浅色", "深色", "跟随系统设置"
            ],
            parent=self.personalGroup
        )
        
        # 添加主题颜色卡片
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor if hasattr(cfg, 'themeColor') else None,
            FIF.PALETTE,
            "主题颜色",
            "改变应用的主题颜色",
            self.personalGroup
        )
        
        # 移除语言选择卡片

        # 主面板组
        self.mainPanelGroup = SettingCardGroup("主面板", self.scrollWidget)
        self.minimizeToTrayCard = SwitchSettingCard(
            FIF.MINIMIZE,
            "关闭后最小化到托盘",
            "应用将在后台继续运行",
            configItem=cfg.minimizeToTray,
            parent=self.mainPanelGroup
        )
        
        # 添加软件更新组
        self.updateSoftwareGroup = SettingCardGroup("软件更新", self.scrollWidget)
        self.updateOnStartUpCard = SwitchSettingCard(
            FIF.UPDATE,
            "在应用程序启动时检查更新",
            "新版本将更加稳定并拥有更多功能",
            configItem=cfg.checkUpdateAtStartUp,
            parent=self.updateSoftwareGroup
        )
        
        # 添加关于组
        self.aboutGroup = SettingCardGroup("关于", self.scrollWidget)
        self.helpCard = HyperlinkCard(
            HELP_URL,
            "打开帮助页面",
            FIF.HELP,
            "帮助",
            "了解五子棋游戏的功能和使用技巧",
            self.aboutGroup
        )
        self.feedbackCard = PrimaryPushSettingCard(
            "提供反馈",
            FIF.FEEDBACK,
            "提供反馈",
            "通过提供反馈帮助我们改进软件",
            self.aboutGroup
        )
        self.aboutCard = PrimaryPushSettingCard(
            "检查更新",
            FIF.INFO,
            "关于",
            '© ' + "版权所有" + f" {YEAR}, {AUTHOR}. " + "当前版本" + f" {VERSION}",
            self.aboutGroup
        )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)  # 将顶部边距从20改为0
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)

        # 初始化样式表
        self.__setQss()

        # 初始化布局
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # 添加卡片到组
        self.gameSettingsGroup.addSettingCard(self.historyDirCard)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)  # 添加主题颜色卡片
        # 移除语言卡片

        self.mainPanelGroup.addSettingCard(self.minimizeToTrayCard)
        
        # 添加更新卡片
        self.updateSoftwareGroup.addSettingCard(self.updateOnStartUpCard)
        
        # 添加关于卡片
        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

        # 添加设置卡片组到布局
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(60, 10, 60, 0)
        self.expandLayout.addWidget(self.gameSettingsGroup)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.mainPanelGroup)
        self.expandLayout.addWidget(self.updateSoftwareGroup)  # 添加软件更新组
        self.expandLayout.addWidget(self.aboutGroup)  # 添加关于组

    def __setQss(self):
        """ 设置样式表 """
        self.scrollWidget.setObjectName('scrollWidget')
        
        # 使用 Theme 枚举，而不是字符串
        theme = Theme.DARK if isDarkTheme() else Theme.LIGHT
        theme_folder = theme.value.lower()
        
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 使用绝对路径加载样式表
        qss_path = os.path.join(current_dir, f'resource/qss/{theme_folder}/setting_interface.qss')
        
        # 检查文件是否存在，如果存在则应用
        if os.path.exists(qss_path):
            with open(qss_path, encoding='utf-8') as f:
                specific_style = f.read()
                # 应用附加样式
                self.setStyleSheet(specific_style)

    def __showRestartTooltip(self):
        """ 显示重启提示 """
        InfoBar.warning(
            '',
            '配置将在重启后生效',
            parent=self.window()
        )

    def __onThemeChanged(self, theme: Theme):
        """ 主题变更处理 """
        # 更改qfluentwidgets的主题
        setTheme(theme)
        
        # 全局应用主题
        app = QApplication.instance()
        palette = app.palette()
        if theme == Theme.DARK:
            palette.setColor(QPalette.Window, QColor(32, 32, 32))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(42, 42, 42))
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(52, 52, 52))
            palette.setColor(QPalette.ButtonText, Qt.white)
        else:
            palette.setColor(QPalette.Window, Qt.white)
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, Qt.white)
            palette.setColor(QPalette.ButtonText, Qt.black)
        app.setPalette(palette)

    def __onHistoryDirClicked(self):
        """历史记录目录卡片点击事件"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择历史记录保存路径", 
            self.history_manager.history_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not directory:
            return
            
        # 尝试创建目录并更新设置
        try:
            os.makedirs(directory, exist_ok=True)
            self.history_manager.set_history_dir(directory)
            self.history_manager.save_settings()
            self.historyDirCard.setContent(directory)
            
            # 发出信号
            self.historyDirChanged.emit(directory)
            
            InfoBar.success(
                title='设置已更新',
                content="历史记录保存路径已更新",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
        except Exception as e:
            InfoBar.error(
                title='路径错误',
                content=f"无法设置目录: {str(e)}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def __connectSignalToSlot(self):
        """ 连接信号和槽 """
        cfg.appRestartSig.connect(self.__showRestartTooltip)
        cfg.themeChanged.connect(self.__onThemeChanged)

        # 历史记录目录
        self.historyDirCard.clicked.connect(self.__onHistoryDirClicked)

        # 主面板
        self.minimizeToTrayCard.checkedChanged.connect(self.minimizeToTrayChanged)
        
        # 主题颜色
        if hasattr(self, 'themeColorCard'):
            self.themeColorCard.colorChanged.connect(setThemeColor)
        
        # 移除语言更改处理

        # 关于卡片
        self.aboutCard.clicked.connect(self.checkUpdateSig)
        self.feedbackCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL)))
