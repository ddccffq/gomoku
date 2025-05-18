# coding:utf-8
from enum import Enum

from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QGuiApplication, QFont
from qfluentwidgets import (qconfig, QConfig, ConfigItem, OptionsConfigItem, BoolValidator,
                            ColorConfigItem, OptionsValidator, RangeConfigItem, RangeValidator,
                            FolderListValidator, EnumSerializer, FolderValidator, ConfigSerializer, 
                            __version__, Theme)

# 从单独的模块导入Language
from mainWindow.language import Language


class LanguageSerializer(ConfigSerializer):
    """ 语言序列化器 """
    def serialize(self, language):
        return language.value.name()

    def deserialize(self, value: str):
        try:
            return Language(QLocale(value))
        except ValueError:
            return Language.CHINESE_SIMPLIFIED


class ThemeSerializer(ConfigSerializer):
    """ 主题序列化器 """
    def serialize(self, theme):
        """将Theme枚举序列化为字符串"""
        if theme == Theme.AUTO:
            return "Auto"
        elif theme == Theme.DARK:
            return "Dark"
        else:
            return "Light"

    def deserialize(self, value: str):
        """将字符串反序列化为Theme枚举"""
        if value == "Auto":
            return Theme.AUTO
        elif value == "Dark":
            return Theme.DARK
        else:
            return Theme.LIGHT


class Config(QConfig):
    """ 应用程序配置 """
    
    # 主窗口
    enableAcrylicBackground = ConfigItem(
        "MainWindow", "EnableAcrylicBackground", False, BoolValidator())
    minimizeToTray = ConfigItem(
        "MainWindow", "MinimizeToTray", True, BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)
    # 保留语言配置项但移除翻译功能
    language = OptionsConfigItem(
        "MainWindow", "Language", Language.CHINESE_SIMPLIFIED, OptionsValidator(Language), LanguageSerializer(), restart=True
    )
    
    # 主题模式 - 使用 Theme 枚举并添加序列化器
    themeMode = OptionsConfigItem(
        "Theme", "ThemeMode", Theme.LIGHT, 
        OptionsValidator([Theme.LIGHT, Theme.DARK, Theme.AUTO]),
        ThemeSerializer()  # 添加序列化器
    )
    
    # 添加主题颜色配置
    themeColor = ColorConfigItem(
        "Theme", "ThemeColor", "#0078d4"
    )
    
    # 游戏设置
    historyDir = ConfigItem(
        "Game", "HistoryDirectory", "game_history", FolderValidator())
    
    # 软件更新
    checkUpdateAtStartUp = ConfigItem(
        "Update", "CheckUpdateAtStartUp", True, BoolValidator())


# 应用元数据
YEAR = 2023
AUTHOR = "BUPT AI课程五子棋小组"
VERSION = "1.0.0"
HELP_URL = "https://bupt.edu.cn" 
FEEDBACK_URL = "https://github.com/your-username/Gomoku"
RELEASE_URL = "https://github.com/your-username/Gomoku/releases"

# 加载配置
cfg = Config()
qconfig.load('config/config.json', cfg)
