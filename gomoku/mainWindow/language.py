# coding:utf-8
from enum import Enum
from typing import Dict, Optional
from PyQt5.QtCore import QObject, QLocale, QCoreApplication, QTranslator


class Language(Enum):
    """语言枚举"""
    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.SimplifiedChineseScript, QLocale.China)
    ENGLISH = QLocale(QLocale.English, QLocale.AnyScript, QLocale.UnitedStates)

    @classmethod
    def getLanguages(cls):
        """获取所有支持的语言"""
        return [cls.CHINESE_SIMPLIFIED, cls.ENGLISH]

    def __str__(self):
        if self == Language.CHINESE_SIMPLIFIED:
            return "简体中文"
        return "English"


class Translator(QObject):
    """翻译器单例"""
    _instance = None

    @classmethod
    def instance(cls) -> "Translator":
        """获取翻译器单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self.translator = QTranslator()
        self._translations = {
            # 黑白棋相关
            "黑棋": {"en_US": "Black"},
            "白棋": {"en_US": "White"},
            "玩家执黑": {"en_US": "Play as Black"},
            "玩家执白": {"en_US": "Play as White"},
            "vs": {"en_US": "vs"},
            
            # 游戏操作
            "开始对局": {"en_US": "Start Game"},
            "悔棋": {"en_US": "Undo Move"},
            "结束游戏": {"en_US": "End Game"},
            "保存对局": {"en_US": "Save Game"},
            "加载对局": {"en_US": "Load Game"},
            "重新开始": {"en_US": "Restart"},
            "退出": {"en_US": "Exit"},
            
            # 状态和结果
            "当前：黑棋": {"en_US": "Current: Black"},
            "当前：白棋": {"en_US": "Current: White"},
            "(玩家)": {"en_US": "(Player)"},
            "(AI)": {"en_US": "(AI)"},
            "游戏结束！": {"en_US": "Game Over!"},
            "游戏结束！胜者：黑棋": {"en_US": "Game Over! Winner: Black"},
            "游戏结束！胜者：白棋": {"en_US": "Game Over! Winner: White"},
            "胜者": {"en_US": "Winner"},
            "未结束": {"en_US": "In Progress"},
            
            # 界面文本
            "五子棋游戏": {"en_US": "Gomoku Game"},
            "AI 课程项目作品": {"en_US": "AI Course Project"},
            "游戏说明": {"en_US": "Game Instructions"},
            "历史对局": {"en_US": "Game History"},
            "棋盘风格：": {"en_US": "Board Style:"},
            "执棋方：": {"en_US": "Play as:"},
            "点击「开始对局」按钮开始游戏": {"en_US": "Click 'Start Game' button to begin"},
            "黑棋先行，双方轮流下棋": {"en_US": "Black plays first, take turns to place stones"},
            "先连成五子一线者获胜": {"en_US": "First to form five in a row wins"},
            "点击「悔棋」可撤销最后一步": {"en_US": "Click 'Undo Move' to revert last move"},
            "点击「结束游戏」可结束当前游戏": {"en_US": "Click 'End Game' to finish current game"},
            "游戏会自动保存到历史记录": {"en_US": "Game will be auto-saved to history"},
            
            # 设置和历史相关
            "设置": {"en_US": "Settings"},
            "经典木色": {"en_US": "Classic Wood"},
            "淡雅青色": {"en_US": "Elegant Cyan"},
            "复古黄褐": {"en_US": "Vintage Brown"},
            "冷酷灰色": {"en_US": "Cool Gray"},
            "暗黑模式": {"en_US": "Dark Mode"},
            "收藏/取消": {"en_US": "Toggle Favorite"},
            "删除记录": {"en_US": "Delete Record"},
            "刷新列表": {"en_US": "Refresh List"},
            "操作": {"en_US": "Actions"},
            "条记录": {"en_US": " Records"},
            "暂无历史对局记录": {"en_US": "No Game History Records"},
            "检测到历史记录变化，正在更新...": {"en_US": "Changes detected, updating..."},
            "历史记录已是最新": {"en_US": "History is up to date"},
            "已加载": {"en_US": "Loaded"},
            
            # 主页内容
            "软件简介": {"en_US": "Software Introduction"},
            "功能特点": {"en_US": "Features"},
            "使用指南": {"en_US": "User Guide"},
            "本软件是北京邮电大学人工智能课程项目作品，是一个基于 PyQt5 开发的五子棋游戏。游戏采用经典的 15×15 棋盘，支持人机对战、历史记录查询等功能。界面设计采用 Fluent Design 风格，美观易用，支持深浅色主题切换。": 
            {"en_US": "This software is an AI course project from Beijing University of Posts and Telecommunications, a Gomoku game developed with PyQt5. The game uses a classic 15×15 board, supports human vs AI gameplay, history tracking and more. The interface is designed with Fluent Design style, both beautiful and easy to use, with light/dark theme support."},
            "<b>· 经典五子棋玩法</b> — 黑白双方轮流落子，五子连珠获胜<br><b>· 多主题棋盘</b> — 提供多种棋盘风格和主题，满足不同审美需求<br><b>· 历史记录</b> — 自动保存每局游戏，随时查看或继续历史对局<br><b>· 操作便捷</b> — 支持悔棋、重开、认输等操作，操作简单直观<br><b>· 主题定制</b> — 支持浅色/深色主题，自定义主题颜色<br>": 
            {"en_US": "<b>· Classic Gomoku Gameplay</b> — Black and white take turns, five in a row wins<br><b>· Multiple Board Themes</b> — Various board styles to meet different aesthetic preferences<br><b>· History Recording</b> — Auto-saves every game, view or continue historical games anytime<br><b>· Convenient Operations</b> — Support undo, restart, surrender and more, intuitive controls<br><b>· Theme Customization</b> — Support light/dark theme and custom theme colors<br>"},
            "<b>1. 开始游戏</b> — 点击左侧导航栏的\"五子棋游戏\"进入游戏界面<br><b>2. 对弈操作</b> — 点击\"开始对局\"按钮，黑棋先行，点击棋盘落子<br><b>3. 历史查询</b> — 在\"历史对局\"页面可查看、筛选和加载历史棋局<br><b>4. 个性设置</b> — 在\"设置\"页面调整主题、语言等个性化选项<br><b>5. 更多帮助</b> — 点击\"设置\"页面中的\"帮助\"链接获取详细指南<br>": 
            {"en_US": "<b>1. Start Game</b> — Click \"Gomoku Game\" in left sidebar to enter game interface<br><b>2. Game Operation</b> — Click \"Start Game\" button, black goes first, click on board to place stones<br><b>3. History Query</b> — View, filter and load history games in \"Game History\" page<br><b>4. Personalization</b> — Adjust theme and other options in \"Settings\" page<br><b>5. More Help</b> — Click \"Help\" link in \"Settings\" page for detailed guide<br>"},
            "© 2023 北京邮电大学 AI课程五子棋小组 - 版本 1.0.0": {"en_US": "© 2023 BUPT AI Course Gomoku Team - Version 1.0.0"}
        }
        self.current_locale = QLocale(QLocale.Chinese, QLocale.SimplifiedChineseScript, QLocale.China)
    
    def setLanguage(self, language: Language):
        """设置当前语言"""
        self.current_locale = language.value
        # 这里可以加载实际的翻译文件
    
    def translate(self, source: str) -> str:
        """翻译文本"""
        # 如果是中文，直接返回源文本
        if self.current_locale.name().startswith("zh_"):
            return source
            
        # 否则查找翻译表
        translations = self._translations.get(source, {})
        locale_name = self.current_locale.name()
        if locale_name in translations:
            return translations[locale_name]
        
        # 如果没有精确匹配，尝试使用语言代码匹配
        lang_code = locale_name.split('_')[0]
        for key, value in translations.items():
            if key.startswith(f"{lang_code}_"):
                return value
        
        # 如果没有找到翻译，返回原文本
        return source
