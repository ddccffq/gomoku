# coding:utf-8
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import SettingCardGroup, NumberSettingCard, SwitchSettingCard, FIF

class ConfigInterface(QWidget):
    """ 配置界面 - 用于设置自我对弈参数 """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.scrollWidget = QWidget(self)
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.layout.addWidget(self.scrollWidget)

        self._setup_self_play_section()

    def _setup_self_play_section(self):
        """设置自我对弈参数区域"""
        self.self_play_group = SettingCardGroup(
            self.tr("自我对弈参数"), self.scrollWidget
        )

        # 设置自我对弈局数
        self.num_games_card = NumberSettingCard(
            FIF.GAME,
            self.tr("对弈局数"),
            self.tr("设置自我对弈的总局数"),
            1, 100, 10, parent=self.self_play_group
        )

        # 设置MCTS模拟次数
        self.mcts_sim_card = NumberSettingCard(
            FIF.SEARCH,
            self.tr("MCTS模拟次数"),
            self.tr("设置每步的蒙特卡洛树搜索模拟次数"),
            100, 2000, 800, parent=self.self_play_group
        )

        # 设置探索温度
        self.exploration_temp_card = NumberSettingCard(
            FIF.TEMPERATURE,
            self.tr("探索温度"),
            self.tr("设置探索温度参数，值越高探索越随机"),
            0.1, 2.0, 1.0, parent=self.self_play_group
        )

        # 添加是否保存所有棋盘的选项
        self.save_all_boards_switch = SwitchSettingCard(
            FIF.SAVE,
            self.tr("保存所有棋盘"),
            self.tr("启用后将保存对弈中每一步的棋盘图像，禁用则只保存最终棋盘"),
            parent=self.self_play_group
        )
        self.save_all_boards_switch.setChecked(False)  # 默认不保存所有步骤

        # 添加所有卡片到组
        self.self_play_group.addSettingCard(self.num_games_card)
        self.self_play_group.addSettingCard(self.mcts_sim_card)
        self.self_play_group.addSettingCard(self.exploration_temp_card)
        self.self_play_group.addSettingCard(self.save_all_boards_switch)

        self.scrollLayout.addWidget(self.self_play_group)

    def get_config(self):
        """获取当前配置"""
        config = {}

        # 自我对弈配置
        config['num_games'] = self.num_games_card.getValue()
        config['mcts_simulations'] = self.mcts_sim_card.getValue()
        config['exploration_temp'] = self.exploration_temp_card.getValue()
        config['save_all_boards'] = self.save_all_boards_switch.isChecked()

        return config