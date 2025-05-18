# coding:utf-8
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import SettingCardGroup, NumberSettingCard, SwitchSettingCard, FIF

class ConfigInterface(QWidget):
    """ ���ý��� - �����������Ҷ��Ĳ��� """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.scrollWidget = QWidget(self)
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.layout.addWidget(self.scrollWidget)

        self._setup_self_play_section()

    def _setup_self_play_section(self):
        """�������Ҷ��Ĳ�������"""
        self.self_play_group = SettingCardGroup(
            self.tr("���Ҷ��Ĳ���"), self.scrollWidget
        )

        # �������Ҷ��ľ���
        self.num_games_card = NumberSettingCard(
            FIF.GAME,
            self.tr("���ľ���"),
            self.tr("�������Ҷ��ĵ��ܾ���"),
            1, 100, 10, parent=self.self_play_group
        )

        # ����MCTSģ�����
        self.mcts_sim_card = NumberSettingCard(
            FIF.SEARCH,
            self.tr("MCTSģ�����"),
            self.tr("����ÿ�������ؿ���������ģ�����"),
            100, 2000, 800, parent=self.self_play_group
        )

        # ����̽���¶�
        self.exploration_temp_card = NumberSettingCard(
            FIF.TEMPERATURE,
            self.tr("̽���¶�"),
            self.tr("����̽���¶Ȳ�����ֵԽ��̽��Խ���"),
            0.1, 2.0, 1.0, parent=self.self_play_group
        )

        # ����Ƿ񱣴��������̵�ѡ��
        self.save_all_boards_switch = SwitchSettingCard(
            FIF.SAVE,
            self.tr("������������"),
            self.tr("���ú󽫱��������ÿһ��������ͼ�񣬽�����ֻ������������"),
            parent=self.self_play_group
        )
        self.save_all_boards_switch.setChecked(False)  # Ĭ�ϲ��������в���

        # ������п�Ƭ����
        self.self_play_group.addSettingCard(self.num_games_card)
        self.self_play_group.addSettingCard(self.mcts_sim_card)
        self.self_play_group.addSettingCard(self.exploration_temp_card)
        self.self_play_group.addSettingCard(self.save_all_boards_switch)

        self.scrollLayout.addWidget(self.self_play_group)

    def get_config(self):
        """��ȡ��ǰ����"""
        config = {}

        # ���Ҷ�������
        config['num_games'] = self.num_games_card.getValue()
        config['mcts_simulations'] = self.mcts_sim_card.getValue()
        config['exploration_temp'] = self.exploration_temp_card.getValue()
        config['save_all_boards'] = self.save_all_boards_switch.isChecked()

        return config