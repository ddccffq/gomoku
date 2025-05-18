# coding:utf-8
from PyQt5.QtWidgets import QHBoxLayout, QWidget


class AppInterface(QWidget):
    """ 应用界面 """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)
        self.setObjectName('Application-Interface')
