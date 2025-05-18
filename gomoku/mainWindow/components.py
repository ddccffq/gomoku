# coding:utf-8
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QStyleOptionViewItem

from qfluentwidgets import TableItemDelegate, isDarkTheme


class CustomTableItemDelegate(TableItemDelegate):
    """ Custom table item delegate """

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex):
        super().initStyleOption(option, index)
        if index.column() != 1:
            return

        if isDarkTheme():
            option.palette.setColor(QPalette.Text, Qt.white)
            option.palette.setColor(QPalette.HighlightedText, Qt.white)
        else:
            option.palette.setColor(QPalette.Text, Qt.red)
            option.palette.setColor(QPalette.HighlightedText, Qt.red)
