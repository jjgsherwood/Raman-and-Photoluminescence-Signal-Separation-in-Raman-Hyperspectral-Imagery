import os
import sys
import glob
from multiprocessing import Pool
from multiprocessing import Process as multiprocess


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *



OPENFILE = 0
OPENFILES = 1
OPENDIRECTORY = 2
SAVEFILE = 3

class FileBrowser(QWidget):
    def __init__(self, title, mode=OPENFILE, filter='All files (*.*)', dirpath=DEFAULT_DIR):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.browser_mode = mode
        self.filter = filter
        self.dirpath = dirpath
        self.filepaths = []

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial",weight=QFont.Bold))
        width = self.label.fontMetrics().boundingRect(self.label.text()).width()
        self.label.setFixedWidth(width+10)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.lineEdit = QPlainTextEdit(self)
        self.lineEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lineEdit.setFixedWidth(320)
        self.lineEdit.setFixedHeight(self.lineEdit.fontMetrics().boundingRect("TEST").height()+10)

        layout.addWidget(self.lineEdit)

        self.button = QPushButton('Search')
        self.button.clicked.connect(self.setFile)
        layout.addWidget(self.button)
        layout.addStretch()

    def clear(self):
        self.lineEdit.clear()
        self.lineEdit.setFixedHeight(self.lineEdit.fontMetrics().boundingRect("TEST").height()+10)

    def setWdith(self, width):
        self.label.setFixedWidth(width)

    def setMode(self, mode):
        """
        Set the mode, see class variables
        """
        self.mode = mode

    def setFileFilter(self, text):
        """
        regex string to blok certain files
        """
        self.filter = text

    def setDefaultDir(self, path):
        self.dirpath = path

    def setFile(self):
        if self.browser_mode == OPENFILE:
            caption = 'Choose File'
            func = QFileDialog.getOpenFileName
        elif self.browser_mode == OPENFILES:
            caption = 'Choose Files'
            func = QFileDialog.getOpenFileNames
        elif self.browser_mode == OPENDIRECTORY:
            caption='Choose Directory'
            # remove filter as keyword
            def func(*args, **kwargs): return [QFileDialog.getExistingDirectory(**{name: value for name,value in kwargs.items() if name != "filter"})], ""
        elif self.browser_mode == SAVEFILE:
            caption='Save/Save As'
            func = QFileDialog.getSaveFileName
        else:
            raise ValueError("FileBrowser only excepts modes Defined in Dialog (0,1,2,3)")

        self.filepaths, self.filter_out = func(self,
                                               caption=caption,
                                               directory=self.dirpath,
                                               filter=self.filter)

        # check if cancel was pressed
        if not self.filepaths:
            return

        # make it into a list
        if not isinstance(self.filepaths, list):
            self.filepaths = [self.filepaths]

        self.lineEdit.clear()
        self.lineEdit.appendPlainText("\n".join(self.filepaths))
        width = 0
        height = 10
        for file in self.filepaths:
            new_width = self.lineEdit.fontMetrics().boundingRect(file).width()
            height += self.lineEdit.fontMetrics().boundingRect(file).height()-1
            width = max(width, new_width)
        self.lineEdit.setFixedWidth(width+10)
        self.lineEdit.setFixedHeight(height)

    def getWidth(self):
        return self.label.fontMetrics().boundingRect(self.label.text()).width() + 10

    def getPaths(self):
        return self.filepaths

class FileBrowserEnableQtw(FileBrowser):
    def __init__(self, title, mode=OPENFILE, filter='All files (*.*)', dirpath=QDir.currentPath(), widget=None):
        super().__init__(title, mode=mode, filter=filter, dirpath=dirpath)

        if widget is None:
            raise ValueError("FileBrowserEnableQtw should be connected with an other widget")
        self.widget = widget

    def setFile(self):
        super().setFile()
        if self.filter_out == "numpy arrays (*.npy)":
            self.widget.setEnabled(True)
        else:
            self.widget.setEnabled(False)
