from StartUp import *

OPENFILE = 0
OPENFILES = 1
OPENDIRECTORY = 2
SAVEFILE = 3

class FileBrowser(QWidget):
    def __init__(self, title, mode=OPENFILE, filter='All files (*.*)', dirpath=DEFAULT_DIR, max_height=200, max_width=450):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.browser_mode = mode
        self.filter = filter
        self.dirpath = dirpath
        self.filepaths = []
        self.max_height = max_height
        self.max_width = max_width

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial",weight=QFont.Bold))
        width = 0
        for l in title.split("\n"):
            if (new_width := self.label.fontMetrics().boundingRect(l).width()) > width:
                width = new_width
        self.label.setFixedWidth(width+10)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.lineEdit = QPlainTextEdit(self)
        self.lineEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lineEdit.setFixedWidth(min(300,self.max_width))
        self.lineEdit.setFixedHeight(self.lineEdit.fontMetrics().boundingRect("TEST").height()+10)
        self.lineEdit.setLineWrapMode(QPlainTextEdit.NoWrap)

        layout.addWidget(self.lineEdit)

        self.button = QPushButton('Search')
        self.button.clicked.connect(self.setFile)
        layout.addWidget(self.button)
        layout.addStretch()

    def clear(self):
        self.lineEdit.clear()
        self.lineEdit.setFixedHeight(self.lineEdit.fontMetrics().boundingRect("TEST").height()+10)
        self.lineEdit.setFixedWidth(min(300,self.max_width))
        self.lineEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lineEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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
            new_width = self.lineEdit.fontMetrics().boundingRect(file).width() + 10
            height += self.lineEdit.fontMetrics().boundingRect(file).height()-1
            width = max(width, new_width)

        # if it is to high make a scrollbar and at width for the scrollbar
        if height > self.max_height:
            self.lineEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            width += 10
        else:
            self.lineEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        if width > self.max_width:
            self.lineEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            height += 20
        else:
            self.lineEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.lineEdit.setFixedWidth(min(self.max_width, width+10))
        self.lineEdit.setFixedHeight(min(self.max_height,height)) #max such that everything fits in the screen
        return height <= self.max_height

    def getPaths(self):
        return self.filepaths

class FileBrowserEnableQtw(FileBrowser):
    def __init__(self, title, mode=OPENFILE, filter='All files (*.*)', dirpath=DEFAULT_DIR, widget=None, mainPanel=None):
        super().__init__(title, mode=mode, filter=filter, dirpath=dirpath)

        if widget is None:
            raise ValueError("FileBrowserEnableQtw should be connected with an other widget")
        self.widget = widget
        self.mainPanel = mainPanel

    def setFile(self):
        is_small = super().setFile()
        if self.filter_out == "numpy arrays (*.npy)":
            self.widget.setEnabled(True)
        else:
            self.widget.setEnabled(False)
        self.mainPanel.change_layout(is_small)
