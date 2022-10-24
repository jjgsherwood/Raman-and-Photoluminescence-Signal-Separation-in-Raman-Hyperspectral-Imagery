import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

class AddIconToWidget(qtw.QHBoxLayout):
    def __init__(self, widget, icon):
        super().__init__()
        label = qtw.QLabel()
        label.setPixmap(widget.style().standardPixmap(icon))

        self.addWidget(widget)
        self.addWidget(label)
        self.insertStretch(-1, 1)
