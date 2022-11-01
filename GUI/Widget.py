from StartUp import *

class AddIconToWidget(qtw.QHBoxLayout):
    def __init__(self, widget, icon):
        super().__init__()
        label = qtw.QLabel()
        label.setPixmap(widget.style().standardPixmap(icon))

        self.addWidget(widget)
        self.addWidget(label)
        self.insertStretch(-1, 1)
