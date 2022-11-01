from StartUp import *

class AddIconToWidget(QHBoxLayout):
    def __init__(self, widget, icon, icontext="", *args, **kwargs):
        super().__init__()
        self.__pre_init__(*args, **kwargs)

        self.label = QLabel()
        self.label.setPixmap(widget.style().standardPixmap(icon))
        self.icontext = icontext

        self.addWidget(widget)
        self.addWidget(self.label)
        # This makes sure the icon is next to the widget
        self.insertStretch(-1, 1)
        self.label.installEventFilter(self)

    def __pre_init__(self, *args, **kwargs):
        pass

    def eventFilter(self, object, event):
        if event.type() == QEvent.Enter:
            # instant tooltip
            QToolTip.showText(
                QCursor.pos(),
                self.icontext,
                self.label
                )
            return True
        return False

class AddIconAndTextToWidget(AddIconToWidget):
    def __pre_init__(self, *args, **kwargs):
        try:
            text = kwargs['text']
        except KeyError:
            try:
                text = str(args[0])
            except IndexError:
                text = ''

        self.text = QLabel(text)
        self.addWidget(self.text)

class EditableComboBox(QComboBox):
    def __init__(self, validator):
        QComboBox.__init__(self)
        self.currentIndexChanged.connect(self.fix)
        self.setInsertPolicy(QComboBox.InsertAtCurrent)
        self.validator = validator(self.lineEdit())
        self.validator.setBottom(0.00)

    def fix(self, index):
        if (self.currentData() == '----'):
            self.setEditable(True)
            self.setValidator(self.float_validator)

        else:
            self.setEditable(False)
