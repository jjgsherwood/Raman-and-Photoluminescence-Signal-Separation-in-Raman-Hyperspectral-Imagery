from StartUp import *

DESIGNER = 0

if DESIGNER:
    import test

    class MainWindow(qtw.QMainWindow, test.Ui_MainWindow):
        def __init__(self):
            super().__init__()
            self.setupUi(self)

else:
    from App import MainWindow


def main():
    app = qtw.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()
