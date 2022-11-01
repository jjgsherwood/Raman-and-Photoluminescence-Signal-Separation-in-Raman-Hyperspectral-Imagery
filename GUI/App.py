from StartUp import *

import Dialog
import Widget

class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Welcome to Raman Spectroscopy Preprocessing module"
        self.preprocessing_methods = set()

        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # set layout
        vlayout = qtw.QVBoxLayout()
        self.setLayout(vlayout)

        self.fileBrowserPanel(vlayout)
        self.addPreProcessMethodPanel(vlayout)
        vlayout.addStretch()
        self.addButtonPanel(vlayout)

    def fileBrowserPanel(self, parentLayout):
        groupbox = qtw.QGroupBox("File Selection section")
        grid = qtw.QGridLayout()
        groupbox.setLayout(grid)

        self.waveFB = Dialog.FileBrowser('Select Wavenumber Info File', Dialog.OPENFILE)
        self.filesFB = Dialog.FileBrowserEnableQtw('Select Raman Hyperspectral Cubes',
                                                   Dialog.OPENFILES,
                                                   filter='text files (*.txt) ;; csv files (*.csv) ;; numpy arrays (*.npy)',
                                                   dirpath='../data',
                                                   widget=self.waveFB)
        self.dirFB   = Dialog.FileBrowser('Open Dir', Dialog.OPENDIRECTORY)
        self.dirFB.setEnabled(False)
        self.waveFB.setEnabled(False)

        grid.addWidget(self.filesFB, 0, 1)
        grid.addWidget(self.waveFB, 1, 1)
        grid.addWidget(self.dirFB, 2, 1)

        radiobutton = qtw.QRadioButton("Process File")
        radiobutton.setChecked(True)
        radiobutton.dir = False
        radiobutton.toggled.connect(self.onChangeFileInput)
        grid.addWidget(radiobutton, 0, 0)

        radiobutton = qtw.QRadioButton("Process Folder")
        radiobutton.dir = True
        radiobutton.toggled.connect(self.onChangeFileInput)
        grid.addWidget(radiobutton, 2, 0)

        parentLayout.addWidget(groupbox)

    def addPreProcessMethodPanel(self, parentLayout):
        """
        Setting for the new number of wavenumbers (min_step, max_step, predetermined number)
        """
        groupbox = qtw.QGroupBox("Extra Preprocessing steps")
        grid = qtw.QGridLayout()
        groupbox.setLayout(grid)

        checkbox = qtw.QCheckBox("Remove cosmic ray noise")
        checkbox.stateChanged.connect(self.changePreProcessingState(0))
        checkboxlayout = Widget.AddIconToWidget(checkbox, qtw.QStyle.SP_MessageBoxInformation)
        grid.addLayout(checkboxlayout, 0, 0)
        checkbox = qtw.QCheckBox("Interpolate saturation")
        checkbox.stateChanged.connect(self.changePreProcessingState(1))
        checkboxlayout = Widget.AddIconToWidget(checkbox, qtw.QStyle.SP_MessageBoxInformation)
        grid.addLayout(checkboxlayout, 1, 0)
        self.checkbox1 = qtw.QCheckBox("Make wavenumber steps constant between samples")
        self.checkbox1.stateChanged.connect(self.changePreProcessingState(2))
        checkboxlayout = Widget.AddIconToWidget(self.checkbox1, qtw.QStyle.SP_MessageBoxInformation)
        grid.addLayout(checkboxlayout, 2, 0)
        self.checkbox2 = qtw.QCheckBox("Make wavenumber steps constant within a sample")
        self.checkbox2.stateChanged.connect(self.changePreProcessingState(3))
        checkboxlayout = Widget.AddIconToWidget(self.checkbox2, qtw.QStyle.SP_MessageBoxWarning)
        grid.addLayout(checkboxlayout, 3, 0)

        parentLayout.addWidget(groupbox)

    def changePreProcessingState(self, box):
        def clickBox(state):
            if state == qtc.Qt.Checked:
                self.preprocessing_methods.add(box)
                # box 3 can not be enabled while box 2 is not.
                if box == 2:
                    self.preprocessing_methods.add(3)
                    self.checkbox2.setChecked(True)
            else:
                self.preprocessing_methods.remove(box)
        return clickBox

    def addProcessMethodPanel(self):
        """ Which noise filter to use. Which grad approximation to use. Other setting such as: Photo width (FWHM)"""
        pass

    def addSavePanel(self):
        """ Checkboxes to save each file as a txt and/or npy. Also save in new dir or same folder"""
        pass

    def addDisplayPanel(self):
        """ Checkboxes to say whether intermediate results and/or end results should be displayed"""
        pass

    def addButtonPanel(self, parentLayout):
        hlayout = qtw.QHBoxLayout()
        hlayout.addStretch()

        button = qtw.QPushButton("Cancel")
        button.clicked.connect(self.cancel)
        hlayout.addWidget(button)
        button = qtw.QPushButton("Start")
        button.clicked.connect(self.process)
        hlayout.addWidget(button)
        parentLayout.addLayout(hlayout)

    def cancel(self):
        self.close()

    def process(self):
        if self.dirFB.isEnabled(): # process whole directory
            return
        else: # process the selected files
            return

    def onChangeFileInput(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            if radioButton.dir:
                self.dirFB.setEnabled(True)
                self.filesFB.setEnabled(False)
                self.filesFB.clear()
                self.waveFB.setEnabled(False)
                self.waveFB.clear()
            else:
                self.dirFB.setEnabled(False)
                self.filesFB.setEnabled(True)
                self.dirFB.clear()
