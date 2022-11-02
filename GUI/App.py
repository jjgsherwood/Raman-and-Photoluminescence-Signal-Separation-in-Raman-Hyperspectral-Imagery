from StartUp import *

import Dialog
import Widget
import Process

class MainWindow(QWidget):
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
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)

        self.fileBrowserPanel(vlayout)
        self.addPreProcessMethodPanel(vlayout)
        vlayout.addStretch()
        self.addButtonPanel(vlayout)

    def fileBrowserPanel(self, parentLayout):
        groupbox = QGroupBox("File Selection section")
        grid = QGridLayout()
        groupbox.setLayout(grid)

        self.waveFB = Dialog.FileBrowser('Select Wavenumber Info File', Dialog.OPENFILE)
        self.filesFB = Dialog.FileBrowserEnableQtw('Select Raman Hyperspectral Cubes',
                                                   Dialog.OPENFILES,
                                                   filter='text files (*.txt) ;; csv files (*.csv) ;; numpy arrays (*.npy)',
                                                   dirpath='../data',
                                                   widget=self.waveFB)
        self.dirFB = Dialog.FileBrowser('Open Directory', Dialog.OPENDIRECTORY)
        self.dirFB.setEnabled(False)
        self.waveFB.setEnabled(False)

        grid.addWidget(self.filesFB, 0, 1)
        grid.addWidget(self.waveFB, 1, 1)
        grid.addWidget(self.dirFB, 2, 1)

        radiobutton = QRadioButton("Process File")
        radiobutton.setChecked(True)
        radiobutton.dir = False
        radiobutton.toggled.connect(self.onChangeFileInput)
        grid.addWidget(radiobutton, 0, 0)

        radiobutton = QRadioButton("Process Folder")
        radiobutton.dir = True
        radiobutton.toggled.connect(self.onChangeFileInput)
        grid.addWidget(radiobutton, 2, 0)

        parentLayout.addWidget(groupbox)

    def addRemoveCosmicRayNoisePanel(self):
        self.cosmic_ray_checkbox = QGroupBox("Remove cosmic ray noise")
        self.cosmic_ray_checkbox.setCheckable(True)
        subgrid = QGridLayout()
        self.cosmic_ray_checkbox.setLayout(subgrid)

        width = 70

        text = QLabel("boundery factor")
        subgrid.addWidget(text, 0, 0)
        text = QLabel("FWHM smoothing")
        subgrid.addWidget(text, 1, 0)
        text = QLabel("region_padding")
        subgrid.addWidget(text, 2, 0)
        text = QLabel("max FWHM")
        subgrid.addWidget(text, 3, 0)
        text = QLabel("max occurrence percentage")
        subgrid.addWidget(text, 4, 0)
        text = QLabel("interpolate degree")
        subgrid.addWidget(text, 5, 0)

        self.n_times_sb = QDoubleSpinBox()
        self.n_times_sb.setRange(2.0,15.0)
        self.n_times_sb.setValue(7.0)
        self.n_times_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.n_times_sb, QStyle.SP_MessageBoxInformation,icontext="The value is a factor which sets the boundery when a point is classified as cosmic ray noise.\nA higher value means that a point is less likely to be classified as cosim ray noise.\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 0, 1)

        self.FWHM_smoothing_sb = QDoubleSpinBox()
        self.FWHM_smoothing_sb.setMinimum(0)
        self.FWHM_smoothing_sb.setValue(3.0)
        self.FWHM_smoothing_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.FWHM_smoothing_sb, QStyle.SP_MessageBoxInformation,icontext="This value is translated to freqency and used as a cutoff point for the low band pass filter, which is used as spectral smoothing method.\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 1, 1)

        self.region_padding_sb = QSpinBox()
        self.region_padding_sb.setRange(1,25)
        self.region_padding_sb.setValue(5)
        self.region_padding_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.region_padding_sb, QStyle.SP_MessageBoxInformation, icontext="\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 2, 1)

        self.max_FWHM_sb = QDoubleSpinBox()
        self.max_FWHM_sb.setMinimum(0)
        self.max_FWHM_sb.setValue(5.0)
        self.max_FWHM_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.max_FWHM_sb, QStyle.SP_MessageBoxInformation, icontext="This value sets the maximum FWHM for a cosmic ray spike.\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 3, 1)

        self.max_oc_sb = QDoubleSpinBox()
        self.max_oc_sb.setRange(0.0,1.0)
        self.max_oc_sb.setValue(0.01)
        self.max_oc_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.max_oc_sb, QStyle.SP_MessageBoxInformation, icontext="\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 4, 1)

        self.interpolate_degree_sb = QSpinBox()
        self.interpolate_degree_sb.setRange(1,5)
        self.interpolate_degree_sb.setValue(3)
        self.interpolate_degree_sb.setMinimumWidth(width)
        spinboxlayout = Widget.AddIconToWidget(self.interpolate_degree_sb, QStyle.SP_MessageBoxInformation, icontext="\nSee the thesis for a full explanation.")
        subgrid.addLayout(spinboxlayout, 5, 1)

        return self.cosmic_ray_checkbox

    def addPreProcessMethodPanel(self, parentLayout):
        """
        Setting for the new number of wavenumbers (min_step, max_step, predetermined number)
        """
        self.cleaning_checkbox = QGroupBox("Extra Preprocessing steps")
        self.cleaning_checkbox.setCheckable(True)
        HLayout = QHBoxLayout()
        self.cleaning_checkbox.setLayout(HLayout)

        VLayout = QVBoxLayout()

        # saturation settings
        self.saturation_checkbox = QGroupBox("Interpolate saturation")
        self.saturation_checkbox.setCheckable(True)
        HLayout1 = QHBoxLayout()
        self.saturation_checkbox.setLayout(HLayout1)

        checkboxlayout = Widget.AddIconToWidget(QLabel("info"), QStyle.SP_MessageBoxInformation, "Replaces saturated datapoints with the average of neighbouring datapoints.")
        HLayout1.addLayout(checkboxlayout)
        self.saturation_region = Widget.EditableComboBox(QIntValidator, minimum=3)
        self.saturation_region.addItem('region size 3', 3)
        self.saturation_region.addItem('region size 5', 5)
        self.saturation_region.addItem('region size 7', 7)
        self.saturation_region.addItem('Type here for custom value', '----')
        checkboxlayout = Widget.AddIconToWidget(self.saturation_region, QStyle.SP_MessageBoxInformation, "Make sure that if you type a custom you press enter.\nValue should only contain numbers e.g. 3\nThe unity is in indices.")
        HLayout1.addLayout(checkboxlayout)

        VLayout.addWidget(self.saturation_checkbox)

        # wavenumber stettings
        self.wavenumber_checkbox = QGroupBox("Make wavenumber steps constant")
        self.wavenumber_checkbox.setCheckable(True)
        HLayout1 = QHBoxLayout()
        self.wavenumber_checkbox.setLayout(HLayout1)

        checkboxlayout = Widget.AddIconToWidget(QLabel("info"), QStyle.SP_MessageBoxWarning, "WARNING, If this function is not used, it is up to the user to make sure that the wavenumbers are constant!\nThis is important because several algorithms depend on it, such as descreet cosine transform.")
        HLayout1.addLayout(checkboxlayout)

        VLayout1 = QVBoxLayout()
        self.constant_wavenumber_choice = Widget.EditableComboBox(QDoubleValidator)
        self.constant_wavenumber_choice.addItem('minimum stepsize', 'min')
        self.constant_wavenumber_choice.addItem('maximum stepsize', 'max')
        self.constant_wavenumber_choice.addItem('Type here for custom value', '----')
        checkboxlayout = Widget.AddIconAndTextToWidget(self.constant_wavenumber_choice, QStyle.SP_MessageBoxInformation, text=4*" ", icontext="Make sure that if you type a custom value you press enter.\nValue should only contain numbers and one dot e.g. 2.5\nThe unity is in wavenumbers.\n\n The stepsize is automatically calculated when choosing minimum or maximum stepsize,\n depending on the stepsizes found in the data.")
        VLayout1.addLayout(checkboxlayout)

        self.consistent_all = QCheckBox("Make wavenumbers consistent\nbetween all images.")
        self.consistent_all.setChecked(True)
        checkboxlayout = Widget.AddIconToWidget(self.consistent_all, QStyle.SP_MessageBoxInformation, "This makes sure that all spectra contain the same wavenumbers.\nThis makes comparing spectra between images possible.")
        VLayout1.addLayout(checkboxlayout)

        HLayout1.addLayout(VLayout1)

        # make Vertical layout complete
        VLayout.addWidget(self.wavenumber_checkbox)

        # make Horizontal layout complete
        HLayout.addLayout(VLayout)
        HLayout.addWidget(self.addRemoveCosmicRayNoisePanel())

        parentLayout.addWidget(self.cleaning_checkbox)

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
        hlayout = QHBoxLayout()
        hlayout.addStretch()

        button = QPushButton("Quit")
        button.clicked.connect(self.quit)
        hlayout.addWidget(button)
        button = QPushButton("Cancel")
        button.clicked.connect(self.cancel)
        hlayout.addWidget(button)
        button = QPushButton("Start")
        button.setToolTip('This will start processing the selected files')
        button.setToolTipDuration(5000)
        button.clicked.connect(self.run)
        hlayout.addWidget(button)
        parentLayout.addLayout(hlayout)

    def quit(self):
        try:
            self.p.terminate()
        except AttributeError:
            pass
        self.close()

    def cancel(self):
        try:
            if not self.p.is_alive():
                return
        except AttributeError:
            pass
        else:
            self.p.terminate()
            print("process canceled")
            QMessageBox.information(self, "Process Canceled", "The current process is canceled")

    def run(self):
        """
        read out all the values and start the process function
        """
        # make sure not another progres is running
        try:
            if self.p.is_alive():
                QMessageBox.warning(self, "Process Error", "Currently there is already a program running!\nPleas press cancel before starting a new process.")
                return
        except AttributeError:
            pass

        if (output := self.__get_files()) is None:
            return

        if (preprocessing_variables := self.__get_preprocessing_variables()) is None:
            return

        # show variables
        if SHOW_INPUT:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"See selected parameters below:")
            msg.setInformativeText('<br>'.join((str(k)+' : '+str(v) for k,v in preprocessing_variables.items())))
            msg.setWindowTitle("Inspect given parameters")
            try:
                wavefiles = '\n'.join(output[1])
            except IndexError:
                wavefiles = ''
            msg.setDetailedText("The following files are selected: \n\n" + '\n'.join(output[0]) + "\n\n" + wavefiles)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            te = msg.findChild(QTextEdit)
            if te is not None:
                te.setLineWrapMode(QTextEdit.NoWrap)
                te.parent().setFixedWidth(te.document().idealWidth() + te.document().documentMargin() + te.verticalScrollBar().width())
                te.setFixedHeight(300)
            if QMessageBox.Cancel == msg.exec_():
                return

        args = [(output, preprocessing_variables, None)]
        self.p = multiprocess(target=Process.run, args=args)
        self.p.start()

    def __get_preprocessing_variables(self):
        """
        check and get the value for the preprocessing step.
        """
        preprocessing_variables = {}
        if not self.cleaning_checkbox.isChecked():
            return preprocessing_variables

        if self.saturation_checkbox.isChecked():
            preprocessing_variables['saturation_width'] = self.saturation_region.currentData()
            if preprocessing_variables['saturation_width'] == '----':
                try:
                    preprocessing_variables['saturation_width'] = int(self.saturation_region.itemText(self.saturation_region.currentIndex()))
                except ValueError:
                    dlg = QMessageBox.warning(self, "Input Error", "Manual region size is selected,\nbut no whole number is given!\n\nDo not forget to press enter.")
                    return

        if self.wavenumber_checkbox.isChecked():
            preprocessing_variables['stepsize'] = self.constant_wavenumber_choice.currentData()
            if preprocessing_variables['stepsize'] == '----':
                try:
                    preprocessing_variables['stepsize'] = float(self.constant_wavenumber_choice.itemText(self.constant_wavenumber_choice.currentIndex()))
                except ValueError:
                    dlg = QMessageBox.warning(self, "Input Error", "Manual wavenumber stepsize is selected,\nbut no number is given!\n\nDo not forget to press enter.")
                    return

            preprocessing_variables['all_images_same_stepsize'] = self.consistent_all.isChecked()


        if self.cosmic_ray_checkbox.isChecked():
            preprocessing_variables['n_times'] = self.n_times_sb.value()
            preprocessing_variables['FWHM_smoothing'] = self.FWHM_smoothing_sb.value()
            preprocessing_variables['region_padding'] = self.region_padding_sb.value()
            preprocessing_variables['max_FWHM'] = self.max_FWHM_sb.value()
            preprocessing_variables['max_oc'] = self.max_oc_sb.value()
            preprocessing_variables['interpolate_degree'] = self.interpolate_degree_sb.value()

        return preprocessing_variables

    def __get_files(self):
        """
        read the filenames and check compatibility.
        """
        if self.dirFB.isEnabled(): # process whole directory
            # check if a directory is selected
            if not self.dirFB.getPaths():
                dlg = QMessageBox.warning(self, "Input Error", "Please select a folder!")
                return

            npy_files = glob.glob(self.dirFB.getPaths()[0]+'/*.npy')
            wave_files = glob.glob(self.dirFB.getPaths()[0]+'/*Wavenumbers.npy')
            txt_files = glob.glob(self.dirFB.getPaths()[0]+'/*.txt')
            # check if there are numpy files in the folder and a wavenumber file
            if npy_files and wave_files:
                if len(wave_files) == 1:
                    return npy_files, wave_files
                for file in npy_files:
                    if file.replace('.npy','_Wavenumbers.npy') not in wave_files:
                        dlg = QMessageBox.warning(self, "Input Error", f"{file} has no wavenumber file!")
                        return
                return npy_files, wave_files
            elif txt_files:
                return (txt_files,)
            else:
                dlg = QMessageBox.warning(self, "Input Error", "Files in folder could not be loaded!\n\nNo .txt files were found in the folder!\nNo .npy files were found in the folder!\nOr ..wavernumbers.npy was missing in the folder!")
            return
        else: # process the selected files
            files = self.filesFB.getPaths()
            if not files:
                dlg = QMessageBox.warning(self, "Input Error", "No files were selected!")
                return

            if self.filesFB.widget.isEnabled():
                wave_files = self.waveFB.getPaths()
                if len(wave_files) == 1:
                    return files, wave_files
                for file in files:
                    if file.replace('.npy','_Wavenumbers.npy') not in wave_files:
                        dlg = QMessageBox.warning(self, "Input Error", f"{file} has no wavenumber file!")
                        return
                return npy_files, wave_files
            else:
                return (files,)

        dlg = QMessageBox.warning(self, "Input Error", "Something unexpected went wrong!")
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
