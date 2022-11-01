# general import for all files
import os
import sys

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

# Import own Raman Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing import *

if __name__ == '__main__':
    # only import main if this file is run.
    import Main

    Main.main()
