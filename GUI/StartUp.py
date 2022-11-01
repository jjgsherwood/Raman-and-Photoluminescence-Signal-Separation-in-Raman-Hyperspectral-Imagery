# general import for all files
import os
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# Import own Raman Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing import *

if __name__ == '__main__':
    # only import main if this file is run.
    import Main

    Main.main()
