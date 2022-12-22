# general import for all files
import os
import sys
import glob
from multiprocessing import Pool
from multiprocessing import Process as multiprocess
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 100

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# Import own Raman Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing import *
from config import *
# weird torch error if this is not included
from utils.module import Conv_FFT, SelectLayer

if __name__ == '__main__':
    # only import main if this file is run.
    import Main

    Main.main()
