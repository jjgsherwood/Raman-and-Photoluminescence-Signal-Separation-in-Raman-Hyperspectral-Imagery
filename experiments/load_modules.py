import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import glob
import copy

from scipy.fft import dct

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing import *
from GUI.Process import load_files, save_data

from IPython.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))