import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import dct

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing import *
