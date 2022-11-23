import numpy as np
import copy
from multiprocessing import Pool

from signal_processing import error, LSQ_approximations as LSQ

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# plt.rcParams['figure.dpi'] = 500

class preliminary_split():
    def __init__(self, wavenumbers, order=9, FWHM=2000, size=1300):
        self.preliminary_photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)

    def __call__(self, img):
        poly = np.empty(img.shape)
        for i,org in enumerate(img):
            org[org <= 0] = 1e-8
            poly[i] = self.preliminary_photo_approximation(org)
        return poly
