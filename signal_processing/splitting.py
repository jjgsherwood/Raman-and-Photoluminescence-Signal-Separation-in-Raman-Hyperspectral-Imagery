import numpy as np
import copy
from multiprocessing import Pool

from signal_processing import error, LSQ_approximations as LSQ

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# plt.rcParams['figure.dpi'] = 500

def split(args):
    idx, orgs, preliminary_photo_approximation, error_func, convergent_error = args
    idx, orgs, input, error_function, convergent_error = args
    preliminary_photo_approximation = LSQ.preliminary_photo_approximation(*input)
    error_func = error.STRING_TO_FUNCTION[error_function]

    # org, *_ = args
    polys = np.empty(orgs.shape)
    for i,org in enumerate(orgs):
        weights = np.ones(org.shape)
        target = copy.copy(org)
        old_error = -1
        while (new_error := error_func(org, target)) - old_error > convergent_error:
            old_error = new_error
            target[target < 0] = 1e-8
            poly = preliminary_photo_approximation(target, np.diag(weights))
            weights[poly > org] += 1
            weights /= np.mean(weights)
            target[poly > org] *= 0.975
        polys[i] = poly
    return idx, polys

class preliminary_split():
    def __init__(self, wavenumbers, error_function="MAPE", convergent_error=1e-3, order=9, FWHM=2000, size=1300):
        self.convergent_error = convergent_error
        self.size = size
        self.input = (wavenumbers, order, FWHM, size)
        self.error_function = error_function
        self.preliminary_photo_approximation = LSQ.preliminary_photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.error_func = error.STRING_TO_FUNCTION[error_function]

    def __call__(self, img):
        args = [(i, pixel, self.input, self.error_function, self.convergent_error) for i,pixel in enumerate(img)]

        photo = []
        with Pool(None) as p:
            result = p.map_async(split, args)
            for result in result.get():
                photo.append(result)
        # photo = [self.__split((pixel,)) for pixel in img]
        return photo
