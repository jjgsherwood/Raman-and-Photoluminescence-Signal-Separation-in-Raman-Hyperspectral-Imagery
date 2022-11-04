from scipy import interpolate
import numpy as np
from math import ceil
from multiprocessing import Pool

def interpolate_image(args):
    w, img, new_wavenumbers = args
    new_data = np.empty((*img.shape[:2], len(new_wavenumbers)))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            f = interpolate.InterpolatedUnivariateSpline(w, img[x,y], k=1, check_finite=True)
            new_data[x,y] = f(new_wavenumbers)
    return new_data

def correct_wavenumbers_between_samples(data, wavenumbers, stepsize='min'):
    """
    Correct wavenumbers such that the stepsize is constant and all samples have the same wavenumbers

    returns a new wavenumber array and the corrected samples
    """
    if len(data) != len(wavenumbers):
        raise ValueError("The number of samples should be equal to the number of wavenumber arrays")

    # left and rigth are determined such that the interpolation is always between points
    # and never outside the range of wavenumbers for all samples.
    left_wavenumber = np.max(wavenumbers[:,0])
    right_wavenumeber = np.min(wavenumbers[:,-1])
    if stepsize == 'min':
        min_stepsize = np.min(wavenumbers[:,1:] - wavenumbers[:,:-1])
    elif stepsize == 'max':
        min_stepsize = np.max(wavenumbers[:,1:] - wavenumbers[:,:-1])
    else:
        min_stepsize = stepsize

    # ceil to add an extra step for linspace (which includes stop)
    # This gives the stepsize that is larger than the smallest stepsize but as close as possible.
    steps = ceil((right_wavenumeber - left_wavenumber) / min_stepsize)
    new_wavenumbers = np.linspace(left_wavenumber, right_wavenumeber, steps)
    args = ((w, img, new_wavenumbers) for w, img in zip(wavenumbers, data))

    with Pool(None) as p:
        new_data = p.map(interpolate_image, args)
    new_data = np.array(new_data)

    return new_data, new_wavenumbers

def correct_wavenumbers_within_samples(data, wavenumbers, stepsize='min'):
    """
    Correct wavenumbers such that the stepsize is constant within a sample

    returns for each sample a new wavenumber array and the corrected sample
    """
    if len(data) != len(wavenumbers):
        raise ValueError("The number of samples should be equal to the number of wavenumber arrays")

    min_stepsize = np.min(wavenumbers[:,1:] - wavenumbers[:,:-1], 1)

    if stepsize == 'min':
        min_stepsize = np.min(wavenumbers[:,1:] - wavenumbers[:,:-1], 1)
    elif stepsize == 'max':
        min_stepsize = np.max(wavenumbers[:,1:] - wavenumbers[:,:-1], 1)
    else:
        min_stepsize = stepsize * np.ones(len(data))

    # ceil to add an extra step for linspace (which includes stop)
    steps = np.ceil((wavenumbers[:,-1] - wavenumbers[:,0]) / min_stepsize).astype(int)
    # This gives the stepsize that is larger than the smallest stepsize but as close as possible.
    new_wavenumbers = [np.linspace(wavenumbers[i,-1], wavenumbers[i,0], steps[i]) for i in range(len(data))]
    args = ((w, img, w_new) for w, img, w_new in zip(wavenumbers, data, new_wavenumbers))

    with Pool(None) as p:
        new_data = p.map(interpolate_image, args)

    return new_data, new_wavenumbers
