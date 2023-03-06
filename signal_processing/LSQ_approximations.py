import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 500

def gaussian(x, mu, sigma):
    x = x.reshape(-1,1)
    x = np.exp(-0.5* ((x - mu).reshape(-1,1) / sigma)**2)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

class photo_approximation():
    def __init__(self, wavenumbers=None, order=1, FWHM=500, size=1300, log=False, width=0.95):
        self.log = log
        order = np.arange(order+1)
        space = np.linspace(0,1,size)
        self.M = space[:, np.newaxis]**order
        if FWHM is None or wavenumbers is None:
            self.kernel = self.M
        else:
            # transform FWHM to guassian sigma and put the wavenumbers between 0 and 1
            # see https://mathworld.wolfram.com/GaussianFunction.html
            sigma = FWHM / (wavenumbers[-1] - wavenumbers[0]) / (2 * np.sqrt(2 * np.log(2)))
            sigma = np.array([sigma])
            width_between_max = (2 * np.sqrt(2 * np.log(1/width))) * sigma[0]
            mu = np.linspace(-width_between_max/2,1+width_between_max/2,int(1/width_between_max)+2)

            self.RBF = gaussian(space, mu, sigma)
            self.kernel = np.concatenate((self.M, self.RBF), 1)

    def __call__(self, x, w=None):
        if self.log:
            # x = np.sqrt(x)
            x = np.log(x)

        if w is not None:
            kw = np.multiply(self.kernel, w[:,np.newaxis])
            xw = np.multiply(x, w[np.newaxis,:]).T
            p, *_ = np.linalg.lstsq(kw, xw, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x.T, rcond=None)

        if self.log:
            # return ((self.kernel @ p)**2).T
            return np.exp(self.kernel @ p).T
        return (self.kernel @ p).T
