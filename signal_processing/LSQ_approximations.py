import numpy as np

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# plt.rcParams['figure.dpi'] = 500

def gaussian(x, mu, sigma):
    x = x.reshape(-1,1)
    x = np.exp(-0.5* ((x - mu).reshape(-1,1) / sigma)**2)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

class line_approximation():
    def __init__(self, order=0, size=1300):
        space = np.linspace(0,1,size)
        self.uniform = uniform(space, np.linspace(0,1,20, endpoint=False), size//20)
        self.kernel = self.uniform

    def __call__(self, x, w=None):
        if w is not None:
            p, *_ = np.linalg.lstsq(w @ self.kernel, w @ x, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.sum(self.kernel * p, 1)

class photo_approximation():
    def __init__(self, order=7, mu=np.linspace(0,1,10), sigma=np.linspace(0.6,0.7,1), size=1300):
        order = np.arange(order+1)
        space = np.linspace(0,1,size)
        self.M = space[:, np.newaxis]**order
        if mu is None or sigma is None:
            self.kernel = self.M
        else:
            self.RBF = gaussian(space, mu, sigma)
            self.kernel = np.concatenate((self.M, self.RBF), 1)

    def __call__(self, x, w=None):
        if w is not None:
            p, *_ = np.linalg.lstsq(w @ self.kernel, w @ x, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.sum(self.kernel * p, 1)

class preliminary_photo_approximation():
    def __init__(self, wavenumbers, order=7, FWHM=500, size=1300):

        # transform FWHM to guassian sigma and put the wavenumbers between 0 and 1
        # see https://mathworld.wolfram.com/GaussianFunction.html
        sigma = FWHM / (wavenumbers[-1] - wavenumbers[0]) / (2 * np.sqrt(2 * np.log(2)))
        sigma = np.array([sigma])
        width_between_max = (2 * np.sqrt(2 * np.log(100/99))) * sigma[0]
        mu = np.linspace(0,1,int(1/width_between_max)+1)

        order = np.arange(order+1)
        space = np.linspace(0,1,size)
        self.M = space[:, np.newaxis]**order
        if mu is None or sigma is None:
            self.kernel = self.M
        else:
            self.RBF = gaussian(space, mu, sigma)
            self.kernel = np.concatenate((self.M, self.RBF), 1)

    def __call__(self, x, w=None):
        x = np.log(x)
        if w is not None:
            p, *_ = np.linalg.lstsq(w @ self.kernel, w @ x, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.exp(np.sum(self.kernel * p, 1))
