import numpy as np
# import copy

# from scipy.fft import dct
# from sklearn.decomposition import PCA
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# plt.rcParams['figure.dpi'] = 500

# def gaussian(x, mu, sigma):
#     x = x.reshape(-1,1)
#     x = np.exp(-0.5* ((x - mu) / sigma)**2)
#     return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def gaussian(x, mu, sigma):
    x = x.reshape(-1,1)
    x = np.exp(-0.5* ((x - mu).reshape(-1,1) / sigma)**2)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

def logistic_sigmoid(x, mu, sigma):
    x = x.reshape(-1,1)
    x = sigmoid((x - mu).reshape(-1,1) / sigma)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

def uniform(x, a, width):
    s = 9
    x = np.zeros((*x.shape, a.shape[0] * s))
    for i, b in enumerate(a * x.shape[0]):
        b = int(b)
        if b+width > x.shape[0]:
            width = x.shape[0]-b
        x[max(0,b):b+width, i*s] = 1
        x[max(0,b):b+width, i*s+1] = np.linspace(0, 1, width)**1 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+2] = np.linspace(0, 1, width)**2 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+3] = np.linspace(0, 1, width)**3 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+4] = np.linspace(0, 1, width)**4 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+5] = np.linspace(1, 0, width)**2 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+6] = np.linspace(1, 0, width)**4 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+7] = np.linspace(1, 0, width)**1 #* np.log(np.linspace(1e-4, 1, width))
        x[max(0,b):b+width, i*s+8] = np.linspace(1, 0, width)**3 #* np.log(np.linspace(1e-4, 1, width))
    return x

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
#             self.uniform = uniform(space, np.linspace(0,1,5, endpoint=False), size//5)
            self.kernel = np.concatenate((self.M, self.RBF), 1)

    def __call__(self, x, w=None):
        if w is not None:
            p, *_ = np.linalg.lstsq(w @ self.kernel, w @ x, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.sum(self.kernel * p, 1)

class preliminary_photo_approximation():
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
        x = np.log(x)
        if w is not None:
            p, *_ = np.linalg.lstsq(w @ self.kernel, w @ x, rcond=None)
        else:
            p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.exp(np.sum(self.kernel * p, 1))
