import numpy as np
# import copy

# from scipy.fft import dct
# from sklearn.decomposition import PCA
# from scipy.optimize import nnls

def gaussian(x, mu, sigma):
    x = x.reshape(-1,1)
    x = np.exp(-0.5* ((x - mu) / sigma)**2)
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def gaussian2(x, mu, sigma):
    x = x.reshape(-1,1)
    x = np.exp(-0.5* ((x - mu).reshape(-1,1) / sigma)**2)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

def logistic_sigmoid(x, mu, sigma):
    x = x.reshape(-1,1)
    x = sigmoid((x - mu).reshape(-1,1) / sigma)
    return x.reshape(-1, np.prod(mu.shape) * np.prod(sigma.shape))

class preliminary_photo_approximation():
    def __init__(self, order=7, mu=np.linspace(0,1,10), sigma=0.6, size=1300):
        order = np.arange(order)
        space = np.linspace(0,1,size)
        self.M = space[:, np.newaxis]**order
        if mu is None or sigma is None:
            self.kernel = self.M
        else:
            self.RBF = gaussian(space, mu, sigma)
            self.kernel = np.concatenate((self.M, self.RBF), 1)
        
    def __call__(self, x):
        p, *_ = np.linalg.lstsq(self.kernel, x, rcond=None)
        return np.sum(self.kernel * p, 1)

    
    
    
    
    
    
    
    
    
    