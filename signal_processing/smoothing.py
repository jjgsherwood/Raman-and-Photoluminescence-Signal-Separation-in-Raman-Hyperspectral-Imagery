import numpy as np
import copy

from scipy.fft import dct
from sklearn.decomposition import PCA

def MAPE(x, y):
    return np.mean(np.abs(1 - y/x))

def RMSPE(x, y):
    return np.mean(np.sqrt(np.mean((1 - y/x)**2, 1)))
    

class RemoveNoiseFFTPCA():
    def __init__(self, algorithm='LPF_PCA', percentage_noise=0.01, wavenumbers=None, min_HWHM=2, error_function=MAPE, Print=False):
        """
        Method to remove noise from raw data or split data.
        
        algorithm: choices are {'PCA', 'LPF', 'LPF_PCA', 'PCA_LPF'}.
            PCA: Only use PCA to reduce noise in the signal. Noise can be automatically be determined or specified.
            LPF: Only uses a low pass filter to reduce noise in the signal. 
                If percentage_noise is not specified, wavenumbers and min_HWHM are used to reduce the noise.
                LPF works by transforming the signal with DCT (discreet cosine transform) and removing the high frequencies.
                Because of the boundery condintion of DCT, 
                averaging the high frequency preserves the edges of the signal better.
            LPF_PCA: First uses LPF and than PCA. Can be used automated or with specific percentage_noise.
            PCA_LPF: First uses PCA and than LPF. Can only be used (semi-)automated.
                Warning final noise is can be higher than calculated percentage_noise, 
                because only PCA depends on the percentage_noise, which is either automated or not.
                The LPF part depends always on the wavenumbers and minimum HWHM.
                This is not the case with LPF_PCA because PCA get the filtered signal.
                Important to note: applying LPF after LPF_PCA would not change the output.
        percentage_noise: is the level of noise in the total signal.
            if None, the noise is determined with a LPF, 
            where all freqencies are removed with a smaller HWHM than min_HWHM.
        wavenumbers: needed to determine the HWHM only used when percentage_noise is None.
        min_HWHM: determines the level of noise in the data, only used wehn percentage_noise is None.
        error_function: determines how the noise is calculated default is MAPE (mean absolute percentage error).
            RMSPE (root mean squared percentage error) can also be used 
            or any other function defined by the user with parameters target and prediction.       
        """
        self.k = None

        if percentage_noise is None:
            if wavenumbers is None:
                raise ValueError("Either the percentage_noise must be specified or the wavenumbers and min_HWHM must be specified!")
            self.k = int((wavenumbers[-1] - wavenumbers[0]) / (2*min_HWHM))
                        
        if algorithm not in {'PCA', 'LPF', 'LPF_PCA', 'PCA_LPF'}:
            raise ValueError("algorithm must be set to: PCA, LPF, LPF_PCA, PCA_LPF.")

        if algorithm == 'PCA_LPF' and self.k is None:
            raise ValueError("wavenumbers are needed to use the algorithm PCA_LPF, see documentation")
            
        self.algorithm = algorithm
        self.percentage_noise = percentage_noise
        self.error = error_function
        self.Print = Print
    
    def __LPF_auto__(self, x):
        cosine = dct(x, type=2, norm='backward')
        cosine = cosine.T
        cosine[self.k:] = np.mean(cosine[self.k:], 0)
        return dct(cosine.T, type=3, norm="forward")
    
    def __LPF_manual__(self, x):
        percentage_noise = self.percentage_noise if self.percentage_noise is not None else self.auto_percentage_noise
        left, right = 1, x.shape[-1]

        cosine = dct(x, type=2, norm='backward')
        cosine = cosine.T
        new_cosine = copy.copy(cosine)
        middle = (left + right) // 2
        while middle != left and middle != right:
            new_cosine[middle:] = np.mean(cosine[middle:], 0)
            x_new = dct(new_cosine.T, type=3, norm="forward")

            if self.error(x, x_new) < percentage_noise:
                right = middle
            else:
                left = middle
            middle = (left + right) // 2
        return x_new
    
    def __PCA__(self, x, x_new):
        percentage_noise = self.percentage_noise if self.percentage_noise is not None else self.auto_percentage_noise
        left, right = 1, x.shape[-1]

        pca = PCA(svd_solver='full')
        pca.fit(x_new)
        components_ = copy.copy(pca.components_)
        middle = (left + right) // 2
        while middle != left and middle != right:
            pca.components_ = components_[:middle]
            x_pca = pca.inverse_transform(pca.transform(x_new))

            if self.error(x, x_pca) < percentage_noise:
                right = middle
            else:
                left = middle
            middle = (left + right) // 2
        return x_pca
    
    def __call__(self, x):
        x_new = x
        if self.percentage_noise is None:
            x_new = self.__LPF_auto__(x)
            self.auto_percentage_noise = self.error(x, x_new)
        
        if self.algorithm in ['LPF', 'LPF_PCA']:
            if self.k is None:
                x_new = self.__LPF_manual__(x)

        if self.algorithm == ['PCA', 'LPF_PCA', 'PCA_LPF']:
            x_new = self.__PCA__(x, x_new)
            
        if self.algorithm == 'PCA_LPF':
            x_new = self.__LPF_auto__(x_new)
            
        if self.Print:
            print(f"Final error: {self.error(x, x_new)}, using the error metric: {self.error.__name__}")
            
        return x_new