import numpy as np
import copy

from scipy.fft import dct
from sklearn.decomposition import PCA
from scipy import signal, ndimage

from signal_processing import error

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# plt.rcParams['figure.dpi'] = 500

class RemoveNoiseFFTPCA():
    def __init__(self, algorithm='PCA', percentage_noise=0.01, wavenumbers=None, min_FWHM=2, error_function="MAPE", gradient_width=3, spike_padding=5, max_spike_width=150, Print=False):
        """
        Method to remove noise from raw data or split data.

        algorithm: choices are {'PCA', 'LPF', 'LPF_PCA', 'PCA_LPF'}.
            PCA: Only use PCA to reduce noise in the signal. Noise can be automatically be determined or specified.
            LPF: Only uses a low pass filter to reduce noise in the signal.
                If percentage_noise is not specified, wavenumbers and min_FWHM are used to reduce the noise.
                LPF works by transforming the signal with DCT (discreet cosine transform) and removing the high frequencies.
                Because of the boundery condintion of DCT,
                averaging the high frequency preserves the edges of the signal better.
            LPF_PCA: First uses LPF and than PCA. Can be used automated or with specific percentage_noise.
            PCA_LPF: First uses PCA and than LPF. Can only be used (semi-)automated.
                Warning final noise is can be higher than calculated percentage_noise,
                because only PCA depends on the percentage_noise, which is either automated or not.
                The LPF part depends always on the wavenumbers and minimum FWHM.
                This is not the case with LPF_PCA because PCA get the filtered signal.
                Important to note: applying LPF after LPF_PCA would not change the output.
        percentage_noise: is the level of noise in the total signal.
            if None, the noise is determined with a LPF,
            where all freqencies are removed with a smaller FWHM than min_FWHM.
        wavenumbers: needed to determine the FWHM only used when percentage_noise is None.
        min_FWHM: determines the level of noise in the data, only used when percentage_noise is None.
        error_function: determines how the noise is calculated default is MAPE (mean absolute percentage error).
            RMSPE (root mean squared percentage error) can also be used
            or any other function defined by the user with parameters target and prediction.
        gradient_width: The number of indices used to calculate the gradient.
            A higher number results in smoother gradient that are less effected by noise.
            A value of at least 2 is advised.
        spike_padding: When a spikes left and right borders are determined this number of indices is added to both sides.
            This is to compensate for the fact that the left and right borders are calculate at 30% of the maximum height instead of 0%.
            The width is calculated at 5% maximum height for the stability of the algorithm.
        max_spike_width: The maximum width of a spike in wavenumbers calculate at FW30M which is the full width at 30 percent of the maximum height.
        """
        self.k = None

        if percentage_noise is None:
            if wavenumbers is None:
                raise ValueError("Either the percentage_noise must be specified or the wavenumbers and min_FWHM must be specified!")
            self.k = int(2.674 * (wavenumbers[-1] - wavenumbers[0]) / (np.pi*min_FWHM))

        if algorithm not in {'PCA', 'LPF', 'LPF_PCA', 'PCA_LPF'}:
            raise ValueError("algorithm must be set to: PCA, LPF, LPF_PCA, PCA_LPF.")

        if algorithm == 'PCA_LPF' and self.k is None:
            raise ValueError("wavenumbers are needed to use the algorithm PCA_LPF, see documentation")

        self.algorithm = algorithm
        self.percentage_noise = percentage_noise
        self.error = error.STRING_TO_FUNCTION[error_function]
        self.Print = Print
        self.gradient_width = gradient_width
        self.spike_padding = spike_padding
        self.max_spike_width = int(max_spike_width * len(wavenumbers) / (wavenumbers[-1] - wavenumbers[0]))

    def __LPF_auto__(self, x):
        # find spike that are to similar to a dirac delta function
        spike = np.zeros(x.shape)
        if not self.gradient_width is None:
            grad = np.abs(x[:,self.gradient_width:] - x[:,:-self.gradient_width])
            std_grad = np.std(grad, 1)
            for i in range(x.shape[0]):
                position, details = signal.find_peaks(x[i], rel_height=.7, prominence=std_grad[i]*3, width=(None,self.max_spike_width))
                for j,p in enumerate(position):
                    half_w = int(details['widths'][j]//2+self.spike_padding)
                    left, right = max(0, p-half_w), min(p+half_w, x.shape[1]-1)
                    base = np.linspace(x[i,left], x[i,right], right-left+1)
                    spike[i,left:right+1] = x[i,left:right+1] - base

        # LPF
        cosine = dct(x-spike, type=2, norm='backward')
        cosine = cosine.T
        cosine[self.k:] = np.mean(cosine[self.k:], 0)
        # cosine[self.k:] = 0
        # cosine[self.k:] = ndimage.gaussian_filter(cosine[self.k:], sigma=(30,0), mode="nearest")

        return dct(cosine.T, type=3, norm="forward")+spike

    def __LPF_manual__(self, x):
        percentage_noise = self.percentage_noise if self.percentage_noise is not None else self.auto_percentage_noise
        left, right = 1, x.shape[-1]

        # find spike that are to similar to a dirac delta function
        spike = np.zeros(x.shape)
        if not self.gradient_width is None:
            grad = np.abs(x[:,self.gradient_width:] - x[:,:-self.gradient_width])
            std_grad = np.std(grad, 1)
            for i in range(x.shape[0]):
                position, details = signal.find_peaks(x[i], rel_height=.7, prominence=std_grad[i]*3, width=(None,self.max_spike_width))
                for j,p in enumerate(position):
                    half_w = int(details['widths'][j]//2+self.spike_padding)
                    left, right = max(0, p-half_w), min(p+half_w, x.shape[1]-1)
                    base = np.linspace(x[i,left], x[i,right], right-left+1)
                    spike[i,left:right+1] = x[i,left:right+1] - base

        cosine = dct(x-spike, type=2, norm='backward')
        cosine = cosine.T
        new_cosine = copy.copy(cosine)
        middle = (left + right) // 2
        while middle != left and middle != right:
            new_cosine[middle:] = np.mean(cosine[middle:], 0)
            x_new = dct(new_cosine.T, type=3, norm="forward") + spike

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
        # if automated is chosen do a LPF and calculate the error
        if self.percentage_noise is None:
            x_tmp = self.__LPF_auto__(x)
            self.auto_percentage_noise = self.error(x, x_tmp)
            # use x_tmp if LPF is the first algorithm and auto is selected.
            x_new = x_tmp if self.algorithm in ['LPF', 'LPF_PCA'] else x
        # else percentage noise is used for the LPF if selected.
        elif self.algorithm in ['LPF', 'LPF_PCA'] and self.k is None:
            x_new = self.__LPF_manual__(x)
        else:
            x_new = x

        # Do PCA this is automatically based on the percentage or error.
        # x_new has already a LPF applied on it or not based on the selected algorithm.
        if self.algorithm in ['LPF_PCA', 'PCA', 'PCA_LPF']:
            x_new = self.__PCA__(x, x_new)

        # after PCA do a LPF.
        if self.algorithm == 'PCA_LPF':
            x_new = self.__LPF_auto__(x_new)

        if self.Print:
            print(f"Final error: {self.error(x, x_new)}, using the error metric: {self.error.__name__}")

        return x_new
