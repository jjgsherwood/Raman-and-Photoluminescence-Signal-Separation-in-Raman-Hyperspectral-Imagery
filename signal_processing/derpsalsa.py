from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import detrend

import numpy as np

# settings:
global min_struct_el
min_struct_el = 7
max_number_baseline_iterations = 30 # number of iterations in baseline search


def derpsalsa_baseline(x, y, display=2, als_lambda=5e7, als_p_weight=1.5e-3):
    """ asymmetric baseline correction
    Algorithm by Sergio Oller-Moreno et al.
    Parameters which can be manipulated:
    als_lambda  ~ 5e7
    als_p_weight ~ 1.5e-3
    (found from optimization with random 5-point BL)
    """

    # 0: smooth the spectrum 16 times
    #    with the element of 1/100 of the spectral length:
    zero_step_struct_el = int(2*np.round(len(y)/200) + 1)
    y_sm = molification_smoothing(y, zero_step_struct_el, 16)
    # compute the derivatives:
    y_sm_1d = np.gradient(y_sm)
    y_sm_2d = np.gradient(y_sm_1d)
    # weighting function for the 2nd der:
    y_sm_2d_decay = (np.mean(y_sm_2d**2))**0.5
    weifunc2D = np.exp(-y_sm_2d**2/2/y_sm_2d_decay**2)
    # weighting function for the 1st der:
    y_sm_1d_decay = (np.mean((y_sm_1d-np.mean(y_sm_1d))**2))**0.5
    weifunc1D = np.exp(-(y_sm_1d-np.mean(y_sm_1d))**2/2/y_sm_1d_decay**2)

    weifunc = weifunc1D*weifunc2D

    # exclude from screenenig the edges of the spectrum (optional)
    weifunc[0:zero_step_struct_el] = 1; weifunc[-zero_step_struct_el:] = 1

    # estimate the peak height
    peakscreen_amplitude = (np.max(detrend(y)) - np.min(detrend(y)))/8 # /8 is good, because this is a characteristic height of a tail
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = als_lambda * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    # k = 10 * morphological_noise(y) # above this height the peaks are rejected
    for i in range(max_number_baseline_iterations):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = als_p_weight * weifunc * np.exp(-((y-z)/peakscreen_amplitude)**2/2) * (y > z) + (1-als_p_weight) * (y < z)
    baseline = z
    #@Test&Debug: #
    if display > 1:
        plt.plot(x, y - z, 'r',
                  x, y, 'k',
                  x, baseline, 'b');
        plot_annotation = 'derpsalsa baseline';
        plt.text(0.5, 0.92, plot_annotation, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes);
        plt.show()
    return z

def molification_smoothing (rawspectrum, struct_el, number_of_molifications):
    """ Molifier kernel here is defined as in the work of Koch et al.:
        JRS 2017, DOI 10.1002/jrs.5010
        The structure element is in pixels, not in cm-1!
        struct_el should be odd integer >= 3
    """
    molifier_kernel = np.linspace(-1, 1, num=struct_el)
    molifier_kernel[1:-1] = np.exp(-1/(1-molifier_kernel[1:-1]**2))
    molifier_kernel[0] = 0; molifier_kernel[-1] = 0
    molifier_kernel = molifier_kernel/np.sum(molifier_kernel)
    denominormtor = np.convolve(np.ones_like(rawspectrum), molifier_kernel, 'same')
    smoothline = rawspectrum
    i = 0
    for i in range (number_of_molifications) :
        smoothline = np.convolve(smoothline, molifier_kernel, 'same') / denominormtor
        i += 1
    return smoothline
