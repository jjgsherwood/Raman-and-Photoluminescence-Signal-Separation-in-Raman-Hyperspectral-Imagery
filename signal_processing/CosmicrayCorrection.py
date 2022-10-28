import numpy as np
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from scipy.fft import dct
import scipy.signal as signal

size = 9
plain = np.zeros((size,size))
plain[size//2,size//2] = 1
gaus = gaussian_filter(plain, (1,1))
CENTRE_CORR = gaus[size//2,size//2]
gaus[size//2,size//2] = 0
SUM_CORR = np.sum(gaus)

class remove_cosmicrays():
    def __init__(self, wavenumbers, n_times=10, FWHM_smoothing=3, min_FWHM=5, region_padding=10):
        self.n_times = n_times
        self.min_FWHM = min_FWHM
        self.k = int(2*(wavenumbers[-1] - wavenumbers[0]) / (3*FWHM_smoothing))
        self.region_padding = region_padding

    def __call__(self, img):
        local_points = self.find_cosmic_ray_noise_spectral(img)
        neighbourhood_points = self.find_cosmic_ray_noise_neighbourhood(img)
        local_points.update(neighbourhood_points)
        return self.remove_cosmicrays(img, local_points)

    def remove_cosmicrays(self, img, cosmicray_points):
        """
        Find left and right point untill the graph goes up.
        minimum distance between spike location and right or left is used as width.
        linear interpolate between spike location min width and spike location plus width
        """
        pass

    def find_cosmic_ray_noise_spectral(self, img):
        """
        find cosmic ray noise in spectral dimension
        """
        cosine = dct(img, type=2, norm='backward')
        cosine[:,:,self.k:] = 0
        smooth = dct(cosine, type=3, norm="forward")
        diff = img - smooth

        return self.find_cosmic_ray_noise(img, diff)

    def find_cosmic_ray_noise_neighbourhood(self, img):
        # remove centre from the gaussian filter
        smooth = (gaussian_filter(img, (1,1,0)) - CENTRE_CORR * img) / SUM_CORR
        diff = np.diff(img) - np.diff(smooth)

        return self.find_cosmic_ray_noise(img, diff)

    def find_cosmic_ray_noise(self, img, diff):
        s = np.std(diff)

        tmp = defaultdict(list)
        for x,y,z in zip(*np.where(np.abs(diff) > self.n_times*s)):
            tmp[(x,y)].append(z)

        # find the spike and check if the width is small enough and height is big enough
        cosmicrays = defaultdict(list)
        for (x,y),z in tmp.items():
            for region in self.find_regions(z):
                l,r = max(0,region[0]-self.region_padding),min(img.shape[-1],region[-1]+self.region_padding)
                values = img[x,y,l:r]
                for p in signal.find_peaks(values, prominence=np.mean(values), width=(None, self.min_FWHM))[0]:
                    cosmicrays[(x,y)].append(p+l)

        return cosmicrays

    def find_regions(self, wavenumbers):
        regions = [[]]
        old_z = wavenumbers[0]-1
        for z in wavenumbers:
            if old_z + self.region_padding >= z:
                regions[-1].append(z)
            else:
                regions.append([z])
            old_z = z
        return regions
