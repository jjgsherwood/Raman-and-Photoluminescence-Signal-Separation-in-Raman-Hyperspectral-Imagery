import numpy as np
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from scipy.fft import dct
import scipy.signal as signal
import scipy.interpolate as interpolate
import copy

import matplotlib.pyplot as plt


size = 9
plain = np.zeros((size,size))
plain[size//2,size//2] = 1
gaus = gaussian_filter(plain, (1,1))
CENTRE_CORR = gaus[size//2,size//2]
gaus[size//2,size//2] = 0
SUM_CORR = np.sum(gaus)

def median(values):
    return sorted(values)[len(values)//2]


def merge_defaultdicts(d,d1):
    for k,v in d1.items():
        if (k in d):
            d[k].update(d1[k])
        else:
            d[k] = d1[k]
    return d

class remove_cosmicrays():
    def __init__(self,
                 wavenumbers,
                 n_times=7,
                 FWHM_smoothing=3,
                 min_FWHM=5,
                 region_padding=5,
                 occurrence_percentage=0.01,
                 interpolate_degree=3):
        """
        Removes cosmic ray noise and return the new image plus the positions of the cosimic rays
        find_peaks is used to find maxima for an explantion see: https://en.wikipedia.org/wiki/Topographic_prominence

        wavenumbers: The wavenumbers used during measurements, used for determining frequencies when using DCT.
        n_times: Used as a threshold multiplier to determine if a spike is create by cosmic ray or not.
        FWHM_smoothing: The FWHM is translate to frequency in DCT domain,
                        which is then used as a boundery for the LBF (low band pass filter).
                        The LBF is then used to smooth the signal is spectral dimension.
        min_FWHM: This FWHM is a criteria for when it is or is not cosmic ray instead of Raman.
                  Everything smaller than this value passes the criteria for cosmic rays.
        region_padding: It is possible that the identified spike is a few wavenumbers (expected <5)
                        left or right of the spike location,
                        therefore paddding is used to be sure to find the spike.
                        It is also used for removing the cosmic ray noise.
                        The region around the spike is used to interpolate the new data.
                        This value determines how large this region is.
        occurrence_percentage: If a spike at a certain wavenumber occurs very often
                               (more than occurrence_percentage of the image),
                               than it is probably not a cosmic ray spike but
                               a Raman spike with a FWHW lower than min_FWHM.
        interpolate_degree: When removing the cosmic ray noise the region
                            around the spike is used to interpolate the new data.
                            This can be done with an interpolation degree between 1 to 5
                            (where 1 would be linear).
        """

        self.n_times = n_times
        self.min_FWHM = min_FWHM / (wavenumbers[-1] - wavenumbers[0]) * len(wavenumbers)
        self.k = int(2.674*(wavenumbers[-1] - wavenumbers[0]) / (np.pi*FWHM_smoothing))
        self.region_padding = region_padding
        self.occ_per = occurrence_percentage
        self.extend = region_padding
        self.interpolate_degree = interpolate_degree
        self.max_spike_diff = 2 # due to measurements errors spike can shift a few indices this determines that number

    def __call__(self, img):
        local_points = self.find_cosmic_ray_noise_spectral(img)
        neighbourhood_points = self.find_cosmic_ray_noise_neighbourhood(img)
        local_points = merge_defaultdicts(local_points, neighbourhood_points)
        local_points = self.wavenumber_check(np.prod(img.shape[:-1]), local_points)
        local_points = self.join_spikes_to_close(local_points)
        return self.remove_cosmicrays(img, local_points)

    def join_spikes_to_close(self, cosmic_ray_points):
        """
        Spike that are to close should be removed together and not seperatly.
        """
        for (x,y), spikes_info in cosmic_ray_points.items():
            spikes = np.array(list(spikes_info.keys()))
            if len(spikes) < 2:
                continue
            # keep joining spikes till they can no longer be joined
            while True:
                for i, spike_diff in enumerate(spikes[1:] - spikes[:-1]):
                    if spike_diff < 3 * self.region_padding:
                         spikes_info[spikes[i]] = (spikes_info[spikes[i]][0], spikes_info[spikes[i+1]][1]) #join spikes take boundery left from left spike and vise versa
                         cosmic_ray_points[(x,y)].pop(spikes[i+1])
                         spikes = np.array(list(spikes_info.keys()))
                         break
                else:
                    break
        return cosmic_ray_points

    def remove_cosmicrays(self, img, cosmic_ray_points):
        """
        Find left and right point untill the graph goes up.
        minimum distance between spike location and right or left is used as width.
        linear interpolate between spike location min width and spike location plus width
        """
        extend = self.region_padding
        for (x,y), dct in cosmic_ray_points.items():
            for left, right in dct.values():
                if right == img.shape[-1]-1:
                    values = list(img[x,y,left-extend+1:left+1])
                    rang = list(range(max(0,left-extend+1),left+1))

                    # use the average as an approximation for the interpolation loss
                    s = sum((img[x,y,left-extend+1:left] - np.mean(img[x,y,left-extend+1:left]))**2)
                elif left == 0:
                    values = list(img[x,y,right:right+extend])
                    rang = list(range(right,min(img.shape[-1], right+extend)))

                    # use the average as an approximation for the interpolation loss
                    s = sum((img[x,y,right:right+extend] - np.mean(img[x,y,right:right+extend]))**2)
                else:
                    values = list(img[x,y,left-extend+1:left+1]) + list(img[x,y,right:right+extend])
                    rang = list(range(max(0,left-extend+1),left+1)) + list(range(right,min(img.shape[-1], right+extend)))

                    # use the average as an approximation for the interpolation loss
                    s = sum((img[x,y,left-extend+1:left] - np.mean(img[x,y,left-extend+1:left]))**2) + \
                        sum((img[x,y,right:right+extend] - np.mean(img[x,y,right:right+extend]))**2)

                # ext=3 makes sure that if left or right is missing the boundery value is used for a horizontal line.
                func = interpolate.UnivariateSpline(rang, values, k=self.interpolate_degree, s=s, ext=3)
                img[x,y,left:right+1] = func(range(left,right+1))

        return img, cosmic_ray_points

    def wavenumber_check(self, img_size, cosmic_ray_points):
        """
        cosmic rays that appear with the same wavenumber in more than x % of the image are very sharp raman spikes.
        """
        n_pixel_threshold = int(img_size * self.occ_per)

        wavenumber, count = np.unique([y for x in cosmic_ray_points.values() for y, _ in x.items()], return_counts=True)
        duplicate_count = copy.copy(count)
        for i, w in enumerate(wavenumber):
            region_sum = 0
            for j, w1 in enumerate(wavenumber[i::-1]):
                if w - w1 <= self.max_spike_diff:
                    region_sum += duplicate_count[i-j]
                else:
                    break
            for j, w1 in enumerate(wavenumber[i+1:]):
                if w1 - w <= self.max_spike_diff:
                    region_sum += duplicate_count[i+j+1]
                else:
                    break
            count[i] = region_sum

        wavenumber_lst = []
        for wavenumber, count in zip(*(wavenumber, count)):
            if count > n_pixel_threshold:
                wavenumber_lst.append(wavenumber)

        for x,y in cosmic_ray_points.keys():
            for wavenumber in wavenumber_lst:
                try:
                    cosmic_ray_points[(x,y)].pop(wavenumber)
                except KeyError:
                    pass
        return cosmic_ray_points

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
        """
        diff is gaussian distributed and if diff is n times the std at a certain point (x,y,lambda).
        This points should be considerd cosmic rays noise.
        However, to be sure that it is cosmic ray find_peaks is used to find the spike.
        find_peaks uses prominence and width(FWHM) to determine if there is a spike.
        width must be smaller than the min_FWHM.
        prominence must be n/2 times larger than the median minus the minimum.
        median is used to make it independent of the potential spike.

        img: raw data.
        diff: the difference between smooth and raw data.
        """
        s = np.std(diff)

        tmp = defaultdict(list)
        for x,y,z in zip(*np.where(np.abs(diff) > self.n_times*s)):
            tmp[(x,y)].append(z)

        # find the spike and check if the width is small enough and height is big enough
        cosmicrays = defaultdict(dict)
        name_to_index = {x:i for i,x in enumerate(signal.find_peaks(img[0,0,:10], prominence=0, width=(None, self.min_FWHM))[1].keys())}
        for (x,y),z in tmp.items():
            for region in self.find_regions(z):
                l,r = max(0,region[0]-self.region_padding),min(img.shape[-1],region[-1]+self.region_padding)
                values = img[x,y,l:r]

                peaks, properties = signal.find_peaks(values, prominence=self.n_times/2*(median(values)-np.min(values)), width=(None, self.min_FWHM), rel_height=0.5) #the prominence is not the height of the guassian so a bit of composition.
                for peak, *properties in zip(peaks,*properties.values()):
                    left = properties[name_to_index['left_bases']] + l
                    right = properties[name_to_index['right_bases']] + l
                    cosmicrays[(x,y)][peak+l] = (left, right)

        return cosmicrays

    def find_regions(self, wavenumbers):
        regions = [[]]
        old_z = wavenumbers[0]-1
        for z in wavenumbers:
            if old_z + 2*self.region_padding >= z:
                regions[-1].append(z)
            else:
                regions.append([z])
            old_z = z
        return regions
