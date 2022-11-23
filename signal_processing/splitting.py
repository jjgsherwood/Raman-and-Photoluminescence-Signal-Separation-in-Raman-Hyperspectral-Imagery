import numpy as np
import copy
from multiprocessing import Pool

from signal_processing import error, LSQ_approximations as LSQ

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 500

class preliminary_split():
    def __init__(self, wavenumbers, order=9, FWHM=2000, size=1300):
        self.preliminary_photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)

    def __call__(self, img):
        # # n = 0
        # # plt.plot(img[n])
        #
        img[img <= 0] = 1e-8
        # target = copy.deepcopy(img)
        # weights = np.ones(img.shape[-1])
        # # for i in range(10):
        # mean_error = np.ones(1)
        # i = 0
        # while sum(mean_error > 0.1):
        # # for i in range(10):
        #     photo = self.preliminary_photo_approximation(target, weights)
        #     to_high = photo > img
        #     mean_error = np.mean(to_high, 0)
        #     weights += mean_error
        #     weights /= np.mean(weights)
        #     target[to_high] *= 0.975
        #     # plt.plot(photo[n], label=i)
        #     # plt.plot(target[n])
        #     # plt.plot(weights)
        #     # plt.plot(np.mean(to_high, 0)*300)
        #     # i += 1
        # # plt.legend()
        # # plt.show()
        # # print(i)
        # return photo

        return self.preliminary_photo_approximation(img)


def Bezier_curve(p0,p1,p2):
    x0, x1, x2 = p0[0], p1[0], p2[0]
    b, a = x0 - x1, x0 - 2*x1 + x2
    d = x1**2 - x0*x2
    def inv_B(x):
        if not a:
            return np.linspace(0,1,x.shape[-1])
        return (b + np.sqrt(d + a * x)) / a

    p0, p1, p2 = p0[1], p1[1], p2[1]
    p1 = 0
    def B(t):
        return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

    return B, inv_B

def fit_gradient_line_segment(x, grad, left, right):
    """
    LSQ seems to work better, but bezier curve is faster.
    Let the user decide
    """
    middle = int((right + left) // 2)

    grad_left = grad[left]
    grad_middle = grad[middle]
    grad_right = grad[right]
#     left_grad = np.linspace(grad_left, grad_middle, middle-left)
#     right_grad = np.linspace(grad_middle, grad_right, right-middle)
#     grad = np.concatenate((left_grad, right_grad))

    axis = np.arange(left, right)
    order = np.arange(3)
    kernel = axis[:, np.newaxis]**order
    p, *_ = np.linalg.lstsq(kernel, grad[left: right], rcond=None)
    grad = np.sum(kernel * p, 1)

#     p0 = np.array([left, grad_left])
#     p1 = np.array([middle, grad_middle])
#     p2 = np.array([right, grad_right])

#     B, inv_B = Bezier_curve(p0, p1, p2)
#     x_axis = np.arange(left, right+1)
#     grad = B(inv_B(x_axis))[:-1]

    value = 0
    slope = [0]
    for g in grad:
        value += g
        slope.append(value)

    slope = np.array(slope)
    index = np.argmin(x[left:right+1] - slope)
    slope -= slope[index]

    slope += x[left+index]
#     plt.plot(x_axis, slope)
    return slope, left, right+1

class split():
    def __init__(self, wavenumbers, intervals=50, segment_width=400, order=9, FWHM=2000, size=1300):
        self.preliminary_photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.stepsize = int(size//intervals)
        self.width = segment_width / (wavenumbers[-1] - wavenumbers[0]) * len(wavenumbers)
        self.size = size

    def __grad(self, photo_appr):
        grad =  np.pad((photo_appr[:,2:] - photo_appr[:,:-2]) / 2, ((0,0),(1,1)), 'edge')
        # extrapolate the padded gradient
        grad[:,0] -= grad[:,2] - grad[:,1]
        grad[:,-1] += grad[:,-2] - grad[:,-3]
        return grad

    def __fit_linear_line_segment(self, grad, left, right):
        return np.linspace(grad[:,left], grad[:,right], right-left, endpoint=False).T

    def __call__(self, img, photo):
        grad = self.__grad(photo)

        half_width = int(self.width//2)
        poly_max = np.zeros(img.shape)
        for right in range(10, half_width, self.stepsize):
            new_grad = self.__fit_linear_line_segment(grad, 0, right)
            poly_max[:,:right] = np.maximum(poly_max[:,:right], new_grad)

        for right in range(half_width, self.width, self.stepsize):
            left = right - half_width
            new_grad = self.__fit_linear_line_segment(grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_grad)

        for left in range(0, self.size-self.width, self.stepsize):
            right = left + self.width
            new_grad = fit_gradient_line_segment(x, grad, left, right)
            poly_max[left:right] = np.maximum(poly_max[left:right], new_grad)


        weights = np.ones(img.shape[-1])
        old = -1
        photo = poly_max
        photo2 = poly_max
        while (new_old := error.MAPE(photo, photo2)) - old > 1e-4:
            old = new_old
            photo = self.preliminary_photo_approximation(target, weights)
            to_high = photo > img
            mean_error = np.mean(to_high, 0)
            weights += mean_error
            weights /= np.mean(weights)
            target[to_high] *= 0.975

        return photo
