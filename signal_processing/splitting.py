import numpy as np
import copy
from multiprocessing import Pool

from signal_processing import error, LSQ_approximations as LSQ

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 500

def Bezier_curve(p0,p1,p2):
    x0, x1, x2 = p0[0], p1[0], p2[0]
    b, a = x0 - x1, x0 - 2*x1 + x2
    d = x1**2 - x0*x2
    def inv_B(x):
        if not a:
            return np.linspace(0,1,x.shape[-1])
        return (b + np.sqrt(d + a * x)) / a

    p0, p1, p2 = p0[1], p1[1], p2[1]
    # p1[:] = 0
    def B(t):
        return np.outer((1-t)**2,p0) + np.outer(2*(1-t)*t, p1) + np.outer(t**2, p2)

    return B, inv_B

class preliminary_split():
    def __init__(self, wavenumbers, order=9, FWHM=2000, size=1300, convergence=5e-3):
        self.preliminary_photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.convergence = convergence

    def __call__(self, img):
        n = 0
        plt.plot(img[n])

        img[img <= 0] = 1e-8
        target = copy.deepcopy(img)
        weights = np.ones(img.shape[-1])

        i = 0
        # mean_error = np.ones(1)
        # while sum(mean_error > 0.1):
        old = -1
        photo = target
        while np.abs((new_old := error.MAPE(target, photo)) - old) > self.convergence:
            old = new_old
            photo = self.preliminary_photo_approximation(target, weights)
            to_high = photo > img
            mean_error = np.mean(to_high, 0)
            weights += mean_error
            weights /= np.mean(weights)
            target[to_high] *= 0.975
            plt.plot(photo[n], label=i)
            # plt.plot(target[n])
            # plt.plot(weights)
            plt.plot(np.mean(to_high, 0)*300)
            i += 1
        plt.legend()
        plt.show()
        print(i)
        return photo

        # return self.preliminary_photo_approximation(img)


class split():
    def __init__(self, wavenumbers, convergence=1e-3, intervals=50, segment_width=400, order=9, FWHM=2000, size=1300, algorithm="LS2"):
        self.photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.stepsize = int(size//intervals)
        self.width = int(segment_width / (wavenumbers[-1] - wavenumbers[0]) * len(wavenumbers))
        self.size = size
        self.LS = LSQ.photo_approximation(order=2, size=self.width, log=False)
        self.convergence = convergence
        self.algorithm = algorithm

    def __grad(self, photo_appr):
        grad =  np.pad((photo_appr[:,2:] - photo_appr[:,:-2]) / 2, ((0,0),(1,1)), 'edge')
        # extrapolate the padded gradient
        grad[:,0] -= grad[:,2] - grad[:,1]
        grad[:,-1] += grad[:,-2] - grad[:,-3]
        return grad

    def __grad_to_values(self, img, grad, left, right):
        # reconstruct the signal using the gradient
        rows = range(img.shape[0])
        slope = np.cumsum(grad, axis=1)
        index = np.argmin(img[:,left:right] - slope, axis=1)
        slope -= slope[rows,index,np.newaxis]
        slope += img[rows,left+index,np.newaxis]
        return slope

    def __fit_linear_line_segment(self, grad, left, right):
        return np.linspace(grad[:,left], grad[:,right-1], right-left, endpoint=False).T

    def __approximate_gradient_bounderies(self, img, grad, left, right):
        grad = self.__fit_linear_line_segment(grad, left, right)
        return self.__grad_to_values(img, grad, left, right)

    def __approximate_gradient_linearly(self, grad, left, middle, right):
        return np.concatenate(
            (self.__fit_linear_line_segment(grad, left, middle),
             self.__fit_linear_line_segment(grad, middle, right)),
            axis=1
        )

    def __approximate_gradient_bezier(self, grad, left, middle, right):
        p0 = np.array([left, grad[:,left]])
        p1 = np.array([middle, grad[:,middle]])
        p2 = np.array([right-1, grad[:,right-1]])

        B, inv_B = Bezier_curve(p0, p1, p2)
        x_axis = np.arange(left, right)
        return B(inv_B(x_axis)).T

    def __approximate_gradient_LS(self, grad, left, middle, right):
        grad = grad[:,left], grad[:,middle], grad[:,right-1]

        axis = np.array([left, middle, right])
        order = np.arange(3)
        kernel = axis[:, np.newaxis]**order
        p, *_ = np.linalg.lstsq(kernel, grad, rcond=None)
        axis = np.arange(left, right)
        kernel = axis[:, np.newaxis]**order
        return (kernel @ p).T

    def __fit_gradient_line_segment(self, img, grad, left, right):
        """
        LSQ seems to work better, but bezier curve is faster.
        Let the user decide
        """
        middle = int((right + left) // 2)
        if self.algorithm == "Linear":
            grad = self.__approximate_gradient_linearly(grad, left, middle, right)
        elif self.algorithm == "LS1":
            grad = self.LS(grad[:,left:right])
        elif self.algorithm == "LS2":
            grad = self.__approximate_gradient_LS(grad, left, middle, right)
        elif self.algorithm == "Bezier":
            grad = self.__approximate_gradient_bezier(grad, left, middle, right)
        else:
            raise ValueError("invalid algorithm for determining the new gradient!")

        return self.__grad_to_values(img, grad, left, right)

    def __iteration(self, img, photo):
        grad = self.__grad(photo)

        half_width = int(self.width//2)
        poly_max = np.zeros(img.shape)

        # approximate left bounderie
        for right in range(self.stepsize, self.width, self.stepsize):
            left = max(0, right-half_width)
            new_segment = self.__approximate_gradient_bounderies(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)

        # approximate middle section
        for left in range(0, self.size-self.width, self.stepsize):
            right = left + self.width
            new_segment = self.__fit_gradient_line_segment(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)

        # approximate right bounderie
        for left in range(self.size-self.width, self.size-self.stepsize, self.stepsize):
            right = min(self.size, left+half_width)
            new_segment = self.__approximate_gradient_bounderies(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)

        weights = np.ones(img.shape[-1])
        old = -1
        poly_max[poly_max <= 0] = 1e-8
        while np.abs((new_old := error.MAPE(poly_max, photo)) - old) > self.convergence:
            old = new_old
            photo = self.photo_approximation(poly_max, weights)
            to_high = photo > img
            mean_error = np.mean(to_high, 0)
            weights += mean_error
            weights /= np.mean(weights)
            poly_max[to_high] *= 0.975

        return photo

    def __call__(self, img, photo):
        new_photo = photo
        i = 0
        old = -1
        while np.abs((new_old := error.MAPE(photo, new_photo)) - old) > self.convergence:
            old = new_old
            photo = new_photo
            photo[photo <= 0] = 1e-8
            new_photo = self.__iteration(img, photo)
            i += 1
        print("iterations", i)
        return photo
