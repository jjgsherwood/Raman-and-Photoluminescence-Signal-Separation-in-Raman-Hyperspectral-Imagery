import numpy as np
import copy

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
    def B(t):
        return np.outer((1-t)**2,p0) + np.outer(2*(1-t)*t, p1) + np.outer(t**2, p2)

    return B, inv_B

class preliminary_split():
    def __init__(self, wavenumbers, order=1, FWHM=400, size=1300, convergence=5e-3):
        """
        This will give a photoluminences approximation based on the raw signal.

        The photoluminences approximation is calculated using an iterative algorithm.
        Each iteration consists of a weighted least squared with target correction (WLST)
        and an update of the weights and targets.

        wavenumbers: Wavenumbers are needed to convert FWHM in wavenumbers of the radial basis function (RBF) to indices.
        size: size is needed to convert FWHM in wavenumbers of the radial basis function (RBF) to indices.
        FWHM: Determines the sigma of the RBF, this also effects the number of radial basis (gaussians)
            because the distance between two means is determined based on the width of the gaussian at 80% height.
        order: To fit the photoluminences signal better the kernel used in the least square algorithm includes an RBF kernel and a polynomial kernel.
            This determines the order of the polynomial kernel.
        convergence: convergence determines when the algorithm should stop.
            The convergence is calculated using the mean average percentage error (MAPE) between the approximations of photoluminences at t and t-1.
            So if convergence is set to 1e-3 than the algorithm stops if the difference between two approximation is less than 0.1 percent.
        """
        self.preliminary_photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.convergence = convergence

    def __call__(self, img):
        img[img <= 0] = 1e-8
        target = copy.deepcopy(img)
        weights = np.ones(img.shape[-1])

        photo = target
        photo_old = -1
        while error.MAPE(photo_old, photo) > self.convergence:
            photo, photo_old = self.preliminary_photo_approximation(target, weights), photo
            to_high = photo > img
            mean_error = np.mean(to_high, 0)
            weights += mean_error
            weights /= np.mean(weights)
            target[to_high] += (img-photo)[to_high] * 0.5
            target[target <= 0] = 1e-3
            photo[photo < 1] = 1
        return photo

class split():
    def __init__(self, wavenumbers, size=1300, FWHM=300, order=1, convergence=1e-3, segment_width=400, algorithm="Bezier curve"):
        """
        This will give the photoluminences signal based on either the raw signal or a photoluminences approximation.
        When given an approximation the algorithm becomes much faster.

        The photoluminences signal is calculated using two iterative algorithms, an inner and outer iterative algorithm.
        The inner iterative algorithm consists of a weighted least squared with target correction (WLST) and an update of the weights and targets.
        The outer iterative algorithm consists of a gradient based segment fit algorithm and the inner iterative algorithm.
        The gradient based segment fit algorithm consists of two steps:
            - First, a sliding window goes over the curve and for each segment the gradient is smoothed.
              Here three option are available Linear, Quadratic, Bezier curve.
            - Next, the new gradients segments are intergrated to get curve segments.
            - Then, these curve segment are placed below the raw curve in a way that
              both curves collide but not intersect each other.
            - Last, the maximum is taking for each wavenumber over the curve segments that overlap.

        wavenumbers: Wavenumbers are needed to convert FWHM in wavenumbers of the radial basis function (RBF) to indices.
        size: size is needed to convert FWHM in wavenumbers of the radial basis function (RBF) to indices.
        FWHM: Determines the sigma of the RBF, this also effects the number of radial basis (gaussians)
            because the distance between two means is determined based on the width of the gaussian at 80% height.
        order: To fit the photoluminences signal better the kernel used in the least square algorithm includes an RBF kernel and a polynomial kernel.
            This determines the order of the polynomial kernel.
        convergence: convergence determines when the algorithm should stop.
            This effects the innerloop which converts the segments into a smooth curve and
            the outerloop which determines when the final photoluminences signal is converged.
            The convergence is calculated using the mean average percentage error (MAPE) between the approximations of photoluminences at iteration i and i-1.
            So if convergence is set to 1e-3 than the algorithm stops if the difference between two approximation is less than 0.1 percent.
        segment_width: This determines the segment width for the gradient based segment fit algorithm.
        algorithm: This determines which gradient based segment fit algorithm is used.
            There are three options: Linear, Quadratic, Bezier curve.
        """
        self.photo_approximation = LSQ.photo_approximation(wavenumbers, order=order, FWHM=FWHM, size=size)
        self.width = int(segment_width / (wavenumbers[-1] - wavenumbers[0]) * size)
        self.stepsize = int(self.width // 10)
        self.size = size
        self.LS = LSQ.photo_approximation(order=2, size=self.width, log=False)
        self.convergence = convergence
        self.algorithm = algorithm
        self.wavenumbers = wavenumbers

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
        p0 = np.array([left, grad[:,left]], dtype=object)
        p1 = np.array([middle, grad[:,middle]], dtype=object)
        p2 = np.array([right-1, grad[:,right-1]], dtype=object)

        B, inv_B = Bezier_curve(p0, p1, p2)
        x_axis = np.arange(left, right)
        return B(inv_B(x_axis)).T

    def __approximate_gradient_LS(self, grad, left, middle, right):
        """
        Depricated Overfitting
        """
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
        elif self.algorithm == "Quadratic":
            grad = self.LS(grad[:,left:right])
        elif self.algorithm == "Bezier curve":
            grad = self.__approximate_gradient_bezier(grad, left, middle, right)
        else:
            raise ValueError("invalid algorithm for determining the new gradient!")

        return self.__grad_to_values(img, grad, left, right)

    def __iteration(self, img, photo):
        grad = self.__grad(photo)
        half_width = int(self.width//2)
        poly_max = np.zeros(img.shape)
        #
        # n = 0
        # plt.plot(self.wavenumbers, img[n])
        # approximate left bounderie
        for right in range(self.stepsize, self.width, self.stepsize):
            left = max(0, right-half_width)
            new_segment = self.__approximate_gradient_bounderies(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)
            # plt.plot(self.wavenumbers[left: right], new_segment[n])
        # approximate middle section
        for left in range(0, self.size-self.width, self.stepsize):
            right = left + self.width
            new_segment = self.__fit_gradient_line_segment(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)
            # plt.plot(self.wavenumbers[left: right], new_segment[n])

        # approximate right bounderie
        for left in range(self.size-self.width, self.size-self.stepsize, self.stepsize):
            right = min(self.size, left+half_width)
            new_segment = self.__approximate_gradient_bounderies(img, grad, left, right)
            poly_max[:,left:right] = np.maximum(poly_max[:,left:right], new_segment)
            # plt.plot(self.wavenumbers[left: right], new_segment[n])

        # plt.show()

        # plt.plot(self.wavenumbers, img[n])
        # plt.plot(self.wavenumbers, photo[n])
        # plt.plot(self.wavenumbers, poly_max[n])

        # plt.show()

        weights = np.ones(img.shape[-1])
        poly_max[poly_max <= 0] = 1e-8
        photo_old = -1
        photo = poly_max
        i = 0
        alpha = 1
        # plt.plot(img[n], linewidth=0.3)
        while ((old:=error.MAPE(photo_old, photo)) > self.convergence and i < 30) or i < 5:
            # print("inner error",old, flush=True)
            photo, photo_old =  (1-alpha) * photo + alpha * self.photo_approximation(poly_max, weights), photo
            to_high = photo > img
            mean_error = np.mean(to_high, 0)
            weights += mean_error
            weights /= np.mean(weights)
            # poly_max[to_high] *= 0.975
            poly_max[to_high] += (img-photo)[to_high] * 0.5
            poly_max[poly_max <= 0] = 1e-3
            i += 1
            photo[photo < 1] = 1
            # plt.plot(img[n], linewidth=0.3)
            # plt.plot(photo_old[n], linewidth=0.1)
            # plt.plot(photo[n], linewidth=0.1)
            # plt.show()
        # plt.show()
        # print("inner iterations", i, flush=True)
        # plt.plot(self.wavenumbers, photo[n])
        # plt.show()
        return photo

    def __call__(self, img, new_photo=None):
        if new_photo is None:
            new_photo = img

        new_photo, photo = self.__iteration(img, new_photo), -1
        i = 0
        alpha = 1
        # n = 81
        # plt.plot(img[n], linewidth=0.3)
        # plt.plot(new_photo[n], linewidth=0.1)
        while ((old:=error.MAPE(photo, new_photo)) > self.convergence and i < 30) or i < 2:
            # forget photo approximation
            # if i == 0:
            #     photo = new_photo
            # # learning rate schedular
            # if i >= 10:
            #     alpha *= 0.9
            # print(f"iteration: {i} gives an outer error {old} with a learning rate of {alpha}", flush=True)
            new_photo[new_photo <= 0] = 1e-8
            new_photo, photo = (1-alpha)*photo + alpha * self.__iteration(img, new_photo), new_photo
            photo[photo < 1] = 1
            # plt.plot(img[n], linewidth=0.3)
            # plt.plot(photo[n], linewidth=0.1)
            # plt.plot(new_photo[n], linewidth=0.1)
            # plt.show()
            i += 1
        # plt.show()
        # print(f"iteration: {i} gives an outer error {old} with a learning rate of {alpha}", flush=True)
        # print("outer iterations", i, flush=True)
        return photo
