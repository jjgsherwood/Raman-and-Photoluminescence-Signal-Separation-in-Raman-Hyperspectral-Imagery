#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Implementation of Robust PCA from dganguli:
# Direct Link: https://github.com/dganguli/robust-pca/blob/master/r_pca.py

# Why __future__? https://docs.python.org/2/reference/simple_stmts.html#future
from __future__ import division, print_function

# division would be a function in the future versions
# print was not a function in Python 2

import numpy as np, matplotlib.pyplot as plt, os
from PyPDF2 import PdfFileMerger

try:
    # pylab is needed for matplotlib
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape) # initialization with zeros
        self.Y = np.zeros(self.D.shape) # initialization with zeros

        if mu:
            self.mu = mu
        else:
            # np.prod --> returns the product of the elements
            self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
    
    # About Frobenius norm:
    
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # https://dsp.stackexchange.com/questions/45886/
    
    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')
    
    # https://www.geeksforgeeks.org/class-method-vs-static-method-python/
    
    @staticmethod
    def shrink(M, tau):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))
        

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        # iter is a built-in function in Python 3, so replacing it with rep
        rep = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        while (err > _tol) and rep < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            rep += 1
            if (rep % iter_print) == 0 or rep == 1 or rep > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(rep, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ceil.html
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)
        
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmin.html
        ymin = np.nanmin(self.D)
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmax.html
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        
        plt.rcParams.update({'figure.max_open_warning': 0})   
        
        for n in range(numplots):
            fig, ax = plt.subplots()
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            plt.title(f"Plot {str(n+1)}", fontsize = 20)
            fig.savefig(f"plot_{str(n+1)}.pdf", dpi = 888)
            if not axis_on:
                plt.axis('off')
                       
        # https://stackoverflow.com/questions/3444645/merge-pdf-files

        pdfs = [f"plot_{str(n)}.pdf" for n in range(1, numplots+1)]

        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write("Figures.pdf")
        merger.close()
        
        for pdf in pdfs:
            os.remove(pdf)

