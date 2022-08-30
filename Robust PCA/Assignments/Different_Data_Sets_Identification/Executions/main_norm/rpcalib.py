#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Implementation of Robust PCA from dganguli:
# Direct Link: https://github.com/dganguli/robust-pca/blob/master/r_pca.py

# An implementation using Principle Component Pursuit (PCP) by alternating directions

# Why __future__? https://docs.python.org/2/reference/simple_stmts.html#future
from __future__ import division, print_function

# division would be a function in the future versions
# print was not a function in Python 2

import numpy as np, matplotlib.pyplot as plt, pandas as pd, os

from PyPDF2 import PdfFileMerger
from random import choices

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
    
    #*****************************************************************************#
    
    # Link where the algorithm (page 55 in it) with expansions was found below:
    # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/reportRobustPCA.pdf
    
    # Also, see page 29 in the following link (and explanations before and after):
    # https://www.comp.nus.edu.sg/~cs5240/lecture/robust-pca.pdf
    # This one is used as reference in the comments later.
    
class R_pca:
    
    def __init__(self, data, mu=None, lmbda=None):
        """
        Specifications
        
        Parameters
        ----------
        self: representing the instance of the class
        data: must be a 2D numpy array, with dimensions Points x Intensities
        mu: μ from the algorithm
        lmbda: λ from the algorithm, i.e. the Lagrange multiplier,
               - lambda would be a more appropriate name, but could cause issues
                 as it's a taken word by Python for creating lambda functions,
        
        Function
        --------
                 - The constructor function, which is used to initialize the object.
        """
        
        # Some exception handling:
        if not isinstance(data, np.ndarray): # data must be of numpy array type
            raise Exception('data should be a numpy array. Call again properly')
        
        D = np.copy(data)
        
        self.D = D
        self.S = np.zeros(self.D.shape) # E in algorithm, initialization with zeros
        self.Y = np.zeros(self.D.shape) # Y in algorithm, initialization with zeros

        if mu:
            if mu <= 0: # mu must be positive
                raise Exception('mu must be a positive number')
            self.mu = mu
        else:
            # np.prod --> returns the product of all elements of its input array
            self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))

        self.mu_inv = 1 / self.mu # 1/μ

        if lmbda:
            self.lmbda = lmbda
        else:
            # this is the optimal λ (see pdf)
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
    
    # About Frobenius norm:
    
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # https://dsp.stackexchange.com/questions/45886/
    
    @staticmethod
    def frobenius_norm(M):
        """
        Specifications
        
        It is a static method (declared on top with the appropriate decorator).
        
        Parameters
        ----------
        M: must be an array (numpy array) (exception handling included)
        
        Function
        --------
                 - Uses np.linalg.norm function to compute the Frobenius norm of M
        Returns
        -------
               - The Frobenius norm of M
        
        """
        # Some exception handling:
        if not isinstance(M, np.ndarray): # M must be of numpy array type
            raise Exception('M input should be a numpy array. Call again properly')
        
        return np.linalg.norm(M, ord='fro')
    
    # https://www.geeksforgeeks.org/class-method-vs-static-method-python/
    
    @staticmethod
    def shrink(M, tau):
        """
        Specifications
        
        It is a static method (declared on top with the appropriate decorator).
        
        Parameters
        ----------
        M: must be an array (numpy array) (exception handling included)
        tau: the value to shrink (In Page 16 in the algorithm Tε == Ttau) 
        
        Function
        --------
                 - Uses np.linalg.norm function to compute the Frobenius norm of M
        Returns
        -------
               - The shrinkage's transformation result.
        """
        if not isinstance(M, np.ndarray):
            raise Exception('M input should be a numpy array. Call again properly')
            
        # absolute (np.abs) is the L1 norm:
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))
        
        # Link for the documentation of np.maximum function below:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html
        
        # Link for the documentation of np.sign function below:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sign.html

    def svd_threshold(self, M, tau):
        """
        Specifications
        
        Parameters
        ----------
        self: representing the instance of the class
        M: must be an array (numpy array) (exception handling included)
        tau: the value to shrink
        
        Function
        --------
                - Performs reduced SVD on the M input (step 5 in the algorithm)
        Returns
        -------
               - The calculations of step 6 in the algorithm
        """
        if not isinstance(M, np.ndarray):
            raise Exception('M input should be a numpy array. Call again properly')
            
        # full_matrices = False ---> reduced SVD
        U, S, V = np.linalg.svd(M, full_matrices=False)
        
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        """
        Specifications:
        
        Parameters
        ----------
        self: the corresponding object
        tol: tolerance (the error calculated must get lower or equal to this),
             must be numeric or None (default is None)
        max_iter: number of repetitions, must be an int > 0 (default is 1000)
        iter_print: must be int > 0, used for printing the error
                    after the specified number (default is 100)
        
        Function
        --------
                 - Fits the data and does the transformations/updates 
                   of the values of L, S, Y and error until convergence
                   or (if no convergence) till repetitions reach max_iter
        Returns
        -------
                - the last L, i.e. the Low Rank (A in the algorithm) 
                - the last S, i.e. the Sparse (E in the algorithm)
        """
        
        if isinstance(tol, int) or isinstance(tol, float) or tol == None:
            pass
        else:
            raise Exception('tol must be a positive number')

        if not isinstance(max_iter, int):
            raise Exception("max_iter must be a positive integer")
        
        if not isinstance(iter_print, int):
            raise Exception("iter_print must be a positive integer")
            
        assert(iter_print > 0), "iter_print must be a positive integer"
        assert(max_iter > 0), "max_iter must be a positive integer"
        
        # initializations:
        rep = 0 # iter is a built-in function in Python 3, so replacing it with rep
        err = np.Inf # infinite value
        Sk = self.S 
        Yk = self.Y 
        Lk = np.zeros(self.D.shape)

        if tol:
            assert(tol > 0), "tol must be a positive number"
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        # Repeat until convergence (error < tolerance) or given iterations reached:    
        while (err > _tol) and rep < max_iter:
            
            # Iterative Thresholding Algorithm:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)
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

    def plot_fit(self, size=None, tol=0.1, axis_on=True, print_yrange=False):
        """
        Specifications:
    
        Parameters
        ----------
        self: the corresponding object
        size: the dimensions of the matrix (if given, must be a tuple)
        tol: used to determine the range of y-axis: [ymin - tol, ymax + tol]
             (must be a positive numerical value)
        axis_on: condition if you want axis on (True) or off (False)
        print_yrange: condition if you want to print the range (ymin, ymax)
                      of the y-axis plotted (must be True or False)
        Function
        --------
                 - Creates a plot with all spectra after RPCA:
                   Low Rank + Sparse and Low Rank plots for each point
                 - Saves the plots in PDF format in the working directory,
                   all in one called RPCA_plots.pdf
                   - each page of the PDF has the plot for each point
        """
        if isinstance(tol, int) or isinstance(tol, float):
            pass
        else:
            raise Exception('tol must be a positive number')
            
        assert(tol > 0), "tol must be a positive number"
        
        if not isinstance(axis_on, bool):
            raise Exception("axis_on must have logical value: False or True")
            
        if not isinstance(print_yrange, bool):
            raise Exception("print_yrange must have logical value: False or True")
            
        n, d = self.D.shape
            
        if size:
            if not isinstance(size, tuple):
                raise Exception("size (dimensions) must be given in a tuple: (rows, columns)")
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
        
        if print_yrange:
            print('ymin: {0}, ymax: {1}\n'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        
        plt.figure()
        
        # To omit warning for too many figures opened:
        plt.rcParams.update({'figure.max_open_warning': 0}) 
       
        for n in range(numplots):
            fig, ax = plt.subplots()
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r', label = "Low Rank + Sparse")
            plt.plot(self.L[n, :], 'b', label = "Low Rank")
            plt.title(f"Point {str(n+1)}", fontsize = 20)
            plt.legend(loc = 0) # legend's location set to best
            if not axis_on:
                plt.axis('off')    
            plt.savefig(f"plot_{str(n+1)}.pdf", dpi = 2500) 
        
        ## Merge the PDFs created into a single one ##
        
        # https://stackoverflow.com/questions/3444645/merge-pdf-files

        pdfs = [f"plot_{str(n)}.pdf" for n in range(1, numplots+1)]

        merger = PdfFileMerger()
        
        for pdf in pdfs:
            merger.append(pdf)

        merger.write("RPCA_plots.pdf")
        merger.close()
        
        # Remove the separate PDFs:
        for pdf in pdfs:
            os.remove(pdf)
    
    def plot_fit_one_plot(self, size=None, tol=0.1, axis_on=True, print_yrange=False):
        """
        Specifications:
    
        Parameters
        ----------
        self: the corresponding object
        size: the dimensions of the matrix (if given, must be a tuple)
        tol: used to determine the range of y-axis: [ymin - tol, ymax + tol]
             must be a positive number (exception handling included)
        axis_on: condition if you want axis on (True) or off (False)
        print_yrange: condition if you want to print the range (ymin, ymax)
                      of the y-axis plotted (must be True or False)        
        Function
        --------
                 - Creates a plot with all spectra after RPCA:
                   i)  Low Rank + Sparse and Low Rank plots for each point
                   ii) Adding each plot to the previous one in one plot and
                       so on, resulting in a plot with all plots together
                 - Saves the plot in PDF format in the working directory,
                   it's called All_in_one_plot.pdf
        """
        if isinstance(tol, int) or isinstance(tol, float):
            pass
        else:
            raise Exception('tol must be a positive number')
    
        assert(tol > 0), "tol must be a positive number"
            
        if not isinstance(axis_on, bool):
            raise Exception("axis_on must have logical value: False or True")
        
        if not isinstance(print_yrange, bool):
            raise Exception("print_yrange must have logical value: False or True")            
        
        n, d = self.D.shape

        if size:
            if not isinstance(size, tuple):
                raise Exception("size (dimensions) must be given in a tuple: (rows, columns)")            
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)
        
        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        
        if print_yrange:
            print('ymin: {0}, ymax: {1}\n'.format(ymin, ymax))
            
        numplots = np.min([n, nrows * ncols])
        
        plt.figure()
        
        plt.rcParams.update({'figure.max_open_warning': 0}) 
       
        for n in range(numplots):
            plt.ylim((ymin - tol, ymax + tol))
            
            if n == (numplots-1):
                plt.plot(self.L[n, :] + self.S[n, :], 'k', label='Low Rank + Sparse')
                plt.plot(self.L[n, :], 'maroon', label = "Low Rank")
                plt.title("All Spectra in One Plot", fontsize = 20)
                plt.legend(loc = 0) # legend's location set to best
                if not axis_on:
                    plt.axis('off')   
                plt.savefig("All_in_one_plot.pdf", dpi = 2500)  
            else:
                plt.plot(self.L[n, :] + self.S[n, :], 'black')
                plt.plot(self.L[n, :], 'maroon')
                plt.title(f"Point {str(n+1)}", fontsize = 20)
                if not axis_on:
                    plt.axis('off')
                    
    def plot_four_low_rank_randomly(self, size=None, tol=0.1, axis_on=True, print_yrange=False):
        """
        Specifications:
    
        Parameters
        ----------
        self: the corresponding object
        size: the dimensions of the matrix (if given, must be a tuple)
        tol: used to determine the range of y-axis: [ymin - tol, ymax + tol],
             must be a positive number (exception handling included)
        axis_on: condition if you want axis on (True) or off (False)
        print_yrange: condition if you want to print the range (ymin, ymax)
                      of the y-axis plotted (must be True or False)        
        Function
        --------
                 - Creates a plot of the low rank graphs of 4 randomly chosen points.
                 - Saves the plot in PDF format in the working directory,
                   it's called Low_rank_four_random_points.pdf
        """
        if isinstance(tol, int) or isinstance(tol, float):
            pass
        else:
            raise Exception('tol must be a positive number')
            
        assert(tol > 0), "tol must be a positive number"        
            
        if not isinstance(axis_on, bool):
            raise Exception("axis_on must have logical value: False or True")
        
        if not isinstance(print_yrange, bool):
            raise Exception("print_yrange must have logical value: False or True")        
        
        n, d = self.D.shape
        
        if n < 10:
            return 'Function valid only for 10 or more points'

        if size:
            if not isinstance(size, tuple):
                raise Exception("size (dimensions) must be given in a tuple: (rows, columns)")
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)
        
        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)

        if print_yrange:
            print('ymin: {0}, ymax: {1}\n'.format(ymin, ymax))        
        
        numplots = np.min([n, nrows * ncols])
        
        plt.rcParams.update({'figure.max_open_warning': 0})     
            
        rand_vals = choices(range(numplots), k=4)
        
        while len(set(rand_vals)) != 4:
            rand_vals = choices(range(numplots), k=4)    
        
        l = rand_vals.copy()
        
        fig, ax = plt.subplots()
        plt.ylim((ymin - tol, ymax + tol))
        plt.plot(self.L[l[-4], :], 'darkgreen', label = f"Point {l[-4]}")
        plt.plot(self.L[l[-3], :], 'magenta', label = f"Point {l[-3]}")
        plt.plot(self.L[l[-2], :], 'yellow', label = f"Point {l[-2]}")
        plt.plot(self.L[l[-1], :], 'black', label = f"Point {l[-1]}")
        text = f"Input ({rand_vals}".replace('[','').strip(']') + ')'
        plt.title(text, fontsize = 20)
        plt.legend(loc = 0)
        if not axis_on:
             plt.axis('off')
        plt.savefig(f"Low_rank_four_random_points.pdf", dpi = 2500) 
        
    def plot_sparse_only(self,):
        """
        Specifications:
    
        Parameters
        ----------
        self: the corresponding object
        
        Function
        --------
                 - Creates a plot of all the sparse graphs in the same plot.
                 - Saves the plot in PDF format in the working directory,
                   the pdf saved is named Sparse_only.pdf
        """
        n, d = self.D.shape
        
        if n < 10:
            return 'Function valid only for 10 or more points'
        
        df_S = pd.DataFrame(self.S.T)
        
        df_S.plot(figsize = (15, 12), legend = False)

        plt.title('All Sparse in One Graph', fontsize = 30)

        plt.savefig('Sparse_only.pdf', dpi = 2500)
                  
    def plot_low_rank_only(self,):
        """
        Specifications:
    
        Parameters
        ----------
        self: the corresponding object
        
        Function
        --------
                 - Creates a plot of all the low rank graphs in one plot.
                 - Saves the plot in PDF format in the working directory,
                   the pdf saved is named Low_rank_only.pdf
        """
        n, d = self.D.shape
        
        if n < 10:
            return 'Function valid only for 10 or more points'

        df_L = pd.DataFrame(self.L.T)
        
        df_L.plot(figsize = (15, 12), legend = False)

        plt.title('All Low Ranks in One Graph', fontsize = 30)

        plt.savefig('Low_rank_only.pdf', dpi = 2500) 
        

