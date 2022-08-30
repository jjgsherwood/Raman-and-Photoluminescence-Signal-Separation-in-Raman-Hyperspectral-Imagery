#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# PCA function (simplest implementation) found in the link below:
# https://stackoverflow.com/questions/40721260/how-to-use-robust-pca-output-as-principal-component-eigenvectors-from-traditio

def pca(data, numComponents=None):
    """Principal Components Analysis
    
    From: http://stackoverflow.com/a/13224592/834250
    
    ==============
    Specifications
    ==============
    
    No exception handling included!

    Parameters
    ----------
    data : 'numpy.ndarray'
        numpy array of data to analyse
    numComponents : 'int'
        number of principal components to use

    Returns
    -------
    comps : 'numpy.ndarray'
        Principal components
    evals : 'numpy.ndarray'
        Eigenvalues
    evecs : 'numpy.ndarray'
        Eigenvectors
    """
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    # Use of 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    if numComponents is not None:
        evecs = evecs[:, :numComponents]
    # Carry out the transformation on the data using eigenvectors,
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def elbow_method(eigvals_chosen):
    """
    ==============
    Specifications
    ==============
    
    Parameters
    ----------
    eigvals_chosen: Must be a numpy array vector, dimensions (D, 1) or (D,).
                    Not applicable if the input is of dimensions (1, D).
    Function
    --------
            - Saves a plot of PCAs in descending order of importance.
    """
    if not isinstance(eigvals_chosen, (np.ndarray)):
        raise Exception('The argument must be in numpy array type. Please call again properly!')
    try:
        eigvals_chosen = eigvals_chosen.reshape((eigvals_chosen.shape[0], 1))
    except:
        raise Exception('The input must be a vector!')
    Y = eigvals_chosen/np.sum(eigvals_chosen)
    lvals = len(eigvals_chosen)
    fig, ax = plt.subplots()
    x = np.arange(lvals)
    xlabs = [f'PC{i}' for i in range(1, lvals+1)]
    ax.set_xticklabels(xlabs)
    plt.xticks(x, xlabs, rotation='vertical')
    plt.title('Principal components in descending order of importance',
              loc='center', fontdict = {'fontsize' : 10.88})
    plt.plot(x, Y, '.-', c = 'green')
    fig.set_dpi(480)
    plt.savefig('Inspect_PCAs.pdf', bbox_inches = "tight")

