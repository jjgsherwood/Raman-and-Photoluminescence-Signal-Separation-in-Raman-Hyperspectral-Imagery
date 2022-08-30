#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Method (the how to use RPCA for dimensionality reduction) found in the link below:
# https://stackoverflow.com/questions/40721260/how-to-use-robust-pca-output-as-principal-component-eigenvectors-from-traditio

print('\nPlease be patient, method starting soon...')

from dimreducelib import *
from readplotlib import *
from rpcalib import *

import copy

def method1(filename, percentage_of_variance=85):
    """
    ==============
    Specifications
    ==============
    
    Parameters
    ----------
    filename: the input file
    percentage_of_variance: must be numerical, domain: [0, 100] (default is 85%)
    
    Function
    --------
            - Performs RPCA (Robust PCA), then PCA in the L matrix result of the RPCA.
            - Then performs the elbow method for deciding the number of components for
              the input percentage of variance explained (percentage_of_variance).
            - Saves the plot of the elbow method for the first 20 principal components
              in a PDF format file, the file created is called Inspect_PCAs.pdf
    Returns
    -------
            - number of components that suffice to achieve the percentage_of_variance
    """
    # Get the intensities as array:
    X = get_intensities(filename)
    
    PoV = copy.deepcopy(percentage_of_variance)
    
    # Exception handling:
    if isinstance(PoV, int) or isinstance(PoV, float):
        pass
    else:
        raise Exception("percentage_of_variance must be numerical (float or int)")
        
    if 0 <= PoV <= 100:
        pass
    else:
        raise Exception("percentage_of_variance must be a numerical value in domain [0, 100]\n\n Please call again properly!")

    # R_pca class initialization:
    rpca = R_pca(X.T)
    print("\nPerforming RPCA in the data...\n")
    L, S = rpca.fit(max_iter=10000, iter_print=250)
    print("\nPerforming PCA in the L matrix result of RPCA...\n")
    rcomp, revals, revecs = pca(L)
    # print("Normalised robust eigenvalues: %s" % (revals/np.sum(revals),))
    
    # Truthness check for the eigenvalues result of the pca() function:
    if np.all(np.equal(np.sort(revals)[::-1], revals)):
        pass
    else:
        print('Failure: the PCA function did not return sorted the eigenvalues...\n')

    ## What percentage of variance the 20 first principal components cover ##
    elbow_method(revals[:20]) # creates the plot and saves it as PDF

    ## Deciding how many components ##
    for i in range(revals.shape[0]):

        if np.sum((revals[:i]/np.sum(revals))*100) > PoV:
            num_of_components = i+1
            break
            
    return num_of_components

file = 'liver-norm_3map.txt'

# Let's assume we wish to keep the 95% of variance:
print("\nNumber of principal components to achieve 95% of variance: ",
      method1(filename=file, percentage_of_variance=95), '\n')

