#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using the RPCA of sklearn:
# Source code: https://bsharchilev.github.io/RobustPCA/_modules/rpca/m_est_rpca.html
# Documentations: https://bsharchilev.github.io/RobustPCA/rpca.html

print('\nPlease be patient, method starting soon...')

import loss, rpca, copy, numpy as np

from itertools import accumulate
from dimreducelib import *
from readplotlib import *

def method2(filename, percentage_of_variance=85):
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
    
    ## Transform it using the sklearn Robust PCA ##
    
    huber_loss = loss.HuberLoss(delta=1) # the loss function
    
    num_components = X.T.shape[1]-1 # initialization, max-1
    rpca_transformer = rpca.MRobustPCA(num_components, huber_loss, max_iter=10000)
    print("\nPerforming sklearn RPCA in the data...\n")
    X_rpca = rpca_transformer.fit_transform(X.T) # shape (n_samples, n_features)
    
    eigenvals = rpca_transformer.explained_variance_

    ## Percentage graphically of variance the 20 first principal components cover ##
    elbow_method(eigenvals[:20]) # creates the plot and saves it as PDF

    # If there are less than 20 principal components, it doesn't matter;
    # then the plot will be about all components (<20),
    # i.e. meaning there is no error raised...

    ## Deciding how many components ##
    
    var_explained = rpca_transformer.explained_variance_ratio_*100
    arr = np.array(list(accumulate(var_explained, lambda x,y:x+y)),
                   dtype = str(var_explained.dtype))
    
    for val in np.nditer(arr):
        if val > PoV:
            tmp, = np.where(arr==val)
            num_of_components = tmp.item()
            break
            
    return num_of_components+1

file = 'liver-norm_3map.txt'

# Let's assume we wish to keep the 95% of variance:
print("\nNumber of principal components to achieve 95% of variance: ",
      method2(filename=file, percentage_of_variance=95), '\n')

