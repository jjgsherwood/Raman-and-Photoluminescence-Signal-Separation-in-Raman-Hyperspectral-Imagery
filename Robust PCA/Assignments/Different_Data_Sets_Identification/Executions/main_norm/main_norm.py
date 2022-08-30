#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from readplotlib import *

from rpcalib import *

file = 'liver-norm_3map.txt'

def main(file):
    """
    Specifications:
    
    Parameters
    ----------
    file_name: must be a string with the filename,
               
               Inner file format must be as follows:
               - it's 1st column (gap in the 1st row) corresponds to the Y-axis values,
               - it's 2nd column (gap in the 1st row) corresponds to the X-axis values,
               - it's 1st row (two gaps at start) corresponds to the Z-axis values
               - the rest values are the intensities for the X, Y, Z combinations
               - the values must be tab-separated
               
               If the inner format differs to the one described previously,
                  this function is not applicable for reading the file.
                  
               FYI: For more hardcore inner format checks, special functions called
                    validators exist. Best library for validators is jsonschema.
    Function
    --------
             - Represents the main function which does all the calls.
    """
    # Plotting the spectra (initial data):
    print('Saving the plot of the initial spectra in PDF...\n')
    plot_spectra(file)

    # Performing RPCA:
    data = get_intensities(file)
    rpca = R_pca(data)

    print('Fitting the RPCA with 10000 maximum iterations...\n')
    L, S = rpca.fit(max_iter=10000, iter_print=250)

    print('\nSaving the spectra after RPCA, each on a different plot...\n')
    
    # Plotting the spectra after RPCA:
    rpca.plot_fit(print_yrange = True)

    print('Saving the plot of the Low Rank of four random points...\n')
    rpca.plot_four_low_rank_randomly()

    print('Saving the spectra after RPCA in one plot...\n')
    rpca.plot_fit_one_plot()

    print('Saving all the Low Rank graphs in one plot...\n')
    rpca.plot_low_rank_only()

    print('Saving all the Sparse graphs in one plot...\n')
    rpca.plot_sparse_only()
        
    print('main did all the calls successfully!\n')

main(file)

