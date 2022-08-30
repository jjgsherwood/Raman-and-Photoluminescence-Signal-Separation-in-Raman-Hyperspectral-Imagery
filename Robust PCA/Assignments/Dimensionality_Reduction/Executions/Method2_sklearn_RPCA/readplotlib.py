#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, matplotlib.pyplot as plt, pandas as pd, os

def read_data(file_name):
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
    Returns
    -------
            - all datas in pandas dataframe format
            - X-axis values in a list
            - Y-axis values in a list
            - Z-axis values in a list       
    """
    # Some exception handling:
    if not isinstance(file_name, str):
        raise Exception('Wrong input. Please read the documentation.\n\nRecommended way to do it: print(read_data.__doc__)')
    
    # Reading the file:
    df = pd.read_csv(file_name, sep = '\t')
    
    X_axis_values = [round(i, 5) for i in df[df.columns[1]].to_list()]
    Y_axis_values = [round(i, 5) for i in df[df.columns[0]].to_list()]
    
    Z_axis_values = [round(i, 5) for i in list(map(float, df.columns[2:].to_list()))]
    
    df.columns.name = 'X-axis'
    df.rename(columns = {df.columns[1]:'X-axis'}, inplace = True)
    df.rename(columns = {df.columns[0]:'Y-axis'}, inplace = True)
    df.index = X_axis_values
    df.rename(columns = {df.columns[1]:'Z-axis:'}, inplace = True)
    df['Z-axis:'] = ''
    
    return df, X_axis_values, Y_axis_values, Z_axis_values

def plot_spectra(file_name):
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
            - Plots all the spectra.
            - Saves the plot created in a PDF format in the working directory,
              it's called Spectra.pdf
    """
    # Reading the file and saving the data:
    data, *_ = read_data(file_name)
    
    # The intensities in pandas dataframe fromat:
    data_intensities = data.drop(data.columns[:2], axis = 1).T
    
    data_intensities.index.name = data_intensities.columns.name = None
    data_intensities.columns = range(len(data.index))

    data_intensities.plot(figsize = (15, 12), legend = False)

    plt.title('All Spectra in One Graph', fontsize = 30)
    
    # dpi: dots per inch --> is the resolution
    plt.savefig('Spectra.pdf', dpi = 2500)
    
def get_intensities(file_name):
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
    Returns
    -------
            - An array (numpy array) with all values corresponding to intensities.
    """
    # Reading the file and saving the data:
    data, *_ = read_data(file_name)
    
    # The intensities in pandas dataframe fromat:
    data_intensities = data.drop(data.columns[:2], axis = 1).T
    
    return data_intensities.T.values

