import numpy as np

def MAPE(x, y):
    return np.mean(np.abs(1 - y/x))

def RMSPE(x, y):
    return np.mean(np.sqrt(np.mean((1 - y/x)**2, 1)))

def MSE(x,y):
    return np.mean((x-y)**2)

def MSGE(x):
    return np.mean((x[1:] - x[:-1])**2)

def TMSGE(x, y):
    return np.mean(((x[1:] - x[:-1]) - (y[1:] - y[:-1]))**2)

STRING_TO_FUNCTION = {
    "MAPE": MAPE,
    "RMSPE": RMSPE,
    "MSE": MSE,
    "MSGE": MSGE,
    "TMSGE": TMSGE
}
