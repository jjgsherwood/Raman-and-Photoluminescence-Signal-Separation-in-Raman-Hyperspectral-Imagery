import numpy as np

def MAPE(x, y):
    return np.mean(np.abs(1 - y/x))

def RMSPE(x, y):
    return np.mean(np.sqrt(np.mean((1 - y/x)**2, 1)))

def MSE(x,y):
    return np.mean((x-y)**2)

STRING_TO_FUNCTION = {
    "MAPE": MAPE,
    "RMSPE": RMSPE
}
