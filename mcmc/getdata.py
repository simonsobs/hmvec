import numpy as np

def covariance():
    filename_cov = '../data/toy_cov.npy'

    return np.load(filename_cov)

def datapoints():
    filename_data = '../data/toy_data.npy'

    return np.load(filename_data)

def freq():
    return np.array([545e9], dtype=np.double) 
    