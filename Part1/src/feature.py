import scipy.io
import numpy as np

data = scipy.io.loadmat('../images/MNISTmini.mat')

def printFeaturesShape(data):
    for key in data.keys():
        print(f"{key}: {data[key].shape}")

    
        