import scipy.io
import numpy as np

data = scipy.io.loadmat('../images/MNISTmini.mat')

def printFeaturesShape(data):
    print(f"{data['train_fea1'].shape}")
    print(f"{data['train_gnd1'].shape}")
    print(f"{data['test_fea1'].shape}")
    print(f"{data['test_gnd1'].shape}")



    
printFeaturesShape(data)