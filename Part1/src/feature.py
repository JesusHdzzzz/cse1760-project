import scipy.io
import numpy as np

data = scipy.io.loadmat('../images/MNISTmini.mat')

def printFeaturesShape(data):
    print(f"train_fea1 shape: {data['train_fea1'].shape}")
    print(f"train_gnd1 shape: {data['train_gnd1'].shape}")
    print(f"test_fea1 shape: {data['test_fea1'].shape}")
    print(f"test_gnd1 shape: {data['test_gnd1'].shape}")



    
printFeaturesShape(data)