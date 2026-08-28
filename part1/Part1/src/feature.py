import scipy.io
import numpy as np

data = scipy.io.loadmat('../images/MNISTmini.mat')

def printFeaturesShape(data):
    print(f"train_fea1 shape: {data['train_fea1'].shape}")
    print(f"train_gnd1 shape: {data['train_gnd1'].shape}")
    print(f"test_fea1 shape: {data['test_fea1'].shape}")
    print(f"test_gnd1 shape: {data['test_gnd1'].shape}")
    print(" ")

def printLabelCount(data):
    print(f"Number of training instances with label 5: {np.sum(data['train_gnd1'] == 5)}")
    print(f"Number of training instances with label 6: {np.sum(data['train_gnd1'] == 6)}")
    print(f"Number of testing instances with label 5: {np.sum(data['test_gnd1'] == 5)}")
    print(f"Number of testing instances with label 6: {np.sum(data['test_gnd1'] == 6)}")
    print("")

printFeaturesShape(data)
printLabelCount(data)

data['train_gnd1'] = data['train_gnd1'].flatten()
print(data['train_gnd1'].shape)