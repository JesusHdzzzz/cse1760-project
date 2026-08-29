from pathlib import Path 
import scipy.io 
import numpy as np

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "MNISTmini.mat" 
)

def load_mnist(): 
    mat = scipy.io.loadmat(DATA_PATH) 

    X = mat["train_fea1"]
    y = mat["train_gnd1"].flatten()

    return X, y

def filter_digits(X, y, digits=(5,6)): 
    mask = np.isin(y, digits) 

    return X[mask], y[mask]

def encode_binary_labels(y, positive_class=6): 
    return (y == positive_class).astype(int)

    
    