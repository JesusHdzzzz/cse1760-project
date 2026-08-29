from pathlib import Path 
import scipy.io 
import numpy as np

from sklearn.model_selection import train_test_split 

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

def split_data(X, y, train_size=1000, val_size=1000, test_size=1000, random_state=42): 
    total = train_size + val_size + test_size

    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=total,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_subset,
        y_subset,
        train_size=train_size,
        stratify=y_subset,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_size,
        test_size=test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

