# src/utils_mnist.py
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

def load_mnist_mat(path, feature_key, label_key):
    mat = sio.loadmat(path)
    X = mat[feature_key]                # (60000, 784) OR (60000, 800)
    y = mat[label_key].flatten()        # (60000,)

    # Convert labels from 1..10 → 0..9
    if y.min() == 1:
        y = y - 1

    return X.astype(np.float32), y.astype(np.int64)


def train_val_split(X, y, train_size=55000, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_size,
        stratify=y,
        shuffle=True,
        random_state=random_state
    )
    return X_train, X_val, y_train, y_val