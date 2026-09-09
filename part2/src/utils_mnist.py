from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def convert_mnist_labels(raw_labels):
    """Convert the course MAT encoding (10 means digit 0) to digits 0..9."""
    labels = np.asarray(raw_labels).reshape(-1)
    unique = set(np.unique(labels).tolist())
    matlab_labels = set(range(1, 11))
    digit_labels = set(range(10))

    if unique.issubset(matlab_labels) and 10 in unique:
        labels = np.where(labels == 10, 0, labels)
    elif not unique.issubset(digit_labels):
        raise ValueError(f"Unexpected MNIST labels: {sorted(unique)}")

    converted = labels.astype(np.int64)
    if not set(np.unique(converted).tolist()).issubset(digit_labels):
        raise AssertionError("Converted labels must be decimal digits 0 through 9")
    return converted


def load_mnist_mat(path, feature_key, label_key):
    """Load MNIST features and labels from a MATLAB .mat file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MNIST dataset not found: {path}")
    mat = sio.loadmat(path)
    missing = {feature_key, label_key}.difference(mat)
    if missing:
        raise KeyError(f"Missing MAT keys in {path}: {sorted(missing)}")
    X = mat[feature_key]
    y = convert_mnist_labels(mat[label_key])
    if len(X) != len(y):
        raise ValueError(f"Feature/label row mismatch in {path}: {len(X)} != {len(y)}")
    return X.astype(np.float32), y


def train_val_split(
    X,
    y,
    train_size=55000,
    random_state=42,
):
    """Create a reproducible stratified train/validation split."""
    return train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=y,
        shuffle=True,
        random_state=random_state,
    )
