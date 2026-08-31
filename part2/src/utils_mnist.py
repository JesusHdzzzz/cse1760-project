import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def load_mnist_mat(path, feature_key, label_key):
    """Load MNIST features and labels from a MATLAB .mat file."""
    mat = sio.loadmat(path)

    X = mat[feature_key]
    y = mat[label_key].flatten()

    # Accept either 0-9 labels or MATLAB-style 1-10 labels.
    if y.min() == 1 and y.max() == 10:
        y = y - 1
    elif y.min() == 0 and y.max() == 9:
        pass
    else:
        raise ValueError(
            f"Unexpected label range: {y.min()} to {y.max()}"
        )

    return X.astype(np.float32), y.astype(np.int64)


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