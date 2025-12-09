# src/inspect_mnist_mat.py
import scipy.io as sio
import numpy as np

def summarize_mat(path, name):
    print(f"=== {name} ===")
    mat = sio.loadmat(path)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            print(f"{k}: type={type(v)}, shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}")
    print()

if __name__ == "__main__":
    summarize_mat("images/MNIST.mat", "Pixel features")
    summarize_mat("images/MNIST-LeNet5.mat", "LeNet5 features")
