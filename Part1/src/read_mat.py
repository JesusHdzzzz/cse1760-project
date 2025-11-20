import scipy.io
import numpy as np

mat = scipy.io.loadmat("images/MNISTmini.mat")

y = mat['train_gnd1'].flatten()

unique, counts = np.unique(y, return_counts=True)

print("Unique labels in ALL 60000 examples:")
for u, c in zip(unique, counts):
    print(f"Label {u}: {c} samples")

print("\nMin label:", y.min())
print("Max label:", y.max())
