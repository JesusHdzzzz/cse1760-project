from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and convert to grayscale
img = Image.open("../images/5Train.png").convert("L")
img_np = np.array(img)

H, W = img_np.shape
print(img_np.shape)
# The grid seems roughly ~64x64; infer exact size by inspecting approximate cell size
rows = 64
cols = 64
digit_h = H // rows
digit_w = W // cols

# Extract patches
patches = []
for r in range(rows):
    for c in range(cols):
        patch = img_np[r*digit_h:(r+1)*digit_h, c*digit_w:(c+1)*digit_w]
        patches.append(patch)

# Show first N digits
N = 100
grid = int(np.sqrt(N))

fig, axes = plt.subplots(grid, grid, figsize=(8, 8))
idx = 0
for r in range(grid):
    for c in range(grid):
        axes[r, c].imshow(patches[idx], cmap='gray')
        axes[r, c].axis('off')
        idx += 1

plt.tight_layout()
plt.show()