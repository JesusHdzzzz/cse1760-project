import cv2
import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt

img = cv2.imread("../images/5Train.png", cv2.IMREAD_GRAYSCALE)

tile_size = 15
blocks = view_as_blocks(img, block_shape=(tile_size, tile_size))

digits = blocks.reshape(-1, tile_size, tile_size)

print("Total digits:", len(digits))
print("Each digit shape:", digits[0].shape)

plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(digits[i], cmap='gray')
    plt.axis("off")
plt.show()