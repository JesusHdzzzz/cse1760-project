import cv2
import numpy as np
import os

# Load the image
img = cv2.imread('../images/5Train.png', cv2.IMREAD_GRAYSCALE)

# Optional: Invert if needed (make digits white on black)
# If your image is already white-on-black, skip this.
# img = cv2.bitwise_not(img)

# Define the size of each "5" (you need to measure or estimate this)
# For example, if each "5" is roughly 20x30 pixels:
tile_height = 30
tile_width = 20

# Calculate number of rows and columns
rows = img.shape[0] // tile_height
cols = img.shape[1] // tile_width

# Create output directory
os.makedirs('output_5s', exist_ok=True)

# Split and save each tile
for i in range(rows):
    for j in range(cols):
        # Crop the tile
        y_start = i * tile_height
        x_start = j * tile_width
        tile = img[y_start:y_start+tile_height, x_start:x_start+tile_width]

        # Save the tile
        filename = f'output_5s/5_{i}_{j}.png'
        cv2.imwrite(filename, tile)

print(f"Split into {rows * cols} individual '5' images.")