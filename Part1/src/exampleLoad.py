import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Method 1A: Reshape and access individual images
def get_individual_images(images):
    """Convert batch of images to list of individual images"""
    individual_imgs = []
    for i in range(len(images)):
        individual_imgs.append(images[i])  # Each image is already 28x28
    return individual_imgs

# Get individual training images
train_images = get_individual_images(x_train)
test_images = get_individual_images(x_test)

# Display some images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axes[i//5, i%5]
    ax.imshow(train_images[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()