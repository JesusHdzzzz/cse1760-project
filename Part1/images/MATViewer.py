import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

# Load the .mat file
mat_data = scipy.io.loadmat('MNISTmini.mat')

# Get the data (adjust variable names as needed)
images = mat_data['train_fea1']  # or whatever the actual name is
labels = mat_data['train_gnd1']  # or whatever the actual name is

# Flatten labels if they're in 2D array format
labels = labels.flatten()

# Filter for images with label 5 or 6
mask_5_or_6 = (labels == 5) | (labels == 6)
filtered_images = images[mask_5_or_6]
filtered_labels = labels[mask_5_or_6]

print(f"Original: {len(images)} images")
print(f"Filtered (5 or 6): {len(filtered_images)} images")
print(f"Breakdown - 5s: {np.sum(labels == 5)}, 6s: {np.sum(labels == 6)}")

# Interactive viewer for the filtered images
class SimpleFilteredViewer:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.current_idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # Slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Image', 0, len(images)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)
        
        # Navigation buttons
        ax_prev = plt.axes([0.2, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.previous)
        self.btn_next.on_clicked(self.next)
        
        self.update_display()
    
    def update_display(self):
        self.ax.clear()
        img_data = self.images[self.current_idx]
        
        # Reshape for MNISTmini (10x10)
        img = img_data.reshape(10, 10)
        
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(f'Image {self.current_idx + 1}/{len(self.images)} - Label: {self.labels[self.current_idx]}')
        self.ax.axis('off')
        plt.draw()
    
    def update(self, val):
        self.current_idx = int(val)
        self.update_display()
    
    def previous(self, event):
        self.current_idx = max(0, self.current_idx - 1)
        self.slider.set_val(self.current_idx)
    
    def next(self, event):
        self.current_idx = min(len(self.images)-1, self.current_idx + 1)
        self.slider.set_val(self.current_idx)

# Create the viewer with your filtered data
print("\nLaunching interactive viewer for filtered images (labels 5 and 6)...")
viewer = SimpleFilteredViewer(filtered_images, filtered_labels)
plt.show()