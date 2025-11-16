import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons

class MNISTViewer:
    def __init__(self, mat_file):
        # Load data
        self.mat_data = scipy.io.loadmat(mat_file)
        self.all_images = self.mat_data['train_fea1']
        self.all_labels = self.mat_data['train_gnd1'].flatten().astype(int)
        
        # Image transformation flags
        self.flip_horizontal = False
        self.flip_vertical = False
        self.rotate_180 = False
        
        # Debug info
        print("=== DATA INFO ===")
        print(f"Total images: {len(self.all_images)}")
        print(f"Image shape: {self.all_images[0].shape} -> {int(np.sqrt(self.all_images[0].shape[0]))}x{int(np.sqrt(self.all_images[0].shape[0]))}")
        print(f"Unique labels: {sorted(np.unique(self.all_labels))}")
        
        # Normalize if needed
        if self.all_images.max() > 1.0:
            self.all_images = self.all_images / 255.0
        
        # Available labels
        self.available_labels = np.unique(self.all_labels)
        
        # Start with all images
        self.current_labels = list(self.available_labels)
        self.filtered_images = self.all_images
        self.filtered_labels = self.all_labels
        self.current_idx = 0
        
        self.setup_ui()
        self.update_info_text()
    
    def transform_image(self, img):
        """Apply transformations to the image"""
        img_2d = img.reshape(10, 10)  # MNISTmini is 10x10
        
        if self.flip_horizontal:
            img_2d = np.fliplr(img_2d)
        if self.flip_vertical:
            img_2d = np.flipud(img_2d)
        if self.rotate_180:
            img_2d = np.rot90(img_2d, 2)
            
        return img_2d
    
    def apply_filter(self, label_input=None):
        """Apply label filter based on user input"""
        if label_input is not None:
            try:
                if label_input.lower() == 'all':
                    self.current_labels = list(self.available_labels)
                else:
                    labels = [int(x.strip()) for x in label_input.split(',')]
                    valid_labels = [l for l in labels if l in self.available_labels]
                    if valid_labels:
                        self.current_labels = valid_labels
                    else:
                        self.current_labels = list(self.available_labels)
            except ValueError:
                return
        
        # Apply filter
        mask = np.isin(self.all_labels, self.current_labels)
        self.filtered_images = self.all_images[mask]
        self.filtered_labels = self.all_labels[mask]
        self.current_idx = 0
        
        self.update_slider_range()
        self.update_display()
    
    def setup_ui(self):
        # Create figure with larger size
        self.fig = plt.figure(figsize=(14, 8))
        plt.subplots_adjust(left=0.25, bottom=0.3, right=0.95, top=0.95)
        
        # Main image display
        self.ax_image = plt.axes([0.35, 0.4, 0.6, 0.55])
        
        # Control panels
        self.setup_controls()
        
        # Information display
        self.ax_info = plt.axes([0.35, 0.25, 0.6, 0.1])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.02, 0.5, '', va='center', fontsize=10)
        
        # Initial display
        self.update_display()
    
    def setup_controls(self):
        # Label selection section
        ax_labels_title = plt.axes([0.05, 0.85, 0.2, 0.05])
        ax_labels_title.axis('off')
        ax_labels_title.text(0.5, 0.5, 'Filter by Label', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
        
        # Quick selection buttons
        ax_quick_buttons = plt.axes([0.05, 0.75, 0.2, 0.08])
        quick_labels = ['All', '1-5', '6-9,10', 'Even', 'Odd', '5 & 6']
        self.quick_selector = RadioButtons(ax_quick_buttons, quick_labels, active=0)
        self.quick_selector.on_clicked(self.quick_filter)
        
        # Custom label input
        ax_textbox = plt.axes([0.05, 0.65, 0.2, 0.04])
        self.label_input = TextBox(ax_textbox, 'Custom labels:', initial="all")
        self.label_input.on_submit(self.custom_filter)
        
        # Individual label buttons
        ax_individual = plt.axes([0.05, 0.45, 0.2, 0.15])
        individual_labels = [f'{i}' for i in sorted(self.available_labels)]
        self.individual_selector = RadioButtons(ax_individual, individual_labels)
        self.individual_selector.on_clicked(self.individual_filter)
        
        # Image transformation controls
        ax_transform_title = plt.axes([0.05, 0.35, 0.2, 0.05])
        ax_transform_title.axis('off')
        ax_transform_title.text(0.5, 0.5, 'Image Transform', ha='center', va='center', 
                              fontsize=12, fontweight='bold')
        
        ax_transform = plt.axes([0.05, 0.2, 0.2, 0.12])
        transform_labels = ['Flip Horizontal', 'Flip Vertical', 'Rotate 180°']
        self.transform_selector = CheckButtons(ax_transform, transform_labels)
        self.transform_selector.on_clicked(self.transform_changed)
        
        # Navigation controls
        self.setup_navigation()
    
    def setup_navigation(self):
        # Slider
        ax_slider = plt.axes([0.35, 0.15, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Image', 0, max(0, len(self.filtered_images)-1), 
                           valinit=0, valstep=1)
        self.slider.on_changed(self.slider_update)
        
        # Navigation buttons
        ax_first = plt.axes([0.35, 0.05, 0.08, 0.04])
        ax_prev = plt.axes([0.45, 0.05, 0.08, 0.04])
        ax_next = plt.axes([0.55, 0.05, 0.08, 0.04])
        ax_last = plt.axes([0.65, 0.05, 0.08, 0.04])
        ax_random = plt.axes([0.75, 0.05, 0.1, 0.04])
        
        self.btn_first = Button(ax_first, 'First')
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_last = Button(ax_last, 'Last')
        self.btn_random = Button(ax_random, 'Random')
        
        self.btn_first.on_clicked(self.first_image)
        self.btn_prev.on_clicked(self.previous_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_last.on_clicked(self.last_image)
        self.btn_random.on_clicked(self.random_image)
    
    def quick_filter(self, label):
        if label == 'All':
            self.apply_filter('all')
        elif label == '1-5':
            self.apply_filter('1,2,3,4,5')
        elif label == '6-9,10':
            self.apply_filter('6,7,8,9,10')
        elif label == 'Even':
            self.apply_filter('2,4,6,8,10')
        elif label == 'Odd':
            self.apply_filter('1,3,5,7,9')
        elif label == '5 & 6':
            self.apply_filter('5,6')
    
    def custom_filter(self, text):
        self.apply_filter(text)
    
    def individual_filter(self, label):
        self.apply_filter(label)
    
    def transform_changed(self, label):
        """Handle image transformation changes"""
        if label == 'Flip Horizontal':
            self.flip_horizontal = not self.flip_horizontal
        elif label == 'Flip Vertical':
            self.flip_vertical = not self.flip_vertical
        elif label == 'Rotate 180°':
            self.rotate_180 = not self.rotate_180
        
        print(f"Transform - Horizontal: {self.flip_horizontal}, Vertical: {self.flip_vertical}, Rotate: {self.rotate_180}")
        self.update_display()
    
    def update_slider_range(self):
        max_val = max(0, len(self.filtered_images) - 1)
        self.slider.valmax = max_val
        self.slider.ax.set_xlim(self.slider.valmin, max_val)
        if self.current_idx > max_val:
            self.current_idx = max_val
        self.slider.set_val(self.current_idx)
    
    def update_info_text(self):
        info = f"Displaying: {len(self.filtered_images)} images\n"
        info += f"Labels: {self.current_labels}\n"
        if len(self.filtered_images) > 0:
            info += f"Current: {self.current_idx + 1}/{len(self.filtered_images)} "
            info += f"(Label: {self.filtered_labels[self.current_idx]})"
        self.info_text.set_text(info)
    
    def update_display(self):
        self.ax_image.clear()
        
        if len(self.filtered_images) == 0:
            self.ax_image.text(0.5, 0.5, 'No images to display', 
                             ha='center', va='center', fontsize=14,
                             transform=self.ax_image.transAxes)
        else:
            # Get and transform image
            img_data = self.filtered_images[self.current_idx]
            img_transformed = self.transform_image(img_data)
            
            # Display with nearest neighbor interpolation for crisp pixels
            self.ax_image.imshow(img_transformed, cmap='gray', interpolation='nearest')
            
            # Add transformation info to title
            transform_info = []
            if self.flip_horizontal:
                transform_info.append("H-flip")
            if self.flip_vertical:
                transform_info.append("V-flip")
            if self.rotate_180:
                transform_info.append("180°")
            
            transform_str = " + ".join(transform_info) if transform_info else "Original"
            
            title = f'Image {self.current_idx + 1}/{len(self.filtered_images)} '
            title += f'(Label: {self.filtered_labels[self.current_idx]}) - {transform_str}'
            self.ax_image.set_title(title, fontsize=12, pad=20)
        
        self.ax_image.axis('off')
        self.update_info_text()
        self.fig.canvas.draw()
    
    def slider_update(self, val):
        if len(self.filtered_images) > 0:
            self.current_idx = int(val)
            self.update_display()
    
    def first_image(self, event):
        if len(self.filtered_images) > 0:
            self.current_idx = 0
            self.slider.set_val(self.current_idx)
    
    def previous_image(self, event):
        if len(self.filtered_images) > 0:
            self.current_idx = max(0, self.current_idx - 1)
            self.slider.set_val(self.current_idx)
    
    def next_image(self, event):
        if len(self.filtered_images) > 0:
            self.current_idx = min(len(self.filtered_images)-1, self.current_idx + 1)
            self.slider.set_val(self.current_idx)
    
    def last_image(self, event):
        if len(self.filtered_images) > 0:
            self.current_idx = len(self.filtered_images) - 1
            self.slider.set_val(self.current_idx)
    
    def random_image(self, event):
        if len(self.filtered_images) > 0:
            self.current_idx = np.random.randint(0, len(self.filtered_images))
            self.slider.set_val(self.current_idx)

# Quick test to see original vs transformed
def test_transformations():
    """Test different transformations to find the correct orientation"""
    mat_data = scipy.io.loadmat('MNISTmini.mat')
    images = mat_data['train_fea1']
    labels = mat_data['train_gnd1'].flatten().astype(int)
    
    if images.max() > 1.0:
        images = images / 255.0
    
    # Find some clear examples of different digits
    sample_indices = []
    for digit in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        indices = np.where(labels == digit)[0]
        if len(indices) > 0:
            sample_indices.append(indices[0])
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    transformations = [
        ('Original', lambda x: x.reshape(10, 10)),
        ('Flip Horizontal', lambda x: np.fliplr(x.reshape(10, 10))),
        ('Flip Vertical', lambda x: np.flipud(x.reshape(10, 10))),
        ('Rotate 180°', lambda x: np.rot90(x.reshape(10, 10), 2)),
        ('Both Flips', lambda x: np.flipud(np.fliplr(x.reshape(10, 10))))
    ]
    
    for i, idx in enumerate(sample_indices[:5]):  # Show first 5 digits
        for j, (transform_name, transform_fn) in enumerate(transformations):
            ax = axes[j, i]
            img = transform_fn(images[idx])
            ax.imshow(img, cmap='gray', interpolation='nearest')
            ax.set_title(f'Label: {labels[idx]}\n{transform_name}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading MNISTmini.mat...")
    
    # Uncomment to test transformations first
    # print("Testing different image transformations...")
    # test_transformations()
    
    viewer = MNISTViewer('MNISTmini.mat')
    plt.show()