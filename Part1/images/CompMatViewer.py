import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

class MNISTViewer:
    def __init__(self, mat_file):
        # Load data
        self.mat_data = scipy.io.loadmat(mat_file)
        self.all_images = self.mat_data['train_fea1']
        self.all_labels = self.mat_data['train_gnd1'].flatten()
        
        # Convert uint8 labels to regular integers for easier handling
        self.all_labels = self.all_labels.astype(int)
        
        # Available labels - note: MNIST uses 1-10 where 10 represents 0
        self.available_labels = np.unique(self.all_labels)
        print(f"Available labels: {sorted(self.available_labels)}")
        print(f"Total images: {len(self.all_images)}")
        
        # Start with all images
        self.current_labels = list(self.available_labels)  # Show all initially
        self.filtered_images = self.all_images
        self.filtered_labels = self.all_labels
        self.current_idx = 0
        
        self.setup_ui()
        self.update_info_text()  # Call this after UI is setup
    
    def apply_filter(self, label_input=None):
        """Apply label filter based on user input"""
        if label_input is not None:
            try:
                if label_input.lower() == 'all':
                    self.current_labels = list(self.available_labels)
                else:
                    # Parse comma-separated labels
                    labels = [int(x.strip()) for x in label_input.split(',')]
                    # Validate labels
                    valid_labels = [l for l in labels if l in self.available_labels]
                    if valid_labels:
                        self.current_labels = valid_labels
                    else:
                        print("No valid labels found. Showing all images.")
                        self.current_labels = list(self.available_labels)
            except ValueError:
                print("Invalid input. Use numbers separated by commas, or 'all'.")
                return
        
        # Apply filter
        mask = np.isin(self.all_labels, self.current_labels)
        self.filtered_images = self.all_images[mask]
        self.filtered_labels = self.all_labels[mask]
        self.current_idx = 0
        
        print(f"Now showing {len(self.filtered_images)} images with labels: {self.current_labels}")
        self.update_slider_range()
        self.update_display()
    
    def setup_ui(self):
        # Create figure with larger size for better layout
        self.fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.95)
        
        # Main image display
        self.ax_image = plt.axes([0.35, 0.4, 0.6, 0.55])
        
        # Control panels
        self.setup_controls()
        
        # Information display (create this first)
        self.ax_info = plt.axes([0.35, 0.25, 0.6, 0.1])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.02, 0.5, '', va='center', fontsize=10)
        
        # Initial display
        self.update_display()
    
    def setup_controls(self):
        # Label selection section
        ax_labels_title = plt.axes([0.05, 0.85, 0.25, 0.05])
        ax_labels_title.axis('off')
        ax_labels_title.text(0.5, 0.5, 'Filter by Label', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
        
        # Quick selection buttons for common filters
        ax_quick_buttons = plt.axes([0.05, 0.75, 0.25, 0.08])
        # Note: MNIST labels are 1-10 where 10 represents 0
        quick_labels = ['All', '1-5', '6-10', 'Even', 'Odd', '6 & 7']
        self.quick_selector = RadioButtons(ax_quick_buttons, quick_labels, active=0)
        self.quick_selector.on_clicked(self.quick_filter)
        
        # Custom label input
        ax_textbox = plt.axes([0.05, 0.65, 0.25, 0.04])
        self.label_input = TextBox(ax_textbox, 'Custom labels:\n(comma-separated)', 
                                 initial="all")
        self.label_input.on_submit(self.custom_filter)
        
        # Individual label buttons - convert to strings for display
        ax_individual = plt.axes([0.05, 0.4, 0.25, 0.2])
        individual_labels = [f'{i}' for i in self.available_labels]
        self.individual_selector = RadioButtons(ax_individual, individual_labels)
        self.individual_selector.on_clicked(self.individual_filter)
        
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
        """Handle quick filter selections"""
        # Note: MNIST uses 1-10 where 10 represents 0
        if label == 'All':
            self.apply_filter('all')
        elif label == '1-5':
            self.apply_filter('1,2,3,4,5')
        elif label == '6-10':
            self.apply_filter('6,7,8,9,10')
        elif label == 'Even':
            self.apply_filter('2,4,6,8,10')
        elif label == 'Odd':
            self.apply_filter('1,3,5,7,9')
        elif label == '6 & 7':
            self.apply_filter('6,7')
    
    def custom_filter(self, text):
        """Handle custom label input"""
        self.apply_filter(text)
    
    def individual_filter(self, label):
        """Handle individual label selection"""
        self.apply_filter(label)
    
    def update_slider_range(self):
        """Update slider range based on filtered images"""
        max_val = max(0, len(self.filtered_images) - 1)
        self.slider.valmax = max_val
        self.slider.ax.set_xlim(self.slider.valmin, max_val)
        if self.current_idx > max_val:
            self.current_idx = max_val
        self.slider.set_val(self.current_idx)
    
    def update_info_text(self):
        """Update the information display"""
        info = f"Displaying: {len(self.filtered_images)} images\n"
        info += f"Labels: {self.current_labels}\n"
        if len(self.filtered_images) > 0:
            info += f"Current: {self.current_idx + 1}/{len(self.filtered_images)} "
            info += f"(Label: {self.filtered_labels[self.current_idx]})"
        else:
            info += "Current: No images"
        self.info_text.set_text(info)
    
    def update_display(self):
        """Update the main image display"""
        self.ax_image.clear()
        
        if len(self.filtered_images) == 0:
            self.ax_image.text(0.5, 0.5, 'No images to display\nSelect different labels', 
                             ha='center', va='center', fontsize=14,
                             transform=self.ax_image.transAxes)
            self.ax_image.set_title('No Images Available', fontsize=16, pad=20)
        else:
            # Display current image
            img_data = self.filtered_images[self.current_idx]
            
            # Reshape based on image size
            if img_data.shape[0] == 100:  # MNISTmini (10x10)
                img = img_data.reshape(10, 10)
                interpolation = 'nearest'  # Better for small images
            else:  # Regular MNIST (28x28)
                img = img_data.reshape(28, 28)
                interpolation = 'antialiased'
            
            self.ax_image.imshow(img, cmap='gray', interpolation=interpolation)
            
            # Title with current position
            title = f'Image {self.current_idx + 1}/{len(self.filtered_images)} '
            title += f'(Label: {self.filtered_labels[self.current_idx]})'
            self.ax_image.set_title(title, fontsize=14, pad=20)
        
        self.ax_image.axis('off')
        self.update_info_text()
        self.fig.canvas.draw()
    
    def slider_update(self, val):
        """Handle slider changes"""
        if len(self.filtered_images) > 0:
            self.current_idx = int(val)
            self.update_display()
    
    # Navigation methods
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

# Main execution
if __name__ == "__main__":
    print("Loading MNISTmini.mat...")
    print("Note: MNIST labels 1-10 where 10 represents digit 0")
    viewer = MNISTViewer('MNISTmini.mat')
    plt.show()