import cv2
import numpy as np
from PIL import Image

def remove_rain(image, kernel_size=(5, 5), threshold=50):
    """
    Removes rain streaks from an image using a Morphological Top-Hat transform.
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image with rain streaks.
        kernel_size (tuple): The size of the structuring element. Larger kernels
                             detect thicker/longer streaks.
        threshold (int): The intensity threshold to isolate rain streaks from the
                         top-hat result (0-255).
                         
    Returns:
        PIL.Image: The processed (derained) image.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    is_color = len(img_np.shape) == 3
    
    # Work with grayscale for thresholding the streaks
    if is_color:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
        
    # Create the structuring element (elliptical kernel is often good for rain)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # Apply a Top-Hat transform (Isolates bright objects smaller than the kernel)
    # Rain streaks are essentially bright, thin artifacts against the background.
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Threshold the top_hat result to create a mask of the rain streaks
    _, streak_mask = cv2.threshold(top_hat, threshold, 255, cv2.THRESH_BINARY)
    
    # We want to fill in the areas covered by rain. Inpainting is perfect for this.
    # cv2.INPAINT_TELEA is generally fast and looks smooth for thin streaks.
    # Convert RGB to BGR for standard cv2 processing
    if is_color:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        derained_bgr = cv2.inpaint(img_bgr, streak_mask, 3, cv2.INPAINT_TELEA)
        derained = cv2.cvtColor(derained_bgr, cv2.COLOR_BGR2RGB)
    else:
        derained = cv2.inpaint(img_np, streak_mask, 3, cv2.INPAINT_TELEA)
        
    return Image.fromarray(derained)
