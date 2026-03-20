import cv2
import numpy as np
from PIL import Image

def enhance_low_light(image, gamma=0.6, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances a low-light (Dark IR) image using a combination of Gamma Correction
    and Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image.
        gamma (float): Gamma value for correction (< 1.0 brightens the image).
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple): Grid size for histogram equalization.
        
    Returns:
        PIL.Image: The enhanced image.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    # Ensure image is in RGB format for processing (OpenCV uses BGR by default, but
    # assuming Streamlit/PIL provides RGB, we'll work with standard channels).
    if len(img_np.shape) == 2:
        # Grayscale image
        is_color = False
    else:
        is_color = True
        # Convert RGB to LAB color space to apply CLAHE only to the Lightness channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
    # --- Step 1: Gamma Correction ---
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    if is_color:
        l_channel = cv2.LUT(l_channel, table)
    else:
        img_np = cv2.LUT(img_np, table)

    # --- Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if is_color:
        cl = clahe.apply(l_channel)
        # Merge the CLAHE enhanced L-channel back with A and B channels
        limg = cv2.merge((cl, a, b))
        # Convert back to RGB
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        enhanced_img = clahe.apply(img_np)
        
    return Image.fromarray(enhanced_img)
