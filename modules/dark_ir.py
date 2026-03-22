import cv2
import numpy as np
from PIL import Image

def managing_contrast_and_brightness_mathematically(image, gamma=0.5, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Enhances a low-light image with a grayscale OCR-focused preprocessing chain.
    
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

    # Ensure single-channel grayscale processing regardless of input format.
    if len(img_np.shape) == 2:
        gray = img_np
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
        gray = img_np[:, :, 0]
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Gamma Correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    contrast_enhanced = clahe.apply(gamma_corrected)

    # 3. Bilateral filter
    denoised = cv2.bilateralFilter(contrast_enhanced, 11, 75, 75)

    # 4. Black-hat morphology for dark text/glare separation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

    # 5. Adaptive threshold for OCR-ready character emphasis
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Final cleanup: combine black-hat and threshold responses
    final_output = cv2.add(blackhat, thresh)

    # Keep downstream compatibility: detector/UI expect 3-channel images.
    final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_output_rgb)
