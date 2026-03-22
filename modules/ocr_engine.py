import cv2
import numpy as np
from PIL import Image
import easyocr

# Initialize EasyOCR reader (Downloads model weights on first run)
# 'en' indicates English alphanumeric reading.
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Failed to load EasyOCR: {e}")
    reader = None

def resize_and_clahe(plate_image):
    """
    we use CLAHE to improve contrast.
    """
    if isinstance(plate_image, Image.Image):
        img_np = np.array(plate_image)
    else:
        img_np = plate_image.copy()
        
    # Resize small plate crops to help EasyOCR 
    # Plates smaller than ~150px wide often fail OCR. Upscale by 2x using Lanczos.
    h, w = img_np.shape[:2]
    if w < 200:
        new_w = w * 2
        new_h = h * 2
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
    # Convert to Grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    # Apply CLAHE to improve contrast instead of harsh thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def extract_text(plate_image):
    """
    Runs EasyOCR on the aligned plate image.
    Uses an alphanumeric allowlist to prevent garbage characters
    and joins multiple text blocks if the plate has two lines.
    """
    if reader is None:
        return "OCR Model Failed to Load", 0.0
        
    # EasyOCR expects a NumPy array
    if isinstance(plate_image, Image.Image):
        img_np = np.array(plate_image)
    else:
        img_np = plate_image.copy()
        
    # Enforce alphanumeric allowlist to avoid reading random lines as symbols
    # Tune parameters for small, blurry text:
    # mag_ratio: Image magnification ratio
    # text_threshold / link_threshold: Lowering these helps pick up fainter text
    results = reader.readtext(
        img_np, 
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        mag_ratio=2,
        text_threshold=0.4,
        link_threshold=0.4,
        slope_ths=0.2
    )
    
    if len(results) == 0:
        return "No Text Detected", 0.0
        
    # Extract all text blocks and average the confidence
    texts = [res[1] for res in results]
    confidence = sum([res[2] for res in results]) / len(results)
    
    # Join the texts with a space (for multi-line plates)
    final_text = " ".join(texts)
    
    return final_text, confidence
