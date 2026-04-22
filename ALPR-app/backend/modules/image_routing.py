import cv2
import numpy as np

def analyze_image_conditions(cv_img):
    """
    Analyzes the OpenCV image using mathematical heuristics and returns 
    the highly optimized restoration module to use: 
    'darkir', 'dehaze', 'derain', or 'normal'
    
    This completely bypasses the need for an external MobileNet classifier.
    """
    if cv_img is None:
        return 'normal'
        
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    
    # 1. Low Light Classifier (DarkIR)
    # Checks if overall image intensity is deeply under-exposed
    if brightness < 60:
        return 'darkir'

    # 2. Fog/Haze Classifier (DeHaze) using Dark Channel Prior (DCP)
    # DCP evaluates the minimum pixel value across RGB patches.
    # In a clear image, natural objects have colors, so at least one RGB channel is typically low.
    # If the ENTIRE dark channel is mathematically bright on average, the atmosphere is washing it out.
    kernel_size = 15
    min_pool = cv2.erode(cv_img, np.ones((kernel_size, kernel_size), np.uint8))
    dark_channel = np.min(min_pool, axis=2)
    avg_dcp = float(dark_channel.mean())
    
    # Typical clear photos have an avg_dcp between 15-40. Thick uniform haze pushes it over 100+.
    if avg_dcp > 110 and brightness > 80:
        return 'dehaze'

    # 3. Rain Classifier (DeRain) using Edge Variance
    # Rain creates extremely sharp, high-frequency vertical/diagonal streaks.
    # We analyze Sobel X (vertical gradients) vs Sobel Y (horizontal gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = np.var(sobel_x)
    var_y = np.var(sobel_y)
    
    # If the variance in the X gradient is massively higher than Y, we have vertical streak noise
    if var_y > 0 and (var_x / var_y) > 1.5: 
        return 'derain'

    # 4. Fallback
    return 'normal'
