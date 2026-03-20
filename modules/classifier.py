import streamlit as st
from PIL import Image
import numpy as np

# We'll use a simple heuristic-based classifier for V1 (brightness/contrast variance)
# A deep learning classifier like MobileNetV2 can be swapped in later for more accuracy.

def classify_condition(image_np):
    """
    Analyzes the image to classify the weather/lighting condition.
    Returns: 'Night', 'Rain', 'Haze', or 'Clear'
    """
    gray = np.mean(image_np, axis=2)
    
    # 1. Check for Night/Low Light
    mean_brightness = np.mean(gray)
    if mean_brightness < 60:
        return 'Night'
        
    # 2. Check for Haze (Low contrast / narrow histogram)
    std_dev = np.std(gray)
    if std_dev < 35 and mean_brightness > 100:
        return 'Haze'
        
    # 3. Check for High Glare (Headlights shining directly)
    # High mean or very high localized pixels relative to standard scenes
    max_brightness = np.max(gray)
    if mean_brightness > 140 or (max_brightness > 240 and std_dev > 50):
        return 'Glare'
        
    # 4. Default to Clear (or Rain, since rain requires structural analysis)
    return 'Clear (or Rain)'
