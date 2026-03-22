from modules.dark_ir import managing_contrast_and_brightness_mathematically
from modules.derain import remove_rain
# Placeholder for dehaze module, importing here to show architecture 
# from modules.dehaze import remove_haze

def route_and_restore(image, condition):
    """
    Routes the image to the appropriate restoration module based on the classified condition.
    """
    restored = managing_contrast_and_brightness_mathematically(image, gamma=1.5, clip_limit=3.0)
    return restored, "Applied Low-Light Enhancement"

    '''
    
    if condition == 'Night':
        # Apply Zero-DCE or our OpenCV Dark IR equivalent
        # Gamma > 1.0 brightens the image in standard 1/gamma correction
        restored = enhance_low_light(image, gamma=1.5, clip_limit=3.0)
        return restored, "Applied Low-Light Enhancement"
        
    elif condition == 'Glare':
        # For Glare, DO NOT darken everything globally (which hides the dark car).
        # Use neutral gamma (=1.0) and let CLAHE equalize the local contrast.
        restored = enhance_low_light(image, gamma=1.0, clip_limit=4.0)
        return restored, "Applied Glare Reduction (CLAHE)"
        
    elif condition == 'Haze':
        # Apply AOD-Net or Dark Channel Prior
        # Placeholder till dehaze module is built
        return image, "Haze detected (Dehaze module pending)"
        
    elif condition == 'Rain' or condition == 'Clear (or Rain)':
        # Apply MPRNet or our OpenCV Derain equivalent
        # For our heuristic, we'll apply a light derain pass just in case
        restored = remove_rain(image, kernel_size=(5, 5), threshold=60)
        return restored, "Applied Derain Filter"
        
    return image, "No restoration needed"
    '''
