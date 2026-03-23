from modules.dark_ir import enhance_low_light
from modules.derain import remove_rain
from modules.nafnet import process_with_nafnet
# Placeholder for dehaze module, importing here to show architecture 
# from modules.dehaze import remove_haze

def route_and_restore(image, condition):
    """
    Deblurs the image first with NAFNet, then applies condition-aware restoration.
    """
    try:
        deblurred_image = process_with_nafnet(image)
        deblur_msg = "Applied NAFNet deblurring"
    except Exception as exc:
        deblurred_image = image
        deblur_msg = f"Skipped NAFNet deblurring: {exc}"

    if condition == 'Night':
        restored = enhance_low_light(deblurred_image, gamma=1.5, clip_limit=3.0)
        return deblurred_image, restored, f"{deblur_msg}; applied low-light enhancement"
        
    elif condition == 'Glare':
        restored = enhance_low_light(deblurred_image, gamma=1.0, clip_limit=4.0)
        return deblurred_image, restored, f"{deblur_msg}; applied glare reduction"
        
    elif condition == 'Haze':
        return deblurred_image, deblurred_image, f"{deblur_msg}; dehaze module pending"
        
    elif condition == 'Rain' or condition == 'Clear (or Rain)':
        restored = remove_rain(deblurred_image, kernel_size=(5, 5), threshold=60)
        return deblurred_image, restored, f"{deblur_msg}; applied derain filtering"
        
    return deblurred_image, deblurred_image, deblur_msg
