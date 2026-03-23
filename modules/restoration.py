from modules.dark_ir import process_with_darkir, managing_contrast_and_brightness_mathematically
from modules.derain import remove_rain
from modules.nafnet import process_with_nafnet

def route_and_restore(image, condition):
    """
    Deblurs the image with NAFNet first, then applies DarkIR and a final OCR-focused
    post-processing step. If NAFNet is unavailable, the pipeline falls back to the
    original image.
    """
    messages = []

    try:
        deblurred_image = process_with_nafnet(image)
        messages.append("Applied NAFNet deblurring")
    except Exception as exc:
        deblurred_image = image
        messages.append(f"Skipped NAFNet deblurring: {exc}")

    darkir_source = deblurred_image
    darkir_gamma = 1.5
    darkir_clip_limit = 3.0

    if condition == 'Glare':
        darkir_gamma = 1.0
        darkir_clip_limit = 4.0
        messages.append("Applied DarkIR glare reduction")
    elif condition == 'Night':
        messages.append("Applied DarkIR low-light restoration")
    elif condition == 'Haze':
        messages.append("Applied DarkIR before final enhancement; dehaze module pending")
    elif condition == 'Rain' or condition == 'Clear (or Rain)':
        darkir_source = remove_rain(deblurred_image, kernel_size=(5, 5), threshold=60)
        messages.append("Applied derain filter before DarkIR")
        messages.append("Applied DarkIR after derain")
    else:
        messages.append("Applied DarkIR restoration")

    darkir_restored = process_with_darkir(darkir_source)
    restored = managing_contrast_and_brightness_mathematically(
        darkir_restored,
        gamma=darkir_gamma,
        clip_limit=darkir_clip_limit,
    )
    messages.append("Applied final contrast and brightness enhancement")

    return deblurred_image, darkir_restored, restored, "; ".join(messages)
