from modules.dark_ir import process_with_darkir
from modules.nafnet import process_with_nafnet
from modules.dehaze import process_with_dehaze
from modules.derain import process_with_derain

def prepare_image_for_detection(image):
    """
    Always deblur the image first for detection, then prepare DarkIR, DeHaze and DeRain fallback
    images only if a second (or third/fourth) detection pass is needed.
    """
    messages = []

    try:
        deblurred_image = process_with_nafnet(image)
        messages.append("Applied NAFNet deblurring")
    except Exception as exc:
        deblurred_image = image
        messages.append(f"Skipped NAFNet deblurring: {exc}")

    try:
        darkir_image = process_with_darkir(deblurred_image)
        messages.append("Prepared DarkIR fallback")
    except Exception as exc:
        darkir_image = deblurred_image
        messages.append(f"Skipped DarkIR fallback: {exc}")

    try:
        dehazed_image = process_with_dehaze(deblurred_image)
        messages.append("Prepared DeHaze fallback")
    except Exception as exc:
        dehazed_image = deblurred_image
        messages.append(f"Skipped DeHaze fallback: {exc}")

    try:
        derained_image = process_with_derain(deblurred_image)
        messages.append("Prepared DeRain fallback")
    except Exception as exc:
        derained_image = deblurred_image
        messages.append(f"Skipped DeRain fallback: {exc}")

    return darkir_image, dehazed_image, derained_image, deblurred_image, "; ".join(messages)
