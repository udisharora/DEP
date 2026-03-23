from modules.dark_ir import process_with_darkir
from modules.nafnet import process_with_nafnet

def prepare_image_for_detection(image, condition):
    """
    Always deblur the image first for detection, then prepare a DarkIR fallback
    image only if a second detection pass is needed.
    """
    messages = []

    try:
        deblurred_image = process_with_nafnet(image)
        messages.append("Applied NAFNet deblurring before the first detection pass")
    except Exception as exc:
        deblurred_image = image
        messages.append(f"Skipped initial NAFNet deblurring: {exc}")

    try:
        darkir_image = process_with_darkir(deblurred_image)
        messages.append("Prepared DarkIR fallback image for the second detection pass")
    except Exception as exc:
        darkir_image = deblurred_image
        messages.append(f"Skipped DarkIR fallback: {exc}")

    return darkir_image, deblurred_image, "; ".join(messages)
