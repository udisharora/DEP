# Import all the deep learning based restoration functions from our custom modules
from modules.dark_ir import process_with_darkir
from modules.nafnet import process_with_nafnet
from modules.dehaze import process_with_dehaze
from modules.derain import process_with_derain

# Define the central pipeline function that prepares an image for license plate detection
def prepare_image_for_detection(image):
    """
    Always deblur the image first for detection, then prepare DarkIR, DeHaze and DeRain fallback
    images only if a second (or third/fourth) detection pass is needed.
    """
    # Initialize a list to hold logging messages about which restorations were applied
    messages = []

    # First Attempt: Apply NAFNet for general deblurring and super-resolution
    try:
        # Pass the original image through the NAFNet model
        deblurred_image = process_with_nafnet(image)
        # Log successful application
        messages.append("Applied NAFNet deblurring")
    except Exception as exc:
        # If NAFNet fails (e.g., OOM), fallback to the original unmodified image
        deblurred_image = image
        # Log the failure reason
        messages.append(f"Skipped NAFNet deblurring: {exc}")

    # Second Attempt: Prepare a Dark/Night-time adapted version using the DarkIR model
    try:
        # Apply the DarkIR filter on top of the previously deblurred image
        darkir_image = process_with_darkir(deblurred_image)
        # Log successful preparation
        messages.append("Prepared DarkIR fallback")
    except Exception as exc:
        # Fallback to the deblurred image if DarkIR fails
        darkir_image = deblurred_image
        # Log the failure reason
        messages.append(f"Skipped DarkIR fallback: {exc}")

    # Third Attempt: Prepare a DeHazed version for foggy conditions
    try:
        # Apply the DeHaze filter
        dehazed_image = process_with_dehaze(deblurred_image)
        # Log successful preparation
        messages.append("Prepared DeHaze fallback")
    except Exception as exc:
        # Fallback to the deblurred image if DeHaze fails
        dehazed_image = deblurred_image
        # Log the failure reason
        messages.append(f"Skipped DeHaze fallback: {exc}")

    # Fourth Attempt: Prepare a DeRained version for rainy conditions
    try:
        # Apply the DeRain filter
        derained_image = process_with_derain(deblurred_image)
        # Log successful preparation
        messages.append("Prepared DeRain fallback")
    except Exception as exc:
        # Fallback to the deblurred image if DeRain fails
        derained_image = deblurred_image
        # Log the failure reason
        messages.append(f"Skipped DeRain fallback: {exc}")

    # Return the suite of generated images and the aggregated log string
    return darkir_image, dehazed_image, derained_image, deblurred_image, "; ".join(messages)
