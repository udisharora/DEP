import torch
import numpy as np
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

processor = None
model = None

def init_sr_model():
    """
    Initializes the Swin2SR models from HuggingFace on first call.
    """
    global processor, model
    if processor is None or model is None:
        try:
            processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2sr-classical-sr-x2-64")
            model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2sr-classical-sr-x2-64")
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            model.to(device)
        except Exception as e:
            print(f"Failed to load Swin2SR: {e}")

def enhance_image_resolution(image):
    """
    Upscales the provided plate image by a factor of 2x using Swin2SR.
    Returns a high-resolution PIL Image.
    """
    init_sr_model()
    if model is None or processor is None:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
        
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        
    # Process model raw output tensors back to numpy standard vision array format
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    
    return Image.fromarray(output)
