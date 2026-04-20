import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import torch

try:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    #----REMOVED GPU OPTION
    device = 'cpu'
    model.to(device)
except Exception as e:
    print(f"Failed to load TrOCR: {e}")
    processor, model = None, None

def resize_and_clahe(plate_image):
    """
    Ensure the extracted plate image is properly formatted in RGB for the TrOCR Processor.
    """
    if isinstance(plate_image, np.ndarray):
        if len(plate_image.shape) == 2:
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2RGB)
        elif plate_image.shape[2] == 4:
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(plate_image)
    
    return plate_image.convert("RGB")

def format_indian_plate_strict(text):
    """
    Enforces a strict 2-2-2-4 character rendering block for Indian License Plates.
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    
    num_to_letter = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B', '4': 'A', '6': 'G', 'Q': 'O'}
    letter_to_num = {'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'A': '4', 'G': '6', 'Q': '0'}
    
    def repl_letter(c: str) -> str:
        return str(num_to_letter.get(c, c)) if str(c).isdigit() else str(c)
        
    def repl_num(c: str) -> str:
        return str(letter_to_num.get(c, c)) if str(c).isalpha() else str(c)
    
    char_list = list(cleaned)
    formatted = []
    
    # Strictly parse through as 2 Letters, 2 Numbers, 2 Letters, 4 Numbers
    for i, c in enumerate(char_list):
        if i < 2:
            formatted.append(repl_letter(c))
        elif i < 4:
            formatted.append(repl_num(c))
        elif i < 6:
            formatted.append(repl_letter(c))
        elif i < 10:
            formatted.append(repl_num(c))
        else:
            break
            
    res = "".join(formatted)
    
    # Split into 2-2-2-4 string groupings regardless of final length
    parts = []
    if len(res) > 0:
        parts.append(res[0:2])
    if len(res) > 2:
        parts.append(res[2:4])
    if len(res) > 4:
        parts.append(res[4:6])
    if len(res) > 6:
        parts.append(res[6:10])
        
    return " ".join(parts)

def extract_text(plate_image):
    """
    Runs Microsoft TrOCR on the aligned plate image.
    Generates text directly from the image features.
    """
    if model is None or processor is None:
        return "OCR Model Failed to Load", 0.0
        
    # Ensure PIL format
    if isinstance(plate_image, np.ndarray):
        plate_image = Image.fromarray(plate_image)
    
    # Process image into tensor values
    pixel_values = processor(images=plate_image, return_tensors="pt").pixel_values.to(device)
    
    # Generate OCR text tokens
    outputs = model.generate(
        pixel_values, 
        max_new_tokens=15, 
        return_dict_in_generate=True, 
        output_scores=True
    )
    
    generated_ids = outputs.sequences
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Rough approximation of TrOCR token confidence
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        probs = [torch.softmax(score, dim=-1).max().item() for score in outputs.scores]
        confidence = sum(probs) / len(probs)
    else:
        confidence = 1.0
        
    # Apply strict Indian license plate conditioning
    final_text = format_indian_plate_strict(generated_text)
    
    return final_text, confidence
