import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# For V2 architecture, we load the specific License Plate detector weights
# This model was fine-tuned on ALPR datasets, so its only class (0) is a license plate.
try:
    plate_model = YOLO("yolov8_plate.pt")
except Exception as e:
    print(f"Failed to load fine-tuned license plate model: {e}")
    plate_model = None

def detect_license_plates(image, conf_threshold=0.25):
    """
    Runs our fine-tuned YOLOv8 License Plate model to detect plates directly.
    Returns: Bounding boxes for each detected plate in [x1, y1, x2, y2] format.
    """
    if plate_model is None:
        return []
        
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    # The V2 model specifically points to the plate, bypassing generic vehicles
    results = plate_model(img_np, conf=conf_threshold)
    
    bounding_boxes = []
    if len(results) > 0:
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            bounding_boxes.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": score,
                "class_id": int(class_id)
            })
            
    return bounding_boxes
