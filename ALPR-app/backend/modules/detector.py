# Import OpenCV for image processing tasks like color conversion
import cv2
# Import the YOLO class from ultralytics to load our object detection model
from ultralytics import YOLO
# Import numpy for array manipulation (images are essentially 3D or 2D matrices)
import numpy as np
# Import Pil to handle Python Image Library image objects
from PIL import Image

# For V2 architecture, we load the specific License Plate detector weights
# This model was fine-tuned on ALPR datasets, so its only class (0) is a license plate.
try:
    # Initialize the YOLOv8 model instance using the designated weights file
    plate_model = YOLO("yolov8_plate.pt")
except Exception as e:
    # If the file is missing or corrupted, catch the error gracefully
    print(f"Failed to load fine-tuned license plate model: {e}")
    # Assign None to the model, preventing subsequent operations from crashing
    plate_model = None

# Main function to locate license plates within a given image
def detect_license_plates(image, conf_threshold=0.25):
    """
    Runs our fine-tuned YOLOv8 License Plate model to detect plates directly.
    Returns: Bounding boxes for each detected plate in [x1, y1, x2, y2] format.
    """
    # Safety Check: If the model failed to load during initialization, abort
    if plate_model is None:
        return []
        
    # Check if the incoming variable is a PIL Image object
    if isinstance(image, Image.Image):
        # Convert the PIL Image into a Numpy multidimensional array explicitly
        img_np = np.array(image)
    else:
        # Otherwise, assume it's already a numpy array and create a deepcopy
        img_np = image.copy()

    # YOLO expects 3-channel (RGB) input. Promote grayscale safely when needed.
    # Case 1: The array is purely 2D (height x width), which means it's single-channel grayscale
    if len(img_np.shape) == 2:
        # Transform the single channel into standard 3-channel RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    # Case 2: The array is 3D, but the 3rd dimension only has 1 layer
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
        # Extract the single layer slice and convert that to a standard 3-channel RGB image
        img_np = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
        
    # The ALPR V2 model specifically targets the license plate, completely bypassing generic vehicles
    # Execute the YOLO inference engine with the specified confidence threshold
    results = plate_model(img_np, conf=conf_threshold)
    
    # List to keep track of successfully found bounding boxes
    bounding_boxes = []
    # If the YOLO engine returned results
    if len(results) > 0:
        # Iterate over all detections contained within the first Result object's data matrix
        for r in results[0].boxes.data.tolist():
            # Unpack the 6 values denoting Box coordinates, Score, and Object Class ID
            x1, y1, x2, y2, score, class_id = r
            # Format the output into our standardized dictionary
            bounding_boxes.append({
                # Convert the float coordinates into absolute integer pixel values
                "box": [int(x1), int(y1), int(x2), int(y2)],
                # Append the confidence likelihood (e.g. 0.95 = 95%) 
                "score": score,
                # Append the detected class identifier (almost always 0 for single-class ALPR)
                "class_id": int(class_id)
            })
            
    # Return the populated list of detected plates to the caller
    return bounding_boxes
