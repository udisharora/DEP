import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2

# Import our custom modules
from modules.classifier import classify_condition
from modules.restoration import route_and_restore
from modules.detector import detect_license_plates
from modules.ocr_engine import resize_and_clahe, extract_text

st.set_page_config(page_title="Advanced ALPR Pipeline", layout="wide")

st.title("Auto-Adaptive License Plate Recognition")
st.write("An end-to-end ALPR pipeline that detects adverse weather/night conditions, adaptively restores the image, detects vehicles, crops license plates, and extracts alphanumeric text.")

st.sidebar.header("Pipeline Architecture")
st.sidebar.markdown("""
1. **Classifier**: Night/Rain/Haze
2. **Restoration**: Adaptive Routing
3. **Detector**: YOLOv8 (Vehicle -> Plate)
4. **Alignment**: Binarize & Warp
5. **OCR Engine**: EasyOCR
""")
st.sidebar.info("Upload an image in the main panel to run the full 5-stage pipeline.")

# Main content block
uploaded_file = st.file_uploader("Upload an image of a vehicle...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Stage 0: Load Image
    original_image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(original_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Original Image")
        st.image(original_image, use_column_width=True)
        
    with st.spinner("Running 5-Stage ALPR Pipeline..."):
        # Stage 1: Condition Classifier --------TEMPORARY PLACEHOLDER--------
        condition = 'test'  # Placeholder for testing
        # condition = classify_condition(img_np)  # Uncomment when classifier is implemented
        st.success(f"**Stage 1 - Classifier**: Detected Environment Condition => `{condition}`")
        
        # Stage 2: Adaptive Restoration
        restored_image, restoration_msg = route_and_restore(original_image, condition)
        st.info(f"**Stage 2 - Restoration**: {restoration_msg}")
        
        with col2:
            st.subheader("2. Restored Image")
            st.image(restored_image, use_column_width=True)
            
        # Stage 3: Deep License Plate Detection (YOLOv8)
        st.write("---")
        st.subheader("Stage 3: Deep License Plate Detection")
        
        boxes = detect_license_plates(restored_image)
        
        col3, col4 = st.columns(2)
        if len(boxes) == 0:
            st.warning("No license plates detected by fine-tuned YOLOv8.")
        else:
            # Draw boxes on image
            annotated_img = restored_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Process the most confident plate box
            best_box = max(boxes, key=lambda x: x['score'])
            px1, py1, px2, py2 = best_box['box']
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)
            
            with col3:
                st.write(f"Detected **License Plate** (Confidence: {best_box['score']:.2f})")
                st.image(annotated_img, use_column_width=True)
                
            # Stage 4: Plate ROI Extraction
            plate_crop = np.array(restored_image)[py1:py2, px1:px2]
            
            # Stage 5: Binarization & OCR
            enhanced_plate = resize_and_clahe(plate_crop)
            text, conf = extract_text(enhanced_plate) 
            
            with col4:
                st.write("**License Plate ROI Extraction**")
                st.image(plate_crop, channels="RGB", use_column_width=False, width=300)
                
                st.write("**Enhanced for OCR (CLAHE)**")
                st.image(enhanced_plate, channels="GRAY", use_column_width=False, width=300)
                
                st.success(f"**Final Extracted Text:** `{text}` (Conf: {conf:.2f})")

else:
    st.info("Please upload an image to begin the pipeline.")
