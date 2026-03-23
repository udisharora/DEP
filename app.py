import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

from modules.classifier import classify_condition
from modules.restoration import prepare_image_for_detection
from modules.detector import detect_license_plates
from modules.ocr_engine import resize_and_clahe, extract_text
from modules.nafnet import process_with_nafnet

st.set_page_config(page_title="Advanced ALPR Pipeline", layout="wide")

st.title("Auto-Adaptive License Plate Recognition")
st.write("A detection-first ALPR pipeline that always deblurs once before detection, uses DarkIR only as a fallback detection pass, and finally deblurs the cropped plate before OCR.")

st.sidebar.header("Pipeline Architecture")
st.sidebar.markdown("""
1. **Condition Classifier**: Scene label for inspection
2. **Prep 1**: Always deblur with NAFNet
3. **Detection Pass 1**: Try YOLOv8 on the deblurred image
4. **Fallback Prep 2**: Apply DarkIR on the deblurred image
5. **Plate OCR**: Crop plate -> deblur crop -> OCR preprocessing -> EasyOCR
""")
st.sidebar.info("Upload an image in the main panel to run the full 5-stage pipeline.")

# Main content block
uploaded_file = st.file_uploader("Upload an image of a vehicle...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Stage 0: Load Image
    original_image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(original_image)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("1. Original Image")
        st.image(original_image, use_column_width=True)
        
    with st.spinner("Running 5-Stage ALPR Pipeline..."):
        # Stage 1: Condition Classifier
        condition = classify_condition(img_np)
        st.success(f"**Stage 1 - Classifier**: Detected Environment Condition => `{condition}`")

        # Stage 2: Always deblur before detection, then prepare DarkIR fallback.
        darkir_image, fallback_image, restoration_msg = prepare_image_for_detection(original_image, condition)
        detection_image = fallback_image
        boxes = detect_license_plates(detection_image)
        used_initial_detection = len(boxes) > 0

        if not used_initial_detection:
            detection_image = darkir_image
            boxes = detect_license_plates(detection_image)

        st.info(f"**Stage 2 - Detection Preparation**: {restoration_msg}")

        with col2:
            st.subheader("2. NAFNet Deblur Output")
            st.image(fallback_image, use_column_width=True)

        with col3:
            st.subheader("3. DarkIR Fallback Output")
            st.image(darkir_image, use_column_width=True)

        with col4:
            st.subheader("4. Detection Image Used")
            st.image(detection_image, use_column_width=True)
            
        # Stage 3: License Plate Detection (YOLOv8)
        st.write("---")
        st.subheader("Stage 3: License Plate Detection")

        col3, col4 = st.columns(2)
        if len(boxes) == 0:
            st.warning("No license plates detected on either the deblurred image or the DarkIR fallback image.")
        else:
            # Draw boxes on image
            annotated_img = detection_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Process the most confident plate box
            best_box = max(boxes, key=lambda x: x['score'])
            px1, py1, px2, py2 = best_box['box']
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)
            
            with col3:
                if used_initial_detection:
                    detection_source = "deblurred image"
                else:
                    detection_source = "DarkIR fallback image"
                st.write(f"Detected **License Plate** on the {detection_source} (Confidence: {best_box['score']:.2f})")
                st.image(annotated_img, use_column_width=True)
                
            # Stage 4: Plate ROI Extraction
            plate_crop = np.array(detection_image)[py1:py2, px1:px2]

            try:
                deblurred_plate = process_with_nafnet(plate_crop)
                plate_for_ocr = np.array(deblurred_plate)
                plate_deblur_msg = "Applied NAFNet deblurring to the cropped plate before OCR"
            except Exception as exc:
                deblurred_plate = Image.fromarray(plate_crop)
                plate_for_ocr = plate_crop
                plate_deblur_msg = f"Skipped plate deblurring before OCR: {exc}"
            
            # Stage 5: OCR preprocessing and text extraction
            enhanced_plate = resize_and_clahe(plate_for_ocr)
            text, conf = extract_text(enhanced_plate) 
            
            with col4:
                st.write("**License Plate ROI Extraction**")
                st.image(plate_crop, channels="RGB", use_column_width=False, width=300)

                st.write("**Deblurred Plate for OCR**")
                st.image(deblurred_plate, use_column_width=False, width=300)
                
                st.write("**Prepared for OCR**")
                st.image(enhanced_plate, channels="GRAY", use_column_width=False, width=300)

                st.caption(plate_deblur_msg)
                
                st.success(f"**Final Extracted Text:** `{text}` (Conf: {conf:.2f})")

else:
    st.info("Please upload an image to begin the pipeline.")
