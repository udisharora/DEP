import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

from modules.restoration import prepare_image_for_detection
from modules.detector import detect_license_plates
from modules.ocr_engine import extract_text
from modules.nafnet import process_with_nafnet
from modules.rto_metadata import parse_rto_metadata
from modules.vehicle_lookup import fetch_vehicle_data

st.set_page_config(page_title="Advanced ALPR Pipeline", layout="wide")

st.title("Auto-Adaptive License Plate Recognition")
st.write("A detection-first ALPR pipeline that always deblurs once before detection, uses DarkIR, DeHaze and DeRain as fallback detection passes, and finally deblurs the cropped plate before OCR.")

st.sidebar.header("Pipeline Architecture")
st.sidebar.markdown("""
1. **Prep 1**: Always deblur with NAFNet
2. **Detection Pass 1**: Try YOLOv8 on the deblurred image
3. **Fallback Prep 2**: Apply DarkIR on the deblurred image
4. **Fallback Prep 3**: Apply DeHaze on the deblurred image
5. **Fallback Prep 4**: Apply DeRain on the deblurred image
6. **Plate OCR**: Crop plate -> deblur crop -> TrOCR Engine
""")
st.sidebar.info("Upload an image in the main panel to run the ALPR pipeline.")

# Main content block
uploaded_file = st.file_uploader("Upload an image of a vehicle...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Stage 0: Load Image
    original_image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(original_image)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.subheader("1. Original Image")
        st.image(original_image, use_column_width=True)

    with st.spinner("Running ALPR Pipeline..."):
        darkir_image, dehazed_image, derained_image, fallback_image, restoration_msg = prepare_image_for_detection(original_image)
        
        detection_image = fallback_image
        detection_source = "deblurred image"
        boxes = detect_license_plates(detection_image)

        if len(boxes) == 0:
            detection_image = darkir_image
            detection_source = "DarkIR fallback image"
            boxes = detect_license_plates(detection_image)
            
        if len(boxes) == 0:
            detection_image = dehazed_image
            detection_source = "DeHaze fallback image"
            boxes = detect_license_plates(detection_image)

        if len(boxes) == 0:
            detection_image = derained_image
            detection_source = "DeRain fallback image"
            boxes = detect_license_plates(detection_image)

        st.info(f"**Detection Preparation**: {restoration_msg}")

        with col2:
            st.subheader("2. NAFNet")
            st.image(fallback_image, use_column_width=True)

        with col3:
            st.subheader("3. DarkIR")
            st.image(darkir_image, use_column_width=True)

        with col4:
            st.subheader("4. DeHaze")
            st.image(dehazed_image, use_column_width=True)

        with col5:
            st.subheader("5. DeRain")
            st.image(derained_image, use_column_width=True)

        with col6:
            st.subheader("6. Detection Used")
            st.image(detection_image, use_column_width=True)
            
        # Stage 3: License Plate Detection (YOLOv8)
        st.write("---")
        st.subheader("License Plate Detection")

        col3, col4 = st.columns(2)
        if len(boxes) == 0:
            st.warning("No license plates detected on either the deblurred image, the DarkIR fallback, the DeHaze fallback, or the DeRain fallback image.")
        else:
            # Draw boxes on image
            annotated_img = detection_image.copy()
            draw = ImageDraw.Draw(annotated_img)

            # Process the most confident plate box
            best_box = max(boxes, key=lambda x: x['score'])
            px1, py1, px2, py2 = best_box['box']
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)

            with col3:
                st.write(f"Detected **License Plate** on the {detection_source} (Confidence: {best_box['score']:.2f})")
                st.image(annotated_img, use_column_width=True)

            # Stage 4: Plate ROI Extraction
            h, w = np.array(detection_image).shape[:2]
            y_margin = int((py2 - py1) * 0.05)
            x_margin = int((px2 - px1) * 0.05)
            
            new_py1 = max(0, py1 - y_margin)
            new_py2 = min(h, py2 + y_margin)
            new_px1 = max(0, px1 - x_margin)
            new_px2 = min(w, px2 + x_margin)

            plate_crop_raw = np.array(detection_image)[new_py1:new_py2, new_px1:new_px2]
            
            # Define 4 different padding scales (0%, 5%, 10%, 15%) and perform each once
            padding_scales = [0.0, 0.05, 0.10, 0.15]
            predictions = []

            for pad_scale in padding_scales:
                pad_y = int(plate_crop_raw.shape[0] * pad_scale)
                pad_x = int(plate_crop_raw.shape[1] * pad_scale)
                new_h = plate_crop_raw.shape[0] + 2 * pad_y
                new_w = plate_crop_raw.shape[1] + 2 * pad_x
                if new_h < 20 or new_w < 20:
                    continue
                padded = np.pad(plate_crop_raw, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=255)
                try:
                    deblurred = process_with_nafnet(padded)
                    # sr_result = enhance_image_resolution(np.array(deblurred))  # Swin2SR temporarily removed
                    sr_result = deblurred  # Use deblurred image directly
                    extracted_txt, extracted_conf = extract_text(sr_result)
                    predictions.append((extracted_txt, extracted_conf, sr_result, padded))
                except Exception:
                    continue

            from collections import Counter
            if len(predictions) > 0:
                # Take majority vote on the extracted text strictly
                texts = [p[0] for p in predictions]
                majority_text = Counter(texts).most_common(1)[0][0]
                
                # Retrieve the specific pipeline state matching the majority output with highest confidence
                best_run = max([p for p in predictions if p[0] == majority_text], key=lambda x: x[1])
                text, conf, sr_plate, plate_crop_padded = best_run
                plate_msg = f"TTA Majority Vote on {len(padding_scales)} Padding Sizes (NAFNet + Swin2SR)"
            else:
                text, conf = "Failed TTA OCR", 0.0
                sr_plate = Image.fromarray(plate_crop_raw)
                plate_crop_padded = plate_crop_raw
                plate_msg = "Failed executing TTA Padding loop" 

            with col4:
                st.write("**License Plate Crop**")
                st.image(plate_crop_padded, channels="RGB", use_column_width=False, width=300)

                st.write("**Deblurred & Upscaled**")
                st.image(sr_plate, use_column_width=False, width=300)

                st.caption(plate_msg)

                st.success(f"**Final Extracted Text:** `{text}` (Conf: {conf:.2f})")
                
                rto_meta = parse_rto_metadata(text)
                if rto_meta["state"] != "Unknown" and rto_meta["state"] != "Unknown RTO State":
                    st.info(f"📍 **Registered In:** {rto_meta['state']} (RTO Code: {rto_meta['district_code']})")

                st.markdown("---")
                st.subheader("Vehicle Details")
                if st.button("Fetch RegCheck API Data"):
                    with st.spinner("Fetching data from RegCheck API..."):
                        vehicle_info = fetch_vehicle_data(text)
                        if vehicle_info["valid"]:
                            v_data = vehicle_info["data"]
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Owner:** {v_data.get('owner', 'N/A')}")
                                st.write(f"**Make & Model:** {v_data.get('make', 'N/A')} {v_data.get('model', '')}")
                                st.write(f"**Engine & Fuel:** {v_data.get('engine', 'N/A')}cc | {v_data.get('fuel', 'N/A')}")
                            with col_b:
                                st.write(f"**Location:** {v_data.get('location', 'N/A')}")
                                st.write(f"**Registration Date:** {v_data.get('registration_date', v_data.get('year', 'N/A'))}")
                                st.write(f"**Insurance Expiry:** {v_data.get('insurance', 'N/A')}")
                            st.success("Vehicle data retrieved from RegCheck successfully!")
                        else:
                            st.error(f"Could not fetch vehicle data via RegCheck: {vehicle_info.get('error', 'Unknown Error')}")

            if len(predictions) > 0:
                st.markdown("---")
                st.subheader("🔍 Test-Time Augmentation (TTA) Ensemble Results")
                st.write("Showing all 10 padded extraction passes used to calculate the majority vote.")
                
                with st.expander("View All Prediction Crops", expanded=True):
                    tta_cols = st.columns(5)
                    for idx, pred in enumerate(predictions):
                        p_txt, p_conf, p_sr, p_pad = pred
                        pad_val = padding_scales[idx] * 100
                        with tta_cols[idx % 5]:
                            st.image(p_sr, use_column_width=True)
                            st.caption(f"**Pad {pad_val:.0f}%**\n`{p_txt}`")

else:
    st.info("Please upload an image to begin the pipeline.")
