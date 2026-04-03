from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image, ImageDraw
import numpy as np
import io
import base64
from collections import Counter

from modules.restoration import prepare_image_for_detection
from modules.detector import detect_license_plates
from modules.ocr_engine import extract_text
from modules.nafnet import process_with_nafnet
from modules.rto_metadata import parse_rto_metadata
# from modules.super_resolution import enhance_image_resolution
from modules.vehicle_lookup import fetch_vehicle_data

app = FastAPI(title="ALPR Pipeline API", description="API for Auto-Adaptive License Plate Recognition", version="1.0.0")

# Enable CORS for the Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image_base64(img_pil):
    """Encodes a PIL Image to base64 string"""
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def read_root():
    return {"message": "ALPR API is running"}

@app.post("/analyze")
async def analyze_vehicle(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Stage 1 & 2: Restoration and Prep
        darkir_image, dehazed_image, derained_image, fallback_image, restoration_msg = prepare_image_for_detection(original_image)
        
        # Stage 3: Detection Pass
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

        # Base64 encode intermediate images to send to frontend for visual display
        res_data = {
            "restoration_msg": restoration_msg,
            "images": {
                "original": encode_image_base64(original_image),
                "deblurred": encode_image_base64(fallback_image),
                "darkir": encode_image_base64(darkir_image),
                "dehaze": encode_image_base64(dehazed_image),
                "derain": encode_image_base64(derained_image),
                "detection_used": encode_image_base64(detection_image),
            },
            "detection_source": detection_source,
            "plate_detected": False,
            "plate_results": None,
            "predictions_tta": []
        }

        if len(boxes) > 0:
            res_data["plate_detected"] = True
            
            # Process best box
            best_box = max(boxes, key=lambda x: x['score'])
            px1, py1, px2, py2 = best_box['box']
            
            annotated_img = detection_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)
            
            res_data["images"]["annotated"] = encode_image_base64(annotated_img)
            res_data["detection_confidence"] = float(best_box['score'])

            # Stage 4: Plate ROI Extraction & TTA
            h, w = np.array(detection_image).shape[:2]
            y_margin = int((py2 - py1) * 0.05)
            x_margin = int((px2 - px1) * 0.05)
            
            new_py1 = max(0, py1 - y_margin)
            new_py2 = min(h, py2 + y_margin)
            new_px1 = max(0, px1 - x_margin)
            new_px2 = min(w, px2 + x_margin)

            plate_crop_raw = np.array(detection_image)[new_py1:new_py2, new_px1:new_px2]
            
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
                    sr_result = deblurred 
                    extracted_txt, extracted_conf = extract_text(sr_result)
                    predictions.append({
                        "text": extracted_txt, 
                        "conf": float(extracted_conf), 
                        "pad_scale": pad_scale,
                        "sr_image": encode_image_base64(sr_result)
                    })
                except Exception as e:
                    # Log exception silently
                    continue

            if len(predictions) > 0:
                texts = [p["text"] for p in predictions]
                majority_text = Counter(texts).most_common(1)[0][0]
                
                best_run = max([p for p in predictions if p["text"] == majority_text], key=lambda x: x["conf"])
                
                final_text = best_run["text"]
                final_conf = best_run["conf"]
                res_data["plate_results"] = {
                    "text": final_text,
                    "confidence": final_conf,
                    "plate_crop": encode_image_base64(Image.fromarray(plate_crop_raw)),
                    "best_sr_image": best_run["sr_image"]
                }
                res_data["predictions_tta"] = predictions
                
                # Fetch RTO Metadata
                rto_meta = parse_rto_metadata(final_text)
                res_data["rto_metadata"] = rto_meta
                
                # Vehicle info is fetched separately on-demand
                res_data["vehicle_info"] = None

        return JSONResponse(content=res_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vehicle_info/{plate_text}")
def get_vehicle_info(plate_text: str):
    try:
        vehicle_info = fetch_vehicle_data(plate_text)
        return vehicle_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
