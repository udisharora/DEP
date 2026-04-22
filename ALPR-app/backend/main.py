# Import os module for interacting with the operating system (e.g., environment variables, file paths)
import os
# Import uuid to generate unique identifiers for uploaded files
import uuid
# Import FastAPI and related components for building the API endpoints and handling file uploads
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# Import CORS middleware to allow cross-origin requests from the frontend
from fastapi.middleware.cors import CORSMiddleware
# Import the Celery task function to process ALPR asynchronously
from tasks import process_alpr_task
# Import AsyncResult to check the status of our Celery tasks
from celery.result import AsyncResult
# Import function to fetch mock vehicle registration details
from modules.vehicle_lookup import fetch_vehicle_data

from prometheus_client import make_asgi_app, Counter, Gauge, Histogram

# Initialize the FastAPI application with a custom title
app = FastAPI(title="ALPR API Gateway")

# Expose Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define Prometheus metrics for Data Drift detection
OCR_CONFIDENCE_GAUGE = Gauge('alpr_ocr_confidence', 'Confidence score of the OCR Engine')
OCR_LENGTH_HISTOGRAM = Histogram('alpr_ocr_length', 'Length of the predicted license plate characters')

# Advanced MLOps Metrics tracking pipeline health and data quality
ALPR_BLUR_SCORE = Gauge('alpr_image_blur_score', 'Variance of Laplacian tracking camera focus degradation')
ALPR_BBOX_COUNT = Histogram('alpr_bbox_count', 'Number of license plates detected per frame by YOLO')
ALPR_INFERENCE_TIME = Histogram('alpr_inference_latency_seconds', 'Time taken by PyTorch inference pipeline')
ALPR_DELTA_QUEUE = Gauge('alpr_delta_queue_size', 'Number of items explicitly waiting for Continuous Training')
ALPR_QUARANTINE_QUEUE = Gauge('alpr_quarantine_queue_size', 'Number of items quaratined for human review')

# Section 1 & Section 4 MLOps Metrics (Accuracy, Levenshtein, Input Quality)
ALPR_EXACT_MATCH = Gauge('alpr_exact_match', 'Exact match accuracy vs ground truth')
ALPR_CHAR_ACCURACY = Gauge('alpr_character_accuracy', 'Character accuracy vs ground truth')
ALPR_EDIT_DISTANCE = Histogram('alpr_edit_distance', 'Edit distance vs ground truth')
ALPR_INVALID_FORMAT = Counter('alpr_invalid_format_total', 'Number of predictions violating standard plate formatting')
ALPR_BRIGHTNESS = Gauge('alpr_image_brightness', 'Average pixel intensity tracking day/night illumination trends')
ALPR_DATA_DRIFT = Gauge('alpr_data_drift_score', 'Compound data drift proxy score based on statistical deviations')

# Add middleware to handle Cross-Origin Resource Sharing (CORS)
# This allows our React frontend to communicate with this backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow requests from any origin (CAN BE RESTRICTED IN PRODUCTION)
    allow_credentials=True, # Allow cookies/credentials to be sent
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all HTTP headers
)

# Define the directory where uploaded images will be temporarily stored
# It reads from an environment variable or uses 'shared_uploads' as a fallback
SHARED_UPLOADS = os.environ.get("SHARED_UPLOADS_DIR", "shared_uploads")
# Create the uploads directory if it doesn't already exist
os.makedirs(SHARED_UPLOADS, exist_ok=True)

# Define a POST endpoint for submitting images to process
@app.post("/process-image")
async def process_image(file: UploadFile = File(...), label: str = Form(None)):
    # Validate that the uploaded file is indeed an image by checking its MIME content type
    if not file.content_type.startswith("image/"):
        # If it's not an image, raise a 400 Bad Request HTTP exception
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")
    
    # Read the file's binary contents into memory asynchronously
    contents = await file.read()
    # Validate file size: if larger than 5 Megabytes (5 * 1024 * 1024 bytes)
    if len(contents) > 5 * 1024 * 1024:
        # Raise a 400 Bad Request exception for exceeding size limit
        raise HTTPException(status_code=400, detail="File too large. Must be under 5MB.")
    
    # Generate a unique string ID for the file to prevent naming collisions
    file_id = str(uuid.uuid4())
    # Extract the file extension from the original filename, default to 'jpg' if none is found
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    # Construct the full absolute path for saving the file in our shared uploads directory
    file_path = os.path.join(SHARED_UPLOADS, f"{file_id}.{ext}")
    
    # Open the newly created file path in write-binary mode
    with open(file_path, "wb") as f:
        # Write the image binary contents into the file
        f.write(contents)
        
    # Dispatch the ALPR processing logic to Celery as an asynchronous background task
    # Passing the saved file path rather than the whole image blob for efficiency
    task = process_alpr_task.delay(file_path, label)
    
    # Promptly return the unique task ID to the client so they can start polling for status
    return {"task_id": task.id}

# Define a GET endpoint to check the progress/status of an ongoing Celery task
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    # Fetch the task asynchronously from Celery's result backend using the task_id
    task_result = AsyncResult(task_id)
    # Prepare a result dictionary containing the task ID and its current status (e.g., PENDING, SUCCESS, FAILURE)
    result = {
        "task_id": task_id,
        "status": task_result.status,
    }
    # If the background processing completed successfully
    if task_result.status == "SUCCESS":
        # Embed the final processed ALPR data into the response
        result["data"] = task_result.result
        
        # Log metrics to Prometheus if OCR completed
        confidence = result["data"].get("confidence", 0)
        text_result = result["data"].get("extracted_text", "")
        
        # We only log realistic readings to track drift
        if confidence > 0 and text_result != "Failed OCR" and text_result != "Failed TTA OCR":
            OCR_CONFIDENCE_GAUGE.set(confidence)
            OCR_LENGTH_HISTOGRAM.observe(len(text_result))
            
            # Export hardware and detection metrics
            blur_score = result["data"].get("blur_score", 0)
            if blur_score > 0:
                ALPR_BLUR_SCORE.set(blur_score)
                
            brightness = result["data"].get("brightness", -1)
            if brightness != -1:
                ALPR_BRIGHTNESS.set(brightness)
                
            invalid_format = result["data"].get("invalid_format", False)
            if invalid_format:    
                ALPR_INVALID_FORMAT.inc()
                
            ALPR_DATA_DRIFT.set(result["data"].get("drift_score", 0.0))    
                
            # Log Model Performance Metrics if Ground Truth was supplied (Simulated Mode)
            mlops_perf = result["data"].get("performance_scores")
            if mlops_perf:
                ALPR_EXACT_MATCH.set(mlops_perf["exact_match"])
                ALPR_CHAR_ACCURACY.set(mlops_perf["character_accuracy"])
                ALPR_EDIT_DISTANCE.observe(mlops_perf["edit_distance"])
                
            bbox_count = result["data"].get("bbox_count", 0)
            ALPR_BBOX_COUNT.observe(bbox_count)
            
            inference_time = result["data"].get("inference_time", 0)
            if inference_time > 0:
                ALPR_INFERENCE_TIME.observe(inference_time)
                
            # Actively poll the MLOps Queue sizes to graph CT loop triggers and load
            try:
                import csv
                delta_csv = "data/delta_metadata.csv"
                if os.path.exists(delta_csv):
                    with open(delta_csv, 'r') as f:
                        q_size = max(0, len(list(csv.reader(f))) - 1)
                        ALPR_DELTA_QUEUE.set(q_size)
                
                review_dir = "data/needs_review"
                if os.path.exists(review_dir):
                    ALPR_QUARANTINE_QUEUE.set(len(os.listdir(review_dir)))
            except Exception:
                pass
            
    # If the background processing encountered an exception or crashed
    elif task_result.status == "FAILURE":
        # Expose the error message/stack trace info for debugging
        result["error"] = str(task_result.info)
    
    # Return the dynamic status object to the client
    return result

# Define a GET endpoint to simulate fetching vehicle details given a recognized license plate number
@app.get("/vehicle/{plate_number}")
def get_vehicle(plate_number: str):
    # Call our helper module to fetch registration metadata for the targeted plate
    data = fetch_vehicle_data(plate_number)
    # Return the dictionary of vehicle details or error
    return data
