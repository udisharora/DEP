# Import os module for interacting with the operating system (e.g., environment variables, file paths)
import os
# Import uuid to generate unique identifiers for uploaded files
import uuid
# Import FastAPI and related components for building the API endpoints and handling file uploads
from fastapi import FastAPI, UploadFile, File, HTTPException
# Import CORS middleware to allow cross-origin requests from the frontend
from fastapi.middleware.cors import CORSMiddleware
# Import the Celery task function to process ALPR asynchronously
from tasks import process_alpr_task
# Import AsyncResult to check the status of our Celery tasks
from celery.result import AsyncResult
# Import function to fetch mock vehicle registration details
from modules.vehicle_lookup import fetch_vehicle_data

# Initialize the FastAPI application with a custom title
app = FastAPI(title="ALPR API Gateway")

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
async def process_image(file: UploadFile = File(...)):
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
    task = process_alpr_task.delay(file_path)
    
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
