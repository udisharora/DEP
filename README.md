# Auto-Adaptive License Plate Recognition (ALPR) Pipeline

A highly robust, detection-first Automatic License Plate Recognition (ALPR) system built with Streamlit. This pipeline is designed to handle challenging real-world scenarios, including blurry images, low-light conditions, and adverse weather, by leveraging advanced image restoration techniques and state-of-the-art Deep Learning models.

## Features

- **Advanced Image Restoration Pipeline**: 
  - **NAFNet Deblurring**: Always applies state-of-the-art deblurring before detection to ensure the highest possible accuracy.
  - **DarkIR Fallback**: Utilizes DarkIR image enhancement specifically tailored for low-light or degraded images if initial detection fails.
- **Robust License Plate Detection**: Uses a fine-tuned **YOLOv8** model to accurately detect and crop license plates from the processed images.
- **High-Accuracy OCR with TTA**: 
  - Employs a **TrOCR (Transformer-based Optical Character Recognition)** engine.
  - Features **Test-Time Augmentation (TTA)** using multi-scale padding and strict majority voting to guarantee fault-tolerant, accurate plate reading.
- **Geographical RTO Metadata Parsing**: Automatically parses the recognized Indian license plate text to extract and display Regional Transport Office (RTO) state and district registration metadata.
- **Interactive UI**: An intuitive web interface built with **Streamlit** that outlines the pipeline execution step-by-step and provides a detailed breakdown of intermediate stages (deblurring, bounding boxes, plate cropping, TTA results).

## Project Structure

```text
DEP/
├── app.py                   # Main Streamlit application entry point
├── requirements.txt         # Project dependencies
├── yolov8_plate.pt          # Fine-tuned YOLOv8 model weights for plate detection
├── modules/                 # Core pipeline modules
│   ├── detector.py          # YOLOv8 license plate detection logic
│   ├── ocr_engine.py        # TrOCR text extraction logic
│   ├── restoration.py       # Orchestrates image prep (NAFNet / DarkIR)
│   ├── nafnet.py            # NAFNet deblurring integration
│   ├── rto_metadata.py      # Indian license plate formatting and RTO parsing
│   └── super_resolution.py  # Image upscaling utilities
├── NAFNet/                  # NAFNet model dependencies and weights
└── DarkIR/                  # DarkIR model dependencies and weights
```

## Setup & Installation

**Prerequisites**: Ensure you have Python 3.8+ installed.

1. **Navigate to the Directory**:
   ```bash
   cd /path/to/DEP
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Ensure that PyTorch and torchvision are installed correctly for your hardware (CPU/GPU) to leverage hardware acceleration if available.

## Usage

Start the Streamlit web application:

```bash
streamlit run app.py
```

This will open a new tab in your default web browser (usually at `http://localhost:8501`). 

### Running the Pipeline:
1. Use the **file uploader** in the main panel to upload an image of a vehicle (`.jpg`, `.jpeg`, `.png`).
2. The pipeline will automatically:
   - Run the image through the prep and deblurring stages.
   - Run the YOLOv8 Object Detector.
   - If initial detection fails, run the DarkIR fallback and re-try detection.
   - Crop the bounding box of the highest-confidence license plate.
   - Run a TTA ensemble padding loop on the cropped image.
   - Extract the license plate text using TrOCR.
   - Parse and display the RTO registration information based on the strict Indian formatting heuristics.
3. Review the intermediate pipeline steps, visual outputs, confidence scores, and TTA crops directly in the UI.

## Acknowledgements

- **YOLOv8** by Ultralytics for object detection.
- **TrOCR** by Microsoft (via Hugging Face Transformers) for Optical Character Recognition.
- **NAFNet** for image deblurring and restoration.
- **Streamlit** for the interactive front-end framework.


