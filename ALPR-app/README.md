# Autonomous ALPR System (Active Learning MLOps)

An end-to-end, fully autonomous Automatic License Plate Recognition (ALPR) web application backed by PyTorch (`TrOCR`), YOLO detection, and a highly optimized Continuous Training (CT) lifecycle.

For local development on macOS, this project uses a deliberate large-model workaround: the fine-tuned OCR weights are copied into Docker-managed Linux storage with `docker cp` instead of being loaded directly from the shared Mac filesystem. This avoids a `virtiofs` bottleneck that previously caused unstable reads with the large model artifact.

---

## 🏗️ 1. Project Architecture

The exact architecture comprises a 6-layer microservice stack running entirely through Docker Compose.

* **Frontend:** A React + Vite TypeScript web application where users upload images physically.
* **API Gateway (`backend`):** A FastAPI Python server acting as the router. It receives images, tracks exact MLOps Drift metrics for Prometheus, and passes images down to the Celery pipeline.
* **ML Engine (`worker`):** A Celery PyTorch execution thread wrapping our Custom `TrOCR` models, EasyOCR fallbacks, OpenCV blurring algorithms, and NAFNet image restorations. 
* **Message Broker (`redis`):** Manages the Celery Async task payloads.
* **Reverse Proxy (`nginx`):** Securely binds and routes `frontend` to `backend` APIs.
* **Observability (`prometheus` + `grafana`):** Scrapes metric payloads tracking Drift, Inference Latency, and Bounding Box detection directly from FastAPI.

## ⚙️ 2. The Active Learning Loop (Autonomous CT Engine)
1. **The Pipeline Trigger:** Any plate that scores `< 0.50` confidence drops into the `data/delta_batch` directory naturally by the Python API. 
2. **The Monitor:** You run `python3 backend/scripts/auto_retrain.py` in your Mac Terminal. It natively listens for `delta_batch` to hit 10 items.
3. **The Brain Transplant:** When it triggers, it PyTorch fine-tunes your 5GB `.safetensors` model instantly, wiping its own memory and organically hot-patching (`docker cp`) the new weights straight over the Docker Socket into the running API. 

---

## 🚀 3. How to Run the Project (Cold Boot)

For this repository's local macOS workflow, the fine-tuned model is intentionally kept out of the normal Docker build context. The containers are built first, then the model is copied into the running Docker environment. This is a practical local-dev strategy for this machine and project, not a statement that Docker can only handle small models.

In short:
- `docker compose build ...` builds the services without the large local model.
- `docker cp ...` moves the fine-tuned model into Docker's Linux-side storage once the containers exist.
- `docker compose restart ...` reloads the services so they pick up the copied weights.

Here is the exact sequence to properly boot the project from scratch anytime you close it:

#### Step 1: Boot Docker
```bash
docker compose build worker backend && docker compose up -d
```
*(This purposely builds isolated 100MB lightweight web servers natively WITHOUT the heavy weights).*

#### Step 2: Inject the Weights!
To avoid repeatedly loading the large model through the macOS-to-Docker filesystem bridge, explicitly copy the model over the Docker daemon socket into Docker's internal Linux storage, then reboot the services so they see it there:
```bash
docker cp backend/trocr_finetuned_final alpr-app-worker-1:/app/
docker cp backend/trocr_finetuned_final alpr-app-backend-1:/app/
docker cp backend/tasks.py alpr-app-worker-1:/app/tasks.py
docker cp backend/modules/mlops_metrics.py alpr-app-worker-1:/app/modules/mlops_metrics.py
docker cp backend/main.py alpr-app-backend-1:/app/main.py

docker compose restart worker backend nginx
```

#### Step 3: View the App!
- **Website:** `http://localhost:8080`
- **Grafana Metrics:** `http://localhost:3000`
- **API Status:** `http://localhost:8000/docs`

---

## 📈 4. Setting up Grafana
We natively injected deep PyTorch evaluations into the FastAPI router. Track your health natively using these Prometheus Queries:
* **Edit Distance:** `alpr_edit_distance_sum / alpr_edit_distance_count` 
* **Accuracy:** `alpr_character_accuracy`
* **CT Auto-Queue Counter:** `alpr_delta_queue_size`
* **YOLO BBox Density:** `alpr_bbox_count_sum / alpr_bbox_count_count`
* **CPU Inference Lag:** `alpr_inference_latency_seconds_sum / alpr_inference_latency_seconds_count`

---

## 🛑 5. The MacOS "VirtioFS" Resource Deadlock Explained

If you are examining the deployment steps and wondering why we manually inject (`docker cp`) the large PyTorch weights after the containers boot, rather than reading them directly from the shared Mac filesystem, the reason is a macOS Docker Desktop file-sharing bottleneck called `virtiofs`.

#### The Core Issue:
When Docker runs on a Mac, it creates a hidden Linux Virtual Machine. To allow the container to read files from your Mac's hard drive, Apple uses a synchronization bridge called `virtiofs`.

When our PyTorch engine tried to load the large `.safetensors` model through that bridge, the `virtiofs` layer became the weak point. On this setup, the shared-filesystem path was unreliable enough to trigger freezes and `Resource deadlock avoided` errors.

#### Our Architecture Fix:
To avoid that path during local development, we use `.dockerignore` to keep the fine-tuned model out of the normal image build and then copy the model into Docker-managed Linux storage after the containers start.

Instead, our deployment sequence explicitly uses `docker cp`. That copies the model once into Docker's internal Linux-side storage. After that, PyTorch loads the weights from inside Docker rather than repeatedly crossing the Mac-to-Linux filesystem bridge.

This is a good local strategy for this repo on macOS. It is not a universal Docker requirement, and other environments may use different model-delivery patterns such as Docker volumes or artifact stores.
