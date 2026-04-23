# LoRA Branch — Complete System Guide

## What We Have After the Merge

This branch (`LoRA`) is the **definitive, production-ready** branch. It combines:
- **LoRA branch**: LoRA fine-tuning pipeline + persistent volume caching
- **retrain branch**: Prometheus/Grafana monitoring + image routing classifier + MLOps metrics

---

## Full File Map

### 🐳 Docker Compose Services (7 total)

| Service | Image/Build | Port | Role |
|---|---|---|---|
| `redis` | `redis:alpine` | internal `6379` | Message broker / task queue |
| `backend` | `./backend` | `8000:8000` | FastAPI gateway — receives uploads, dispatches Celery tasks, exposes `/metrics` |
| `worker` | `./backend` | — | Celery worker — runs the full ALPR ML pipeline |
| `frontend` | `./frontend` | internal `80` | React + TypeScript UI |
| `nginx` | `nginx:alpine` | `8080:80` | Reverse proxy to frontend & backend |
| `prometheus` | `Dockerfile.prometheus` | `9090:9090` | Scrapes `/metrics` every 5s |
| `grafana` | `grafana/grafana` | `3000:3000` | Dashboard over Prometheus data |

### 📦 Persistent Docker Volumes (3 total)

| Volume | Mount Path | What it stores |
|---|---|---|
| `hf_cache` | `/app/.huggingface_cache` | Base TrOCR model weights — downloaded once, reused forever |
| `lora_weights` | `/app/lora_adapters` | LoRA fine-tuned adapter files — hot-swapped after each CT cycle |
| `shared_uploads` | `/app/shared_uploads` | Temporary image uploads (claim-check pattern) |

### 🧠 Backend Modules (`backend/modules/`)

| File | Role |
|---|---|
| `ocr_engine.py` | Loads base TrOCR from `hf_cache`, merges LoRA adapters from `lora_weights` if present, runs OCR inference |
| `classifier.py` | Simple heuristic classifier — decides which restoration module to use (brightness/blur thresholds) |
| `image_routing.py` | Advanced heuristic classifier — DCP fog detection, Sobel rain detection, low-light check |
| `mlops_metrics.py` | `calculate_mlops_scores()` — computes exact match, edit distance, character accuracy against ground truth |
| `detector.py` | YOLOv8 license plate detection |
| `restoration.py` | Orchestrates the restoration pipeline (calls darkir/dehaze/derain) |
| `nafnet.py` | NAFNet super-resolution / deblurring |
| `dark_ir.py` | DarkIR low-light enhancement |
| `dehaze.py` | DeHaze atmospheric correction |
| `derain.py` | DeRain streak removal |
| `rto_metadata.py` | Parses plate text to infer Indian RTO state/district |
| `vehicle_lookup.py` | Mock vehicle registration lookup |

### 📊 Monitoring Files

| File | Role |
|---|---|
| `prometheus.yml` | Scrape config — hits `backend:8000/metrics/` every 5s |
| `Dockerfile.prometheus` | Builds Prometheus with `prometheus.yml` baked in |

### 🔄 MLOps Scripts (`scripts/` + `training/`)

| File | Role |
|---|---|
| `scripts/auto_retrain.py` | Daemon — monitors delta queue, triggers LoRA training, hot-swaps weights into containers |
| `training/train_trocr_lora.py` | LoRA fine-tuning script — trains only adapter layers, saves `adapter_model.bin` + `adapter_config.json` |
| `scripts/prepare_dataset.py` | Dataset preparation helper |
| `scripts/simulate_traffic.py` | Sends synthetic traffic to the backend for load/drift testing |

---

## Prometheus Metrics Being Tracked

| Metric | Type | What it tracks |
|---|---|---|
| `alpr_ocr_confidence` | Gauge | TrOCR confidence score per prediction |
| `alpr_ocr_length` | Histogram | Character length of predicted plate text |
| `alpr_image_blur_score` | Gauge | Laplacian variance — camera focus degradation |
| `alpr_image_brightness` | Gauge | Average pixel intensity — day/night drift |
| `alpr_bbox_count` | Histogram | Plates detected per frame by YOLO |
| `alpr_inference_latency_seconds` | Histogram | End-to-end PyTorch inference time |
| `alpr_delta_queue_size` | Gauge | Items waiting for continuous training |
| `alpr_quarantine_queue_size` | Gauge | Items flagged for human review |
| `alpr_exact_match` | Gauge | Exact match vs ground truth (when label supplied) |
| `alpr_character_accuracy` | Gauge | CER-based accuracy vs ground truth |
| `alpr_edit_distance` | Histogram | Levenshtein distance vs ground truth |
| `alpr_invalid_format_total` | Counter | Predictions violating Indian plate format |
| `alpr_data_drift_score` | Gauge | Compound drift proxy score |

---

## How to Run Everything

### Prerequisites

```bash
# Install Python deps for the training daemon (runs on your Mac, not in Docker)
pip install transformers peft torch datasets
```

---

### Step 1 — Start the Docker Compose App

```bash
cd /Users/ekamsinghsethi/Desktop/DEP/ALPR-app

# First time: build all images
docker compose up -d --build

# On restarts (no code changes): no rebuild needed
docker compose up -d
```

**On first boot**, the `worker` and `backend` containers will download the TrOCR model from Hugging Face (~1GB) and cache it into the `hf_cache` volume. All subsequent restarts load from disk — no internet required.

---

### Step 2 — Check Services Are Up

| URL | What you'll see |
|---|---|
| `http://localhost:8080` | React frontend |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:8000/metrics/` | Raw Prometheus metrics |
| `http://localhost:9090` | Prometheus query UI |
| `http://localhost:3000` | Grafana dashboard (admin/admin on first login) |

---

### Step 3 — Run the MLOps Training Daemon (on your Mac)

The daemon polls the `data/delta_metadata.csv` queue every 30 seconds. When 10+ labeled images accumulate, it automatically triggers LoRA fine-tuning.

```bash
cd /Users/ekamsinghsethi/Desktop/DEP
python3 scripts/auto_retrain.py
```

You'll see output like:
```
================================================
  MLOps LoRA Continuous Training Daemon
================================================
[00:16:00] Current Delta Queue Size: 3/10 images
[00:16:30] Current Delta Queue Size: 3/10 images
...
[00:17:30] Current Delta Queue Size: 10/10 images

[2026-04-23 00:17:30] 🚨 DATA DRIFT DETECTED!
Initiating LoRA Continuous Training (CT) Pipeline...

[Step 1/4] Running LoRA fine-tuning...
  trainable params: 1,572,864 || all params: 334,143,744 || trainable%: 0.47
  Epoch [1/3] Step [5/10] Loss: 0.8421
  ...
  ✅ LoRA training complete.

[Step 2/4] Copying LoRA adapters → containers (/app/lora_adapters)...
  ✅ Adapter weights injected.

[Step 3/4] Restarting worker and backend containers...
  ✅ Containers restarted with new LoRA weights.

[Step 4/4] Purging delta batch queue...
  ✅ Delta memory purged.
✅ LoRA MLOps loop closed successfully.
```

---

### Step 4 — Manually Trigger a Fine-Tuning Run (for testing)

```bash
cd /Users/ekamsinghsethi/Desktop/DEP

# Run the LoRA trainer directly against your dataset
python3 training/train_trocr_lora.py \
  --dataset data/delta_metadata.csv \
  --img_dir  data/delta_batch \
  --output   trocr_lora_adapters \
  --epochs   3 \
  --lr       5e-4 \
  --lora_r   16

# Manually inject the adapter weights into the running containers
docker cp trocr_lora_adapters alpr-app-worker-1:/app/lora_adapters
docker cp trocr_lora_adapters alpr-app-backend-1:/app/lora_adapters

# Restart to load new weights
docker compose restart worker backend
```

---

### Step 5 — Simulate Traffic for Prometheus Data

```bash
cd /Users/ekamsinghsethi/Desktop/DEP
python3 scripts/simulate_traffic.py
```

This sends synthetic requests with ground-truth labels, which populates all the Prometheus accuracy and drift metrics.

---

### Step 6 — Set Up Grafana Dashboard

1. Open `http://localhost:3000` → login with `admin` / `admin`
2. Go to **Connections → Data Sources → Add data source**
3. Select **Prometheus** → URL: `http://prometheus:9090` → **Save & Test**
4. Go to **Dashboards → New → Add visualization**
5. Query example: `alpr_ocr_confidence` — shows real-time OCR confidence
6. Query example: `alpr_delta_queue_size` — shows when the next CT cycle will trigger

---

## How LoRA Injection Works at Runtime

```
Container starts
    │
    ▼
ocr_engine.py imports
    │
    ├── Loads base TrOCR from hf_cache volume (offline, ~1s)
    │
    ├── Checks /app/lora_adapters/adapter_config.json
    │       │
    │       ├── EXISTS → PeftModel.from_pretrained() → merge_and_unload()
    │       │             (merges LoRA deltas into base weights, ~2s)
    │       │
    │       └── ABSENT → Uses base model as-is (first boot, before any CT cycle)
    │
    └── model ready for inference
```

> [!IMPORTANT]
> `merge_and_unload()` permanently bakes the LoRA adapter weights into the base model in memory. This means **zero runtime overhead** at inference — the model runs at exactly the same speed as the base model, with no extra computation from LoRA branches.

---

## Commit the Merge

```bash
cd /Users/ekamsinghsethi/Desktop/DEP/ALPR-app
git add docker-compose.yml
git commit -m "Merge retrain into LoRA: add Prometheus/Grafana, classifier, image_routing, mlops_metrics"
```
