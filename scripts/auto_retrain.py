import sys
import time
import subprocess
import os
import datetime

# ──────────────────────────────────────────────────────────────────────
# MLOps Configuration
# ──────────────────────────────────────────────────────────────────────

# Directory name where the LoRA adapter weights are saved by the trainer.
# These files (adapter_model.bin + adapter_config.json) are tiny (~50MB)
# compared to the full model, and are injected into the containers at runtime.
LORA_ADAPTER_DIR  = "trocr_lora_adapters"

# The target path *inside* the containers where the adapters are mounted.
# This path matches the lora_weights Docker volume mount in docker-compose.yml.
CONTAINER_LORA_PATH = "/app/lora_adapters"


def check_delta_batch_size():
    """Checks the number of explicitly labeled images waiting in the active learning queue."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(backend_dir, "data/delta_metadata.csv")

    if not os.path.exists(csv_path):
        return 0

    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) <= 1:
            return 0
        return len(data) - 1  # Subtract 1 for the header


def trigger_retraining():
    """
    Kicks off the LoRA fine-tuning pipeline, then hot-swaps the adapter
    weights into the running Docker containers without a full rebuild.

    Flow:
      1. Run training/train_trocr_lora.py  →  produces trocr_lora_adapters/
      2. Copy the tiny adapter weight files into the worker and backend
         containers via `docker cp`  (NO rebuild required)
      3. Restart worker + backend so ocr_engine.py re-imports and picks
         up the new adapters from the volume
      4. Purge the delta CSV and images to reset the queue
    """
    print(f"\n[{datetime.datetime.now()}] 🚨 DATA DRIFT DETECTED!")
    print("Initiating LoRA Continuous Training (CT) Pipeline...")

    try:
        backend_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_dir  = os.path.join(backend_dir, "ALPR-app")  # Where docker-compose.yml lives
        delta_csv      = "data/delta_metadata.csv"
        delta_csv_path = os.path.join(backend_dir, delta_csv)

        if not os.path.exists(delta_csv_path):
            print("No Delta Batch CSV found yet! Waiting for more data...")
            return

        # ── Step 1: LoRA fine-tuning ──────────────────────────────────
        # Saves lightweight adapter weights only — NOT the full model.
        # Outputs: trocr_lora_adapters/adapter_model.bin
        #          trocr_lora_adapters/adapter_config.json
        print("\n[Step 1/4] Running LoRA fine-tuning...")
        # Use sys.executable so we call the same Python interpreter (conda env)
        # that is running this daemon — not the system `python3`.
        subprocess.run(
            [
                sys.executable, "training/train_trocr_lora.py",
                "--dataset", delta_csv,
                "--img_dir", "data/delta_batch",
                "--output",  LORA_ADAPTER_DIR,
                "--epochs",  "3",
            ],
            cwd=backend_dir,
            check=True
        )
        print("  ✅ LoRA training complete.")

        # ── Step 2: Inject adapter weights into containers ────────────
        # We copy only the tiny adapter directory via the Docker socket.
        # This avoids a full `docker build` and any VirtioFS deadlocks.
        adapter_local_path = os.path.join(backend_dir, LORA_ADAPTER_DIR)
        print(f"\n[Step 2/4] Copying LoRA adapters → containers ({CONTAINER_LORA_PATH})...")
        # Trailing slash on the source copies the *contents* of the directory
        # into CONTAINER_LORA_PATH directly, avoiding a nested sub-directory.
        subprocess.run(["docker", "cp", adapter_local_path + "/.", f"alpr-app-worker-1:{CONTAINER_LORA_PATH}"],  check=True)
        subprocess.run(["docker", "cp", adapter_local_path + "/.", f"alpr-app-backend-1:{CONTAINER_LORA_PATH}"], check=True)
        print("  ✅ Adapter weights injected.")

        # ── Step 3: Restart containers to reload ocr_engine.py ───────
        # docker compose must run from the directory containing docker-compose.yml
        print("\n[Step 3/4] Restarting worker and backend containers...")
        subprocess.run(["docker", "compose", "restart", "worker", "backend"], check=True, cwd=compose_dir)
        print("  ✅ Containers restarted with new LoRA weights.")

        # ── Step 4: Purge the delta batch to reset the queue ─────────
        print("\n[Step 4/4] Purging delta batch queue...")
        os.remove(delta_csv_path)
        import shutil
        delta_batch_dir = os.path.join(backend_dir, "data/delta_batch")
        if os.path.exists(delta_batch_dir):
            shutil.rmtree(delta_batch_dir)
            os.makedirs(delta_batch_dir, exist_ok=True)
        print("  ✅ Delta memory purged.")

        print("\n✅ LoRA MLOps loop closed successfully.\n")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed during execution. Error: {e}")


if __name__ == "__main__":
    print("================================================")
    print("  MLOps LoRA Continuous Training Daemon        ")
    print("================================================")
    print("Monitoring Delta Batch Queue Directory...")
    print("Data Drift Action Threshold: Queue > 10 Images")

    while True:
        queue_size = check_delta_batch_size()
        print(f"[{datetime.datetime.now().time()}] Current Delta Queue Size: {queue_size}/10 images")

        if queue_size >= 10:
            trigger_retraining()
            print("Sleeping for 10 minutes (cooldown) before monitoring again...")
            time.sleep(600)  # Cooldown period so we don't spam hardware on rapid drift
        else:
            time.sleep(30)  # Poll directory every 30 seconds when queue is below threshold

