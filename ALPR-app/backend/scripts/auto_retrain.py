import requests
import time
import subprocess
import os
import datetime

# MLOps Configuration
def check_delta_batch_size():
    """Checks the number of explicitly labeled images waiting in the active learning queue"""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(backend_dir, "data/delta_metadata.csv")
    
    if not os.path.exists(csv_path):
        return 0
        
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) <= 1:
            return 0
        return len(data) - 1 # Subtract 1 for the header

def trigger_retraining():
    """Kicks off the fine-tuning pipeline and hot-swaps the container"""
    print(f"\n[{datetime.datetime.now()}] 🚨 DATA DRIFT DETECTED!")
    print("Initiating Continuous Training (CT) Pipeline...")
    
    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        delta_csv = "data/delta_metadata.csv"
        delta_csv_path = os.path.join(backend_dir, delta_csv)
        
        if not os.path.exists(delta_csv_path):
            print("No Delta Batch CSV found yet! Waiting for more data...")
            return

        # Step 1: Run the training script blockingly with Active Learning injections
        process = subprocess.run(
            ["python3", "training/train_trocr.py", "--dataset", delta_csv, "--img_dir", "data/delta_batch"],
            cwd=backend_dir,
            check=True
        )
        print("\n✅ Training Complete. Injecting new weights and Hot-Swapping the Celery Docker Containers...")
        # Step 2: Inject weights over the Docker Socket directly into the isolated ext4 filesystem.
        # This completely avoids VirtioFS deadlocks AND stops the `docker build` Virtual hard drive exhaustion.
        subprocess.run(["docker", "cp", "trocr_finetuned_final", "alpr-app-worker-1:/app/"], check=True)
        subprocess.run(["docker", "cp", "trocr_finetuned_final", "alpr-app-backend-1:/app/"], check=True)
        
        # Step 3: Simply restart the fastAPI and Celery processes to load the newly placed files.
        subprocess.run(["docker", "compose", "restart", "worker", "backend"], check=True)
        
        # Step 3: Purge the Delta Batch CSV and physical images 
        os.remove(delta_csv_path)
        import shutil
        delta_batch_dir = os.path.join(backend_dir, "data/delta_batch")
        if os.path.exists(delta_batch_dir):
            shutil.rmtree(delta_batch_dir)
            os.makedirs(delta_batch_dir, exist_ok=True)
            
        print("✅ Container restarted & Delta memory purged! MLOps Retraining Loop Successfully Closed.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed during execution. Error: {e}")

if __name__ == "__main__":
    print("================================================")
    print(" MLOps Continuous Training Daemon Initiated ")
    print("================================================")
    print("Monitoring Delta Batch Queue Directory...")
    print("Data Drift Action Threshold: Queue > 10 Images")
    
    while True:
        queue_size = check_delta_batch_size()
        print(f"[{datetime.datetime.now().time()}] Current Delta Queue Size: {queue_size}/10 images")
        
        if queue_size >= 10:
            trigger_retraining()
            print("Sleeping for 10 minutes (cooldown) before monitoring again...")
            time.sleep(600)  # Cooldown period so we don't spam hardware
        
        time.sleep(30) # Poll directory every 30 seconds
