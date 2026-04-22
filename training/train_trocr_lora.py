"""
LoRA Fine-Tuning Script for TrOCR (microsoft/trocr-base-printed)
-----------------------------------------------------------------
Uses PEFT (Parameter Efficient Fine-Tuning) to train only a small
set of low-rank adapter weights instead of all model parameters.

Output: A directory of LoRA adapter weights (adapter_model.safetensors +
        adapter_config.json) that can be hot-injected at inference
        time without touching the frozen base model.

Usage:
    python3 training/train_trocr_lora.py \
        --dataset data/delta_metadata.csv \
        --img_dir  data/delta_batch \
        --output   trocr_lora_adapters \
        --epochs   3

Fixed bugs:
    - VisionEncoderDecoderConfig.pad_token_id AttributeError (set explicitly)
    - decoder_start_token_id not set (causes silent bad generation)
    - Forward pass generates decoder_input_ids from labels via shift_tokens_right
"""

import argparse
import os
import csv
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType


# ──────────────────────────────────────────────
# 1. CLI Arguments
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LoRA fine-tuning for TrOCR")
parser.add_argument("--dataset",    required=True,            help="Path to delta_metadata.csv")
parser.add_argument("--img_dir",    required=True,            help="Directory containing plate images")
parser.add_argument("--output",     default="trocr_lora_adapters", help="Output dir for LoRA adapter weights")
parser.add_argument("--epochs",     type=int,   default=3,    help="Number of training epochs")
parser.add_argument("--lr",         type=float, default=5e-4, help="Learning rate")
parser.add_argument("--batch",      type=int,   default=4,    help="Batch size")
parser.add_argument("--lora_r",     type=int,   default=16,   help="LoRA rank")
parser.add_argument("--lora_alpha", type=int,   default=32,   help="LoRA alpha scaling factor")
args = parser.parse_args()


# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class PlateDataset(Dataset):
    """
    Reads (image_filename, plate_text) pairs from the delta CSV and
    returns (pixel_values, labels) tensors ready for TrOCR training.

    CSV columns expected: filename, text
    (Written by tasks.py active-learning router)
    """
    def __init__(self, csv_path, img_dir, processor):
        self.samples   = []
        self.img_dir   = img_dir
        self.processor = processor

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(img_dir, row["filename"])
                if os.path.exists(img_path):
                    self.samples.append((img_path, row["text"]))
                else:
                    print(f"  [WARN] Image not found, skipping: {img_path}")

        if len(self.samples) == 0:
            print("  [ERROR] No valid training samples found. Check --img_dir and CSV paths.")
            sys.exit(1)

        print(f"  Loaded {len(self.samples)} training samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_text = self.samples[idx]

        # Load and convert image to RGB
        image = Image.open(img_path).convert("RGB")

        # Encode image into pixel values for the TrOCR encoder
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize the ground-truth plate text for the decoder labels
        labels = self.processor.tokenizer(
            label_text,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Replace pad token id with -100 so CrossEntropyLoss ignores padding positions
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels


# ──────────────────────────────────────────────
# 3. Load Base Model and Apply LoRA
# ──────────────────────────────────────────────
# When running locally on Mac: default to DEP/.huggingface_cache
# When running inside Docker: HF_HOME=/app/.huggingface_cache is set by docker-compose
default_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".huggingface_cache")
)
cache_dir = os.environ.get("HF_HOME", default_cache)
device    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[LoRA Trainer] Device: {device}")
print(f"[LoRA Trainer] Cache dir: {cache_dir}")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed",
    cache_dir=cache_dir,
    local_files_only=False   # Allow download only if cache is missing
)

base_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed",
    cache_dir=cache_dir,
    local_files_only=False
)

# ── FIX 1: Set required token IDs on the model config ────────────────────────
# VisionEncoderDecoderModel.forward() reads pad_token_id and
# decoder_start_token_id from model.config. Without this, the forward pass
# raises: AttributeError: 'VisionEncoderDecoderConfig' has no attribute 'pad_token_id'
base_model.config.pad_token_id          = processor.tokenizer.pad_token_id
base_model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
# Also set on the decoder config for completeness
base_model.config.decoder.bos_token_id  = processor.tokenizer.bos_token_id
base_model.config.decoder.pad_token_id  = processor.tokenizer.pad_token_id

# LoRA configuration — target the query and value projection matrices inside
# the decoder attention layers (most impactful for sequence generation tasks)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# Wrap the base model: only LoRA adapter parameters will have requires_grad=True
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(device)


# ──────────────────────────────────────────────
# 4. DataLoader
# ──────────────────────────────────────────────
dataset    = PlateDataset(args.dataset, args.img_dir, processor)
dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)


# ──────────────────────────────────────────────
# 5. Training Loop
# ──────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
)

print(f"\n[LoRA Trainer] Starting LoRA fine-tuning for {args.epochs} epoch(s)...")

model.train()
for epoch in range(args.epochs):
    total_loss = 0.0

    for step, (pixel_values, labels) in enumerate(dataloader):
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        # ── FIX 2: Pass decoder_input_ids explicitly ──────────────────────────
        # VisionEncoderDecoderModel.forward() requires decoder_input_ids when
        # labels are provided (it does not auto-generate them in all versions).
        # We shift the labels right to produce the teacher-forced decoder input.
        # Positions where labels == -100 (padding) are replaced with pad_token_id.
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = processor.tokenizer.pad_token_id
        # Shift right: prepend bos, drop last token
        decoder_input_ids = torch.cat([
            torch.full(
                (decoder_input_ids.size(0), 1),
                processor.tokenizer.bos_token_id,
                dtype=torch.long,
                device=device
            ),
            decoder_input_ids[:, :-1]
        ], dim=1)

        outputs = model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 5 == 0 or (step + 1) == len(dataloader):
            print(f"  Epoch [{epoch+1}/{args.epochs}] "
                  f"Step [{step+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    print(f"  ✅ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")


# ──────────────────────────────────────────────
# 6. Save ONLY the LoRA Adapter Weights
# ──────────────────────────────────────────────
# Saves adapter_model.safetensors + adapter_config.json — NOT the full model.
# These tiny files (<50MB) are injected at runtime on top of the frozen base.
os.makedirs(args.output, exist_ok=True)
model.save_pretrained(args.output)

print(f"\n✅ LoRA adapter weights saved to: {os.path.abspath(args.output)}")
print("   Files: adapter_model.safetensors, adapter_config.json")
print("   These will be hot-injected into the Docker containers.\n")
