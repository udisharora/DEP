from modules.nafnet import process_with_nafnet
from PIL import Image

print("[DEBUG] Creating test image...")
img = Image.new("RGB", (256, 256), color="white")
print("[DEBUG] Calling process_with_nafnet...")
result = process_with_nafnet(img)
print("[DEBUG] process_with_nafnet returned:", result)
