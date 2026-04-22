import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import csv
import argparse

def process_voc_folder(voc_dir, out_img_dir, out_csv_path):
    # Create the output directories if they don't exist
    os.makedirs(out_img_dir, exist_ok=True)
    csv_dir = os.path.dirname(out_csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        
    # Recursively find XML files
    xml_files = glob.glob(os.path.join(voc_dir, "**", "*.xml"), recursive=True)
    print(f"Found {len(xml_files)} XML annotation files in '{voc_dir}'")
    
    success_count = 0
    file_exists = os.path.exists(out_csv_path)
    
    # Open the CSV file in append mode
    with open(out_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['filename', 'text']) # Write header if this is a new file
            
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                filename_node = root.find("filename")
                if filename_node is None:
                    continue
                filename = filename_node.text
                
                # Loop over every object bounding box tagged in the image
                for obj in root.findall("object"):
                    name_tag = obj.find("name")
                    if name_tag is None or not name_tag.text:
                        continue
                    text_label = name_tag.text.strip().upper().replace(" ", "")
                    # Basic filter just in case the label says "license_plate" instead of the text
                    if text_label.lower() in ['license_plate', 'number_plate', 'car']:
                        continue
                        
                    bndbox = obj.find("bndbox")
                    xmin = int(float(bndbox.find("xmin").text))
                    ymin = int(float(bndbox.find("ymin").text))
                    xmax = int(float(bndbox.find("xmax").text))
                    ymax = int(float(bndbox.find("ymax").text))
                    
                    # 1. Attempt to resolve the image path safely
                    img_path = os.path.join(voc_dir, filename)
                    if not os.path.exists(img_path):
                        # Some XMLs have mismatched filenames, attempt to guess by trimming
                        base = os.path.basename(xml_file)
                        img_path = os.path.join(voc_dir, base.replace('.xml', '.jpeg'))
                        if not os.path.exists(img_path):
                            img_path = os.path.join(voc_dir, base.replace('.xml', ''))
                            if not os.path.exists(img_path):
                                print(f"Skipping {base}, image not found.")
                                continue
                            
                    # 2. Extract and save the cropped image
                    with Image.open(img_path) as img:
                        # Convert to RGB to standardize channels
                        img = img.convert("RGB")
                        width, height = img.size
                        # Prevent negative coordinates or going completely out of bounds
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(width, xmax), min(height, ymax)
                        
                        crop = img.crop((xmin, ymin, xmax, ymax))
                        
                        # Secure unique filename
                        out_filename = f"crop_{success_count}.jpg"
                        crop.save(os.path.join(out_img_dir, out_filename))
                        
                        writer.writerow([out_filename, text_label])
                        success_count += 1
            except Exception as e:
                print(f"Warning: Issue parsing {xml_file} -> {e}")
                
    print(f"Successfully extracted {success_count} plate crops to '{out_img_dir}'")
    print(f"Metadata appended into '{out_csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PASCAL VOC XML to CSV Data Preparer")
    parser.add_argument("--input_dir", required=True, help="Directory containing pairs of .xml and .jpg files")
    parser.add_argument("--out_img_dir", required=True, help="Output destination for cropped plate images")
    parser.add_argument("--out_csv", required=True, help="Output destination for metadata CSV")
    
    args = parser.parse_args()
    process_voc_folder(args.input_dir, args.out_img_dir, args.out_csv)
