"""
voc_to_yolo.py
Dataset Preprocessing: Convert Pascal VOC XML Annotations to YOLO TXT format.

Author: Ravish Patel
Project: Road Damage Detection PBL (India Dataset)
"""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# -------------------------------------------------------------
# Class Mapping for Road Damage Detection (RDD India)
# -------------------------------------------------------------
# Standard RDD Damage Classes:
# D00: Longitudinal Crack
# D10: Transverse Crack
# D20: Alligator Crack (Fatigue cracking)
# D40: Pothole / Rutting / Bump
#
# We map sub-classes (e.g. D01 -> D00, D11 -> D10) for cleaner,
# robust detection categories.
CLASS_MAPPING = {
    'D00': 0,
    'D01': 0,   # Grouped with longitudinal crack
    'D10': 1,
    'D11': 1,   # Grouped with transverse crack
    'D20': 2,
    'D40': 3,
}

CLASS_NAMES = [
    'D00_Longitudinal_Crack',
    'D10_Transverse_Crack',
    'D20_Alligator_Crack',
    'D40_Pothole'
]


def find_data_paths(base_dir="."):
    """
    Automatically detects the location of train images and annotations,
    whether running locally or inside Google Drive (RoadDamagePBL).
    """
    # Case 1: Google Drive structure (raw_data/India/train)
    drive_train = os.path.join(base_dir, "raw_data", "India", "train")
    if os.path.exists(drive_train):
        img_dir = os.path.join(drive_train, "images")
        xml_dir = os.path.join(drive_train, "annotations", "xmls")
        return img_dir, xml_dir

    # Case 2: Local folder structure (train/images and train/annotations/xmls)
    local_train = os.path.join(base_dir, "train")
    if os.path.exists(local_train):
        img_dir = os.path.join(local_train, "images")
        xml_dir = os.path.join(local_train, "annotations", "xmls")
        return img_dir, xml_dir

    raise FileNotFoundError("Could not find 'train' directory. Ensure you are in the project folder.")


def convert_voc_bbox_to_yolo(size, box):
    """
    Convert bounding box from Pascal VOC (xmin, ymin, xmax, ymax)
    to normalized YOLO format (x_center, y_center, width, height).
    All output values are normalized between 0.0 and 1.0.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    # Calculate center coordinates and dimensions
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    # Normalize
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    # Clamp values to valid [0, 1] range to avoid edge overflow
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x, y, w, h


def parse_xml_annotation(xml_path):
    """
    Parse a single VOC XML file and return image size and object boxes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find('size')
    if size_elem is None:
        return None, []

    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)

    # Some images might have 0 in XML if corrupted
    if width <= 0 or height <= 0:
        return None, []

    boxes = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None or not name_elem.text:
            continue

        raw_name = name_elem.text.strip()
        if raw_name not in CLASS_MAPPING:
            continue  # Skip unmapped or irrelevant classes (like white line blur)

        class_id = CLASS_MAPPING[raw_name]
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        # Basic validity check
        if xmax <= xmin or ymax <= ymin:
            continue

        yolo_box = convert_voc_bbox_to_yolo((width, height), (xmin, xmax, ymin, ymax))
        boxes.append((class_id, yolo_box))

    return (width, height), boxes


def prepare_yolo_dataset(base_dir=".", output_dir="dataset", split_ratio=0.8, max_samples=None, random_seed=42):
    """
    Main function to process dataset:
    1. Parse XMLs and filter valid annotations.
    2. Split into train (80%) and validation (20%) sets.
    3. Copy images and write YOLO label txt files.
    4. Generate data.yaml configuration file for YOLOv8.
    """
    random.seed(random_seed)

    img_dir, xml_dir = find_data_paths(base_dir)
    print(f"[+] Found Images Directory: {img_dir}")
    print(f"[+] Found Annotations Directory: {xml_dir}")

    # Create destination directories
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    print(f"[+] Total XML files found: {len(xml_files)}")

    # Collect valid samples
    valid_samples = []
    class_stats = {name: 0 for name in CLASS_NAMES}

    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        img_name = base_name + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        size, boxes = parse_xml_annotation(xml_file)
        if size is None:
            continue

        # We keep samples with valid damages, plus a balanced ratio of background images
        valid_samples.append((img_path, base_name, boxes))

        for cid, _ in boxes:
            class_stats[CLASS_NAMES[cid]] += 1

    print("\n[+] Total Damage Instances Found:")
    for cname, count in class_stats.items():
        print(f"    - {cname}: {count}")

    # Optional limit for fast testing
    if max_samples and max_samples < len(valid_samples):
        valid_samples = valid_samples[:max_samples]
        print(f"[!] Limiting processing to {max_samples} samples.")

    # Shuffle for train/val split
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * split_ratio)
    train_set = valid_samples[:split_idx]
    val_set = valid_samples[split_idx:]

    print(f"\n[+] Splitting data: {len(train_set)} Train images, {len(val_set)} Validation images.")

    def export_split(dataset, target_img_dir, target_lbl_dir, split_name):
        print(f"[*] Processing {split_name} split...")
        count = 0
        for img_path, base_name, boxes in dataset:
            # 1. Copy image
            dest_img = os.path.join(target_img_dir, base_name + ".jpg")
            if not os.path.exists(dest_img):
                shutil.copy(img_path, dest_img)

            # 2. Write YOLO label file
            label_file = os.path.join(target_lbl_dir, base_name + ".txt")
            with open(label_file, "w") as f:
                for cid, (x, y, w, h) in boxes:
                    f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            count += 1
            if count % 1000 == 0:
                print(f"    {count}/{len(dataset)} processed...")

    export_split(train_set, train_img_dir, train_lbl_dir, "Train")
    export_split(val_set, val_img_dir, val_lbl_dir, "Validation")

    # Generate data.yaml
    yaml_path = os.path.join(base_dir, "data.yaml")
    # Resolve absolute path for YOLO configuration
    abs_output = os.path.abspath(output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"# YOLOv8 Road Damage Detection (India Dataset)\n")
        f.write(f"path: {abs_output}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names:\n")
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"  {idx}: {name}\n")

    print(f"\n[✓] Successfully created 'data.yaml' at {yaml_path}")
    print(f"[✓] YOLO dataset is ready in '{output_dir}/' folder!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VOC XML to YOLO format for Road Damage PBL")
    parser.add_argument("--base_dir", type=str, default=".", help="Base project directory containing train/ folder")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Destination directory for YOLO dataset")
    parser.add_argument("--split", type=float, default=0.8, help="Train/Val split ratio (default: 0.8)")
    parser.add_argument("--sample", type=int, default=None, help="Process only a sample count (for quick test)")

    args = parser.parse_args()
    prepare_yolo_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        split_ratio=args.split,
        max_samples=args.sample
    )
