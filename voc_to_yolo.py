"""
Dataset preparation: converts Pascal VOC XML bounding box annotations to normalized YOLO format.
Organizes images and label files into train and validation splits, generating data.yaml.
"""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET

# Road Damage Detection (RDD) India Class Mapping
# Grouping minor sub-classes (D01 -> D00, D11 -> D10) to strengthen sample support
CLASS_MAPPING = {
    'D00': 0,
    'D01': 0,   # Grouped with longitudinal cracks (wheel path / joint)
    'D10': 1,
    'D11': 1,   # Grouped with transverse cracks
    'D20': 2,   # Alligator crack
    'D40': 3,   # Pothole / rutting
}

CLASS_NAMES = [
    'D00_Longitudinal_Crack',
    'D10_Transverse_Crack',
    'D20_Alligator_Crack',
    'D40_Pothole'
]


def resolve_dataset_directories(base_dir="."):
    """Finds raw training images and XML annotations across Colab or local setups."""
    candidates = [
        (os.path.join(base_dir, "raw_data", "India", "train"), "Google Drive / raw_data structure"),
        (os.path.join(base_dir, "train"), "Local project root structure")
    ]
    for path, desc in candidates:
        img_dir = os.path.join(path, "images")
        xml_dir = os.path.join(path, "annotations", "xmls")
        if os.path.isdir(img_dir) and os.path.isdir(xml_dir):
            return img_dir, xml_dir

    raise FileNotFoundError(
        "Could not locate training images and XML annotations. "
        "Expected 'train/images' and 'train/annotations/xmls' in either './' or 'raw_data/India/'."
    )


def voc_to_yolo_bbox(img_w, img_h, xmin, xmax, ymin, ymax):
    """
    Converts absolute pixel coordinates to normalized center-based coordinates (0.0 to 1.0).
    """
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    # Clip to [0, 1] range to safeguard against boundary annotations
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h))
    )


def parse_xml_file(xml_path):
    """Extracts image dimensions and valid damage bounding boxes from a VOC XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None, []

    size_elem = root.find('size')
    if size_elem is None:
        return None, []

    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)
    if width <= 0 or height <= 0:
        return None, []

    boxes = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None or not name_elem.text:
            continue

        raw_label = name_elem.text.strip()
        if raw_label not in CLASS_MAPPING:
            continue

        cid = CLASS_MAPPING[raw_label]
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        if xmax <= xmin or ymax <= ymin:
            continue

        norm_box = voc_to_yolo_bbox(width, height, xmin, xmax, ymin, ymax)
        boxes.append((cid, norm_box))

    return (width, height), boxes


def split_samples(samples, split_ratio=0.8, strategy="chunk", chunk_size=50, seed=42):
    """
    Splits dataset into train and validation sets.
    - 'chunk' (recommended): Splits sequential video frames in contiguous blocks (e.g. 50 frames).
      This prevents adjacent frames from the same road section leaking into both train and validation splits.
    - 'random': Standard random shuffle.
    """
    if strategy == "chunk":
        # Group samples in blocks to respect temporal video continuity
        chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
        random.seed(seed)
        random.shuffle(chunks)

        train_set, val_set = [], []
        target_train_count = int(len(samples) * split_ratio)
        current_count = 0

        for ch in chunks:
            if current_count + len(ch) <= target_train_count:
                train_set.extend(ch)
                current_count += len(ch)
            else:
                val_set.extend(ch)
        return train_set, val_set
    else:
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        idx = int(len(shuffled) * split_ratio)
        return shuffled[:idx], shuffled[idx:]


def process_dataset(base_dir=".", output_dir="dataset", split_ratio=0.8, split_strategy="chunk", max_samples=None):
    img_dir, xml_dir = resolve_dataset_directories(base_dir)
    print(f"Reading images from: {img_dir}")
    print(f"Reading XML annotations from: {xml_dir}")

    # Create destination directories
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    print(f"Found {len(xml_files)} annotation files.")

    samples = []
    class_tallies = {name: 0 for name in CLASS_NAMES}

    for xml_path in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = os.path.join(img_dir, base_name + ".jpg")
        if not os.path.exists(img_path):
            continue

        size, boxes = parse_xml_file(xml_path)
        if size is None:
            continue

        # Retain all images: damaged frames as positive samples,
        # and undamaged frames as negative background samples (empty txt) to minimize false positives.
        samples.append((img_path, base_name, boxes))

        for cid, _ in boxes:
            class_tallies[CLASS_NAMES[cid]] += 1

    print("\nAnnotation Instance Counts:")
    for name, cnt in class_tallies.items():
        print(f"  {name:<25}: {cnt}")

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"Limited run: processing {max_samples} samples.")

    train_data, val_data = split_samples(samples, split_ratio=split_ratio, strategy=split_strategy)
    print(f"\nSplit strategy '{split_strategy}': {len(train_data)} train images, {len(val_data)} validation images.")

    def write_split(dataset, out_img_dir, out_lbl_dir, name):
        print(f"Exporting {name} split ({len(dataset)} items)...")
        for i, (src_img, bname, bboxes) in enumerate(dataset):
            dst_img = os.path.join(out_img_dir, bname + ".jpg")
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)

            # Write YOLO label file (empty if undamaged background sample)
            dst_lbl = os.path.join(out_lbl_dir, bname + ".txt")
            with open(dst_lbl, "w") as f:
                for cid, (x, y, w, h) in bboxes:
                    f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    write_split(train_data, train_img_dir, train_lbl_dir, "Train")
    write_split(val_data, val_img_dir, val_lbl_dir, "Validation")

    yaml_path = os.path.join(base_dir, "data.yaml")
    abs_out = os.path.abspath(output_dir)
    with open(yaml_path, "w") as f:
        f.write("# YOLOv8 Dataset Configuration - Road Damage Detection (India)\n")
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"  {idx}: {name}\n")

    print(f"\nConfiguration file written to: {yaml_path}")
    print(f"Dataset successfully staged in '{output_dir}/'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert VOC XML to YOLO format with leakage prevention")
    parser.add_argument("--base_dir", type=str, default=".", help="Project base path containing train/ or raw_data/")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output path for YOLO dataset")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val ratio (default: 0.8)")
    parser.add_argument("--split_strategy", type=str, default="chunk", choices=["chunk", "random"],
                        help="Splitting strategy: 'chunk' blocks sequential video frames to avoid leakage")
    parser.add_argument("--sample", type=int, default=None, help="Process limited sample count for quick tests")

    args = parser.parse_args()
    process_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        split_ratio=args.split,
        split_strategy=args.split_strategy,
        max_samples=args.sample
    )
