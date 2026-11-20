import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
import argparse

# Converts RDD India Pascal VOC XML annotations to YOLO format.
# Also splits data into train/val and generates data.yaml.

CLASS_MAP = {
    'D00': 0, 'D01': 0,  # Longitudinal crack
    'D10': 1, 'D11': 1,  # Transverse crack
    'D20': 2,            # Alligator crack
    'D40': 3             # Pothole
}

CLASS_NAMES = [
    'D00_Longitudinal_Crack',
    'D10_Transverse_Crack',
    'D20_Alligator_Crack',
    'D40_Pothole'
]


def parse_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find('size')
        if size is None:
            return None, []
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w <= 0 or h <= 0:
            return None, []

        boxes = []
        for obj in root.findall('object'):
            label = obj.find('name').text.strip() if obj.find('name') is not None else ""
            if label not in CLASS_MAP:
                continue

            bnd = obj.find('bndbox')
            if bnd is None:
                continue

            xmin = float(bnd.find('xmin').text)
            xmax = float(bnd.find('xmax').text)
            ymin = float(bnd.find('ymin').text)
            ymax = float(bnd.find('ymax').text)

            if xmax <= xmin or ymax <= ymin:
                continue

            # Normalized YOLO coordinates (center x, center y, width, height)
            cx = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / w))
            cy = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / h))
            bw = max(0.0, min(1.0, (xmax - xmin) / w))
            bh = max(0.0, min(1.0, (ymax - ymin) / h))

            boxes.append((CLASS_MAP[label], (cx, cy, bw, bh)))

        return (w, h), boxes
    except Exception:
        return None, []


def split_data(samples, split_ratio=0.8, strategy="chunk", chunk_size=50, seed=42):
    # Chunk splitting keeps consecutive video frames together so we don't leak train frames into val
    if strategy == "chunk":
        chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
        random.seed(seed)
        random.shuffle(chunks)

        train_set, val_set = [], []
        target = int(len(samples) * split_ratio)
        for c in chunks:
            if len(train_set) + len(c) <= target:
                train_set.extend(c)
            else:
                val_set.extend(c)
        return train_set, val_set
    else:
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        idx = int(len(shuffled) * split_ratio)
        return shuffled[:idx], shuffled[idx:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="project base path")
    parser.add_argument("--output_dir", default="dataset", help="where to save yolo dataset")
    parser.add_argument("--split", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--split_strategy", default="chunk", choices=["chunk", "random"])
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    # Look for train images and xmls in common paths
    candidates = [
        (os.path.join(args.base_dir, "raw_data", "India", "train")),
        (os.path.join(args.base_dir, "train"))
    ]
    img_dir, xml_dir = None, None
    for p in candidates:
        i_dir = os.path.join(p, "images")
        x_dir = os.path.join(p, "annotations", "xmls")
        if os.path.isdir(i_dir) and os.path.isdir(x_dir):
            img_dir, xml_dir = i_dir, x_dir
            break

    if not img_dir:
        print("Error: Could not find train/images and train/annotations/xmls folders")
        return

    print(f"Reading images from {img_dir}")
    print(f"Reading annotations from {xml_dir}")

    # Create YOLO folder structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    print(f"Found {len(xml_files)} annotation files")

    samples = []
    counts = {name: 0 for name in CLASS_NAMES}

    for x_path in xml_files:
        name = os.path.splitext(os.path.basename(x_path))[0]
        img_path = os.path.join(img_dir, name + ".jpg")
        if not os.path.exists(img_path):
            continue

        size, boxes = parse_xml(x_path)
        if size is None:
            continue

        # Keep both damaged and clean images (empty txt file = negative background sample)
        samples.append((img_path, name, boxes))
        for cid, _ in boxes:
            counts[CLASS_NAMES[cid]] += 1

    print("\nClass counts in dataset:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if args.sample and args.sample < len(samples):
        samples = samples[:args.sample]

    train_data, val_data = split_data(samples, args.split, args.split_strategy)
    print(f"\nSplit ({args.split_strategy}): {len(train_data)} train images, {len(val_data)} val images")

    def save_split(dataset, split_name):
        img_out = os.path.join(args.output_dir, "images", split_name)
        lbl_out = os.path.join(args.output_dir, "labels", split_name)
        for src_img, bname, boxes in dataset:
            dst_img = os.path.join(img_out, bname + ".jpg")
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)

            # Write label file (empty for undamaged images)
            dst_lbl = os.path.join(lbl_out, bname + ".txt")
            with open(dst_lbl, "w") as f:
                for cid, (x, y, w, h) in boxes:
                    f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Writing files...")
    save_split(train_data, "train")
    save_split(val_data, "val")

    # Generate data.yaml
    yaml_path = os.path.join(args.base_dir, "data.yaml")
    abs_out = os.path.abspath(args.output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")

    print(f"Done! Created dataset in '{args.output_dir}/' and saved '{yaml_path}'")


if __name__ == "__main__":
    main()
