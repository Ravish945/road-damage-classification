"""
Evaluates trained YOLOv8 road damage model checkpoints on validation or test splits.
Outputs mAP@0.50, mAP@0.50:0.95, Precision, and Recall metrics per class.
"""

import os
import argparse
from ultralytics import YOLO


def run_evaluation(weights_path="models/best.pt", data_yaml="data.yaml", split="val", imgsz=640, device="cpu"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file '{weights_path}' does not exist. Train the model first using train.py."
        )

    print(f"Loading checkpoint : {weights_path}")
    print(f"Dataset config     : {data_yaml}")
    print(f"Evaluation split   : {split}")
    print(f"Resolution         : {imgsz}x{imgsz}\n")

    model = YOLO(weights_path)

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=device,
        project="results",
        name="evaluation",
        exist_ok=True
    )

    print("\n--- Quantitative Evaluation Metrics ---")
    print(f"mAP@0.50        : {metrics.box.map50:.4f}")
    print(f"mAP@0.50-0.95   : {metrics.box.map:.4f}")
    print(f"Mean Precision  : {metrics.box.mp:.4f}")
    print(f"Mean Recall     : {metrics.box.mr:.4f}\n")

    print("Per-Class mAP@0.50 Breakdown:")
    for idx, name in model.names.items():
        if idx < len(metrics.box.maps):
            print(f"  {name:<25}: {metrics.box.maps[idx]:.4f}")

    print("\nDetailed evaluation outputs and PR curves stored in 'results/evaluation/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Road Damage Model")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to weights file")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', 'mps')")

    args = parser.parse_args()
    run_evaluation(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device
    )
