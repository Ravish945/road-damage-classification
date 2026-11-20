"""
evaluate.py
Validation and Performance Evaluation for Trained Road Damage Model.

Author: Ravish Patel
Project: Road Damage Detection PBL
"""

import os
import argparse
from ultralytics import YOLO


def evaluate_model(weights_path="models/best.pt", data_yaml="data.yaml", imgsz=640, device="cpu"):
    """
    Evaluates the model on validation data and prints Precision, Recall, and mAP metrics.
    """
    if not os.path.exists(weights_path):
        print(f"[!] Model weights not found at '{weights_path}'.")
        print("    If you just trained or downloaded weights, verify the path or train first using train.py.")
        return

    print("=" * 60)
    print("ROAD DAMAGE DETECTION - MODEL EVALUATION")
    print("=" * 60)
    print(f"Model Weights: {weights_path}")
    print(f"Dataset      : {data_yaml}")
    print("=" * 60)

    model = YOLO(weights_path)

    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=device,
        project="results",
        name="evaluation",
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY METRICS")
    print("=" * 60)
    print(f"mAP@0.50       : {metrics.box.map50:.4f}")
    print(f"mAP@0.50-0.95  : {metrics.box.map:.4f}")
    print(f"Mean Precision : {metrics.box.mp:.4f}")
    print(f"Mean Recall    : {metrics.box.mr:.4f}")
    print("=" * 60)

    # Class-specific mAP breakdown
    class_names = model.names
    print("\nPer-Class Performance (mAP@0.50):")
    for idx, name in class_names.items():
        if idx < len(metrics.box.maps):
            print(f"  - {name:<25}: {metrics.box.maps[idx]:.4f}")

    print("\n[✓] Detailed evaluation metrics and plots stored in 'results/evaluation/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on Road Damage dataset")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to weights file")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', 'mps')")

    args = parser.parse_args()
    evaluate_model(weights_path=args.weights, data_yaml=args.data, device=args.device)
