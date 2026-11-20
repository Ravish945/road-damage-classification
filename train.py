"""
train.py
Model Training Script for Road Damage Detection using YOLOv8.

Author: Ravish Patel
Project: Road Damage Detection PBL
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def get_optimal_device():
    """
    Auto-detect the best compute device:
    1. CUDA (NVIDIA GPU - e.g. Google Colab T4/V100)
    2. MPS (Apple Silicon GPU - M1/M2/M3)
    3. CPU (Standard fallback)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[+] Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("[+] Using Apple Silicon MPS GPU")
    else:
        device = 'cpu'
        print("[!] No GPU found. Running on CPU.")
    return device


def train_model(data_yaml="data.yaml", model_size="yolov8n.pt", epochs=25, imgsz=640, batch=16, device=None):
    """
    Trains YOLOv8 on the prepared Road Damage dataset.
    """
    if device is None:
        device = get_optimal_device()

    print("=" * 60)
    print("ROAD DAMAGE DETECTION - TRAINING PIPELINE (YOLOv8)")
    print("=" * 60)
    print(f"Data config : {data_yaml}")
    print(f"Base model  : {model_size}")
    print(f"Epochs      : {epochs}")
    print(f"Image Size  : {imgsz}")
    print(f"Batch Size  : {batch}")
    print(f"Device      : {device}")
    print("=" * 60)

    # Load pre-trained YOLOv8 weights (transfer learning)
    model = YOLO(model_size)

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        patience=10,        # Early stopping patience
        save=True,
        project="runs/detect",
        name="road_damage_exp",
        exist_ok=True,
        plots=True          # Generate confusion matrix and PR curves
    )

    # Organize outputs
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/screenshots", exist_ok=True)

    # YOLOv8 might save to runs/detect/road_damage_exp or nested
    possible_weight_dirs = [
        "runs/detect/road_damage_exp/weights",
        "runs/detect/runs/detect/road_damage_exp/weights"
    ]
    for wdir in possible_weight_dirs:
        best_candidate = os.path.join(wdir, "best.pt")
        if os.path.exists(best_candidate):
            shutil.copy(best_candidate, "models/best.pt")
            print(f"\n[✓] Saved best model weights to: models/best.pt")
            # Also copy plots
            exp_parent = os.path.dirname(wdir)
            for f in os.listdir(exp_parent):
                if f.endswith(".png"):
                    shutil.copy(os.path.join(exp_parent, f), os.path.join("results/screenshots", f))
            break

    print("[✓] Saved training metrics and charts to 'results/screenshots/'")
    print("\nTraining completed successfully!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Road Damage Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (use 8 or 16 depending on GPU VRAM)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: '0', 'cpu', 'mps'")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
