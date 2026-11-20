"""
Training pipeline for YOLOv8 Road Damage Detection on Indian road conditions.
Supports MPS (Apple Silicon), CUDA (NVIDIA GPU), and CPU compute devices.
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def detect_best_device():
    """Selects the best available accelerator."""
    if torch.cuda.is_available():
        print(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS acceleration.")
        return 'mps'
    print("No hardware accelerator found. Running on CPU.")
    return 'cpu'


def train(data_yaml="data.yaml", model_size="yolov8n.pt", epochs=25, imgsz=640, batch=16, device=None):
    if device is None:
        device = detect_best_device()

    print(f"Dataset config : {data_yaml}")
    print(f"Base backbone  : {model_size}")
    print(f"Epochs         : {epochs}")
    print(f"Resolution     : {imgsz}x{imgsz}")
    print(f"Batch size     : {batch}")
    print(f"Target device  : {device}\n")

    # Initialize model with pre-trained weights for transfer learning
    model = YOLO(model_size)

    # Launch training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        patience=10,        # Early stopping if no mAP improvement in 10 epochs
        save=True,
        project="runs/detect",
        name="road_damage_exp",
        exist_ok=True,
        plots=True          # Generates confusion matrix, PR curves, and loss plots
    )

    # Save best checkpoint and key evaluation curves
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/screenshots", exist_ok=True)

    possible_weight_dirs = [
        "runs/detect/road_damage_exp/weights",
        "runs/detect/runs/detect/road_damage_exp/weights"
    ]
    for wdir in possible_weight_dirs:
        best_candidate = os.path.join(wdir, "best.pt")
        if os.path.exists(best_candidate):
            shutil.copy(best_candidate, "models/best.pt")
            print("Successfully saved best weights to: models/best.pt")

            exp_parent = os.path.dirname(wdir)
            for f in os.listdir(exp_parent):
                if f.endswith(".png"):
                    shutil.copy(os.path.join(exp_parent, f), os.path.join("results/screenshots", f))
            print("Saved training metrics and performance curves to: results/screenshots/")
            break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Road Damage Model")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights: yolov8n.pt, yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: '0', 'cpu', 'mps'")

    args = parser.parse_args()
    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
