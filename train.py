import os
import shutil
import argparse
import torch
from ultralytics import YOLO

# Road Damage Detection Training using YOLOv8
# Trained on RDD2020 India dataset (D00, D10, D20, D40)

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Road Damage Dataset")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default=None, help="Device to train on (cuda, mps, cpu)")
    args = parser.parse_args()

    # Select device: Colab GPU if available, else Apple Silicon MPS or CPU
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Training {args.model} on device '{device}' for {args.epochs} epochs...")

    # Load pretrained YOLO weights
    model = YOLO(args.model)

    # Train model
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project="runs",
        name="road_damage_exp",
        exist_ok=True,
        patience=10,
        plots=True
    )

    # Copy best weights to models/
    os.makedirs("models", exist_ok=True)
    best_weights = "runs/road_damage_exp/weights/best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "models/best.pt")
        print("Copied best model weights to models/best.pt")

    # Copy plots to results/screenshots if present
    os.makedirs("results/screenshots", exist_ok=True)
    run_dir = "runs/road_damage_exp"
    if os.path.exists(run_dir):
        for f in os.listdir(run_dir):
            if f.endswith(".png"):
                shutil.copy(os.path.join(run_dir, f), os.path.join("results/screenshots", f))


if __name__ == "__main__":
    main()
