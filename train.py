import os
import shutil
import argparse
import torch
from ultralytics import YOLO

# Train YOLOv8 on Indian road damage dataset (RDD India)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="dataset config")
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained model")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Use CUDA if on Colab, else MPS on Mac, else CPU
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Starting training on {device} ({args.epochs} epochs)...")

    model = YOLO(args.model)
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

    # Save best checkpoint to models/
    os.makedirs("models", exist_ok=True)
    best_pt = "runs/road_damage_exp/weights/best.pt"
    if os.path.exists(best_pt):
        shutil.copy(best_pt, "models/best.pt")
        print("Saved best model to models/best.pt")

    # Copy generated plots to results/screenshots
    os.makedirs("results/screenshots", exist_ok=True)
    exp_dir = "runs/road_damage_exp"
    if os.path.exists(exp_dir):
        for f in os.listdir(exp_dir):
            if f.endswith(".png"):
                shutil.copy(os.path.join(exp_dir, f), os.path.join("results/screenshots", f))


if __name__ == "__main__":
    main()
