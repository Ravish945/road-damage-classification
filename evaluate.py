import os
import argparse
from ultralytics import YOLO

# Evaluate trained YOLOv8 model on validation or test split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt", help="path to best.pt")
    parser.add_argument("--data", default="data.yaml", help="dataset config")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: model weights not found at {args.weights}")
        print("Please train the model first with train.py")
        return

    print(f"Evaluating {args.weights} on {args.split} split...")
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=args.device,
        project="results",
        name="evaluation",
        exist_ok=True
    )

    print(f"\n--- Validation Results ({args.split}) ---")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50-95:   {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}\n")

    print("Class breakdown (mAP50):")
    for idx, name in model.names.items():
        if idx < len(metrics.box.maps):
            print(f"  {name}: {metrics.box.maps[idx]:.4f}")


if __name__ == "__main__":
    main()
