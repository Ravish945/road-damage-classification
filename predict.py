import os
import argparse
from ultralytics import YOLO

# Inference script to run detection on test images or folders

def get_severity(counts):
    potholes = counts.get("D40_Pothole", 0)
    alligators = counts.get("D20_Alligator_Crack", 0)
    cracks = counts.get("D00_Longitudinal_Crack", 0) + counts.get("D10_Transverse_Crack", 0)

    if potholes > 0 or alligators >= 2:
        return "HIGH (urgent patch needed)"
    elif alligators > 0 or cracks >= 2:
        return "MEDIUM (reseal surface)"
    elif cracks > 0:
        return "LOW (monitor)"
    return "NORMAL (road surface fine)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="demo_samples", help="image, video, or folder path")
    parser.add_argument("--weights", default="models/best.pt", help="trained weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: model weights not found at {args.weights}")
        return

    print(f"Loading {args.weights}...")
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project="results",
        name="predictions",
        exist_ok=True
    )

    print("\nDetection Summary:")
    for r in results:
        img_name = os.path.basename(r.path)
        counts = {}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names.get(cls_id, f"Class_{cls_id}")
            counts[name] = counts.get(name, 0) + 1

        severity = get_severity(counts)
        print(f"\n{img_name}:")
        if counts:
            for k, v in counts.items():
                print(f"  - {k}: {v}")
            print(f"  Priority: {severity}")
        else:
            print("  - No defects detected")

    print("\nVisual results saved in results/predictions/")


if __name__ == "__main__":
    main()
