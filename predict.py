"""
Run inference on single images, videos, or directories of road photos.
Outputs annotated detection visualizations and logs summary statistics.
"""

import os
import argparse
from ultralytics import YOLO

# Display names for terminal summary
CLASS_LABELS = {
    "D00_Longitudinal_Crack": "Longitudinal Crack",
    "D10_Transverse_Crack": "Transverse Crack",
    "D20_Alligator_Crack": "Alligator (Fatigue) Crack",
    "D40_Pothole": "Pothole / Rutting"
}


def evaluate_severity(counts):
    """
    Categorizes pavement condition based on detected distresses:
    - High: Any pothole (D40) or multiple structural fatigue cracks (D20).
    - Medium: Longitudinal or transverse cracks requiring joint resealing.
    - Low/Normal: Minor or no structural defects detected.
    """
    potholes = counts.get("D40_Pothole", 0)
    alligators = counts.get("D20_Alligator_Crack", 0)
    cracks = counts.get("D00_Longitudinal_Crack", 0) + counts.get("D10_Transverse_Crack", 0)

    if potholes > 0 or alligators >= 2:
        return "HIGH - Immediate repair required (pothole or severe fatigue)"
    elif alligators > 0 or cracks >= 2:
        return "MEDIUM - Surface resealing and crack filling needed"
    elif cracks > 0:
        return "LOW - Minor surface distress monitoring"
    return "NORMAL - Pavement in acceptable condition"


def run_inference(source="demo_samples", weights="models/best.pt", conf=0.25, save_dir="results/predictions"):
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Model checkpoint '{weights}' not found. Please train the model with train.py "
            f"or specify an existing weights path using --weights."
        )

    print(f"Loading weights from: {weights}")
    print(f"Running inference on: {source} (conf >= {conf})")

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project="results",
        name="predictions",
        exist_ok=True
    )

    print("\n--- Detection Summary ---")
    for r in results:
        img_name = os.path.basename(r.path)
        counts = {}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names.get(cls_id, f"Class_{cls_id}")
            counts[name] = counts.get(name, 0) + 1

        status = evaluate_severity(counts)
        print(f"\n{img_name}:")
        if counts:
            for cname, ccount in counts.items():
                label = CLASS_LABELS.get(cname, cname)
                print(f"  - {label}: {ccount}")
            print(f"  Condition: {status}")
        else:
            print("  - No distresses detected.")

    print(f"\nSaved labeled detection images to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Damage Detection CLI Inference")
    parser.add_argument("--source", type=str, default="demo_samples", help="Path to image, video, or folder")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    run_inference(source=args.source, weights=args.weights, conf=args.conf)
