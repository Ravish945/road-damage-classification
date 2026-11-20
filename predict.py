"""
predict.py
Inference Script: Detect and Classify Road Damages on Images and Videos.

Author: Ravish Patel
Project: Road Damage Detection PBL
"""

import os
import argparse
from ultralytics import YOLO


# Friendly labels for reporting
FRIENDLY_NAMES = {
    "D00_Longitudinal_Crack": "Longitudinal Crack",
    "D10_Transverse_Crack": "Transverse Crack",
    "D20_Alligator_Crack": "Alligator (Fatigue) Crack",
    "D40_Pothole": "Pothole / Rutting"
}


def assess_damage_severity(detection_counts):
    """
    Simple severity heuristic for road maintenance:
    - High: Potholes present (immediate hazard) or multiple alligator cracks.
    - Medium: Longitudinal or Transverse cracks present.
    - Low / Good: No significant structural damages detected.
    """
    potholes = detection_counts.get("D40_Pothole", 0)
    alligators = detection_counts.get("D20_Alligator_Crack", 0)
    cracks = detection_counts.get("D00_Longitudinal_Crack", 0) + detection_counts.get("D10_Transverse_Crack", 0)

    if potholes > 0 or alligators >= 2:
        return "HIGH - Immediate repair required (potholes/severe fatigue)"
    elif alligators > 0 or cracks >= 2:
        return "MEDIUM - Surface resealing and crack filling needed"
    elif cracks > 0:
        return "LOW - Minor crack monitoring"
    else:
        return "NORMAL - Road surface in acceptable condition"


def run_inference(source="demo_samples", weights="models/best.pt", conf=0.25, save_dir="results/predictions"):
    """
    Runs YOLOv8 object detection on images or videos.
    """
    if not os.path.exists(weights):
        # If models/best.pt is not trained yet, fall back to yolov8n.pt for testing
        print(f"[!] Custom weights '{weights}' not found.")
        print("[*] Falling back to default 'yolov8n.pt' for code verification.")
        weights = "yolov8n.pt"

    print("=" * 60)
    print("ROAD DAMAGE DETECTION - INFERENCE")
    print("=" * 60)
    print(f"Source       : {source}")
    print(f"Model        : {weights}")
    print(f"Confidence   : {conf}")
    print(f"Save Path    : {save_dir}")
    print("=" * 60)

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project="results",
        name="predictions",
        exist_ok=True
    )

    print("\n[+] Detection Summary:")
    for idx, r in enumerate(results):
        img_name = os.path.basename(r.path)
        counts = {}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        severity = assess_damage_severity(counts)
        print(f"\nImage: {img_name}")
        if counts:
            for cname, ccount in counts.items():
                friendly = FRIENDLY_NAMES.get(cname, cname)
                print(f"  • {friendly}: {ccount}")
            print(f"  --> Severity Status: {severity}")
        else:
            print("  • No road damage detected.")

    print(f"\n[✓] Visualized prediction images saved to '{save_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Road Damage Detection Inference")
    parser.add_argument("--source", type=str, default="demo_samples", help="Path to image, video, or folder")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    run_inference(source=args.source, weights=args.weights, conf=args.conf)
