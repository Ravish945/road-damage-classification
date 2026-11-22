import os
import urllib.request

MODELS = {
    "models/rdd_yolov8s.pt": "https://huggingface.co/dronefreak/rdd2022-yolov8s/resolve/main/best.pt",
    "models/pothole_yolov8.pt": "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
}

def main():
    os.makedirs("models", exist_ok=True)
    for path, url in MODELS.items():
        if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
            print(f"✓ Already present: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
        else:
            print(f"Downloading {path} from {url} ...")
            urllib.request.urlretrieve(url, path)
            print(f"✓ Downloaded: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
    print("\nAll model weights ready for road damage detection!")

if __name__ == "__main__":
    main()
