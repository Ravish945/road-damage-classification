# Codebase Line-by-Line Explanation Guide
## Road Damage Detection & Classification (PBL)
**Author:** Ravish Patel  
**Purpose:** Prepare for teacher, examiner, or jury code walkthroughs.

---

This document explains every file, function, and critical line of code in the repository so you can explain any line your professor points to.

---

## 1. `voc_to_yolo.py` (Dataset Preprocessing Script)

### Purpose:
Converts raw Pascal VOC XML bounding box annotations into normalized YOLO TXT labels and splits the 7,706 images into 80% train and 20% validation.

### Key Lines Explained:
```python
CLASS_MAPPING = {
    'D00': 0,
    'D01': 0,   # Grouped with longitudinal crack
    'D10': 1,
    'D11': 1,   # Grouped with transverse crack
    'D20': 2,
    'D40': 3,
}
```
* **Why:** In Indian road conditions, `D01` is a construction joint longitudinal crack, essentially the same structural distress as `D00`. `D11` is similarly a transverse construction crack. Grouping them improves sample support for underrepresented classes without confusing the detector.

```python
def convert_voc_bbox_to_yolo(size, box):
    dw = 1.0 / size[0]   # 1 / image_width
    dh = 1.0 / size[1]   # 1 / image_height

    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
```
* **Teacher Question:** *"Why are you dividing by width and height?"*
* **Answer:** *"Pascal VOC stores absolute pixel coordinates `[xmin, xmax, ymin, ymax]`. YOLO requires normalized coordinates relative to the image size `[0.0 to 1.0]`. If we train on 640×640 resolution or resize images during data augmentation, normalized coordinates remain scale-invariant."*

```python
x = max(0.0, min(1.0, x))
```
* **Why:** Clamping ensures no bounding box coordinate accidentally exceeds 1.0 or drops below 0.0 due to crowdsourced annotators drawing slightly outside image borders.

```python
# We keep samples with valid damages, plus a balanced ratio of background images
valid_samples.append((img_path, base_name, boxes))
```
* **Teacher Question:** *"What happens if an image has no `<object>` in the XML? Why not delete it?"*
* **Answer:** *"Images without objects are negative/background samples. In the script, we write an empty `.txt` file for them. This teaches YOLO what clean asphalt looks like, preventing false alarms on road shadows, zebra crossings, and manholes."*

---

## 2. `train.py` (Model Training Script)

### Purpose:
Loads pre-trained YOLOv8 weights and trains them on Indian road distresses using transfer learning.

### Key Lines Explained:
```python
def get_optimal_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
```
* **Why:** Makes the code platform-independent. Automatically runs on NVIDIA GPU on Google Colab (`cuda`), Apple Silicon GPU on macOS (`mps`), or CPU fallback.

```python
model = YOLO(model_size)  # Loads 'yolov8n.pt'
```
* **Teacher Question:** *"Why are you using pre-trained weights instead of training from scratch?"*
* **Answer:** *"Training from scratch requires millions of images to learn basic edge and texture filters. Transfer learning from COCO pre-trained weights gives us pre-trained convolution kernels, allowing faster convergence in just 25–50 epochs with higher accuracy."*

```python
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    patience=10,        # Early stopping
    plots=True          # Generate confusion matrix & PR curves
)
```
* **`patience=10`:** Early stopping parameter. If validation mAP does not improve for 10 consecutive epochs, training automatically halts to prevent overfitting and save compute time.
* **`plots=True`:** Instructs Ultralytics to automatically compute and render the confusion matrix, F1-confidence curve, Precision-Recall curve, and training loss plots.

---

## 3. `evaluate.py` (Model Evaluation Script)

### Purpose:
Calculates quantitative metrics on the held-out validation set (1,545 images).

### Key Lines Explained:
```python
metrics = model.val(
    data=data_yaml,
    imgsz=imgsz,
    conf=0.25,
    iou=0.6,
    device=device
)
```
* **`conf=0.25`:** Confidence threshold. Any predicted box with less than 25% model certainty is ignored.
* **`iou=0.6`:** Intersection over Union threshold for Non-Maximum Suppression (NMS). Overlapping boxes for the same distress with IoU > 0.6 are merged.

```python
print(f"mAP@0.50       : {metrics.box.map50:.4f}")
print(f"mAP@0.50-0.95  : {metrics.box.map:.4f}")
```
* **Teacher Question:** *"What is the difference between mAP@0.50 and mAP@0.50-0.95?"*
* **Answer:** *"mAP@0.50 measures Average Precision when a predicted box has at least 50% overlap with ground truth. mAP@0.50:0.95 is the COCO standard metric that averages mAP across 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05, strictly testing localization precision."*

---

## 4. `predict.py` (Inference & Severity Rating Script)

### Purpose:
Runs detection on input images or video streams and computes a municipal severity rating.

### Key Lines Explained:
```python
def assess_damage_severity(detection_counts):
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
```
* **Teacher Question:** *"Why did you implement this severity function?"*
* **Answer:** *"Road inspection teams do not just need bounding boxes; they need automated triage. A single pothole can burst a tire or cause a two-wheeler accident, so any frame with `D40` is immediately classified as HIGH severity for emergency patching within 24–48 hours."*

---

## 5. `app.py` (Interactive Streamlit Web Dashboard)

### Purpose:
Provides a web GUI for live visual demonstrations to instructors or examiners.

### Key Lines Explained:
```python
@st.cache_resource
def load_model(weights_path):
    from ultralytics import YOLO
    return YOLO(weights_path)
```
* **Why:** `@st.cache_resource` ensures the PyTorch model weights are loaded into memory once and reused across user clicks, avoiding reloading the 6.4 MB model on every user interaction.

```python
annotated_array = results.plot()
annotated_img = Image.fromarray(annotated_array[..., ::-1])
```
* **Why `[::-1]`?** OpenCV stores images in BGR format (Blue-Green-Red), while PIL and Streamlit expect RGB (Red-Green-Blue). `[::-1]` reverses the 3rd array dimension from BGR to RGB so colors render correctly.
