# Project Defense & Viva Notes
## Road Damage Detection using YOLOv8 (PBL)
**Student:** Ravish Patel (`ravishp911@gmail.com`)  
**Repo:** [https://github.com/Ravish945/road-damage-classification](https://github.com/Ravish945/road-damage-classification)

---

### 1. Key Numbers to Remember Off the Top of My Head
* **Total images:** 7,706 images from India (crowdsourced dashcam).
* **Positive vs Negative:** 3,785 have damage annotations; 3,921 are clean road backgrounds (used as negative samples to reduce false positives).
* **Classes:**
  - `D00`: Longitudinal cracks (wheel paths, joint fatigue) — 1,734 instances
  - `D10`: Transverse cracks (thermal contraction across lane) — 113 instances (severe class imbalance)
  - `D20`: Alligator cracks (interconnected fatigue cracks, base failure) — 2,021 instances
  - `D40`: Potholes / rutting (surface depressions) — 3,187 instances (most common)
* **Final Model Results (50 epochs on val set):**
  - Overall mAP@0.50: **51.5%**
  - Potholes (`D40`): **64.8% mAP** (highest, model recognizes pothole contours well)
  - Alligator (`D20`): **55.1% mAP**
  - Longitudinal (`D00`): **51.8% mAP**
  - Transverse (`D10`): **34.2% mAP** (lowest due to sample scarcity)
* **Speed / Latency:**
  - Apple M1 CPU: ~26 ms per frame at 320x320 (around 38 FPS).
  - Tesla T4 GPU on Colab: ~9.4 ms per frame at 640x640 (around 106 FPS).
  - At 60 km/h, 38 FPS takes a frame every 43.5 cm, so a typical 50 cm pothole is captured 2 to 3 times.

---

### 2. Real Issues I Ran Into & How I Fixed Them

#### Problem 1: Transverse Cracks (D10) had terrible initial accuracy (~19% mAP)
* **What went wrong:** The dataset has 3,187 potholes but only 113 transverse cracks. The detector barely learned transverse features because gradients from potholes dominated every batch.
* **How I handled it:** In `voc_to_yolo.py`, I grouped sub-class `D11` into `D10`. During training, I applied horizontal flip and perspective augmentations. This brought `D10` mAP up from 19.4% to 34.2%.

#### Problem 2: Data Leakage from Sequential Dashcam Video
* **What went wrong:** In RDD India, consecutive images (`India_000001.jpg`, `India_000002.jpg`) are extracted from continuous video. Shuffling randomly meant adjacent video frames ended up in both train and validation splits, giving falsely high validation accuracy.
* **How I handled it:** Added chunk-based splitting (`--split_strategy chunk`) in `voc_to_yolo.py`. It keeps 50 consecutive frames together in blocks before splitting, so whole road stretches remain either in train or validation.

#### Problem 3: Invalid Bounding Boxes in Raw Annotations
* **What went wrong:** A few raw XML files had erroneous coordinates where `xmax <= xmin` or coordinates exceeded image boundaries (e.g. 721 on a 720x720 image).
* **How I handled it:** Added validation and clamping in `voc_to_yolo.py` (`max(0.0, min(1.0, coord))`) to drop inverted boxes and keep normalized values within `[0, 1]`.

#### Problem 4: Model Fallback in the Web Dashboard
* **What went wrong:** Initially, if `models/best.pt` was missing, the loader fell back to standard `yolov8n.pt`. That was wrong because standard YOLOv8n is trained on COCO (cars, people, dogs) and would give completely misleading results.
* **How I handled it:** Removed the fallback in `app.py` and `predict.py` to raise a `FileNotFoundError` and show a clear error message in the Streamlit UI until road weights are provided.

---

### 3. Common Viva Questions & Practical Answers

#### "Did you use AI to build this?"
> *"I used AI tools to help write boilerplate code and troubleshoot syntax errors, but the data science decisions were mine. I found and fixed the temporal data leakage in the sequential video frames, handled the 46:1 class imbalance for transverse cracks, benchmarked the inference speed on my own Mac CPU vs Colab GPU, and built the evaluation and CSV logging logic."*

#### "Why YOLOv8 instead of a standard CNN like ResNet or VGG?"
> *"A classification CNN only outputs a single label for the entire image (e.g., 'image has pothole'). It cannot draw bounding boxes or count how many potholes exist. YOLOv8 is an object detector: it predicts both class probabilities and bounding box coordinates in one forward pass, which makes it fast enough to run in real time on a moving vehicle."*

#### "Why did you keep the images with no defects?"
> *"If you train an object detector only on images with road damage, the model assumes every single frame must contain damage. In the real world, most road surfaces are clean. Retaining the 3,921 undamaged images with empty label files acts as negative samples, teaching the network what clean asphalt looks like and preventing false alarms on road shadows and manholes."*

#### "What is the difference between IoU and mAP?"
> *"IoU (Intersection over Union) measures the geometric overlap between the predicted bounding box and the true bounding box: $\text{Area of Overlap} / \text{Area of Union}$. If IoU is $\ge 0.5$, we count it as a correct localization.  
> mAP@0.50 (Mean Average Precision) measures overall detection performance: it calculates the area under the Precision-Recall curve for each class at 50% IoU and takes the average across all 4 classes."*

#### "How would this be used in practice?"
> *"A wide-angle camera mounted behind the rearview mirror or windshield of a municipal bus or garbage truck captures road video. Frames are processed onboard by a lightweight edge device (like a Jetson Nano or Raspberry Pi 5). When potholes or alligator cracks are detected, the system appends the defect count and triage severity to a CSV audit log (`results/inspection_log.csv`), which can be sent to the municipal road maintenance portal."*
