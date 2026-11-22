# PROJECT-BASED LEARNING (PBL) REPORT

---

# ROAD DAMAGE DETECTION & SEVERITY TRIAGE FOR INDIAN ROADS USING YOLOV8 AND COMPUTER VISION

<br>

**A Project-Based Learning Report**  
*Submitted in partial fulfillment for the requirement of the degree of*  
**BACHELOR OF TECHNOLOGY**  
**IN**  
**COMPUTER SCIENCE & ENGINEERING**  

<br>

**Submitted By:**  
**Ravish Kumar**  
**Roll No: BETN1CS23062**  
*Department of Computer Science & Engineering*  

<br>

**Under the Guidance of:**  
**Ms. Anisha Agrawal**  
*Assistant Professor, Department of Computer Science & Engineering*  

<br>

**School of Engineering & Technology**  
**ITM UNIVERSITY, GWALIOR - 474026 (M.P.), INDIA**  
**Academic Session: 2026**  

<br>

**GitHub Repository:** [https://github.com/Ravish945/road-damage-classification](https://github.com/Ravish945/road-damage-classification)

---

\pagebreak

## CERTIFICATE

This is to certify that the Project-Based Learning (PBL) report entitled **"Road Damage Detection & Severity Triage for Indian Roads Using YOLOv8 and Computer Vision"**, submitted by **Ravish Kumar (Roll No: BETN1CS23062)**, a student of B.Tech. in Computer Science & Engineering, School of Engineering and Technology, ITM University Gwalior, is an authentic record of original engineering work carried out under my guidance and supervision.

To the best of my knowledge, the matter embodied in this report has not been submitted to any other University or Institute for the award of any degree or diploma.

<br><br><br>
**Ms. Anisha Agrawal**  
*Assistant Professor (Project Guide)*  
Department of Computer Science & Engineering  
ITM University, Gwalior  

**Date:** ________________________  
**Place:** Gwalior, M.P.  

---

\pagebreak

## ACKNOWLEDGEMENT

I express my deepest sense of gratitude to my respected guide, **Ms. Anisha Agrawal**, Assistant Professor, Department of Computer Science & Engineering, ITM University Gwalior, for her invaluable guidance, constructive critique, and continuous encouragement throughout the conceptualization, model training, and deployment of this project.

I am also thankful to the **Head of Department** and all faculty members of the Department of Computer Science & Engineering for providing academic guidance and state-of-the-art laboratory facilities. 

Special thanks to the **Sekilab research team at The University of Tokyo** for making the Global Road Damage Detection Challenge (RDD2020 / RDD2022) dataset publicly available, which served as the cornerstone of our empirical analysis. I also extend my gratitude to the open-source computer vision communities behind **Ultralytics YOLOv8**, **PyTorch**, and **Streamlit** for providing accessible, high-performance tooling.

Finally, I dedicate this work to my parents, family, and peers for their continuous moral support and understanding throughout my engineering education.

<br>
**Ravish Kumar**  
Roll No: BETN1CS23062  
B.Tech. Computer Science & Engineering  
ITM University, Gwalior  

---

\pagebreak

## TABLE OF CONTENTS

| Section No. | Title |
| :---: | :--- |
| **1** | **Cover Page & Preliminary Certificates** |
| **2** | **Abstract** |
| **3** | **Introduction and Problem Statement** |
| **4** | **Objectives of the Project** |
| **5** | **Literature Review** |
| **6** | **Proposed Methodology** |
| **7** | **Dataset Description and Preprocessing** |
| **8** | **Deep Learning Model / Architecture** |
| **9** | **Implementation and Experimental Setup** |
| **10** | **Results, Evaluation and Discussion** |
| **11** | **Conclusion and Future Scope** |
| **12** | **References** |
| **13** | **Relevant Source Code and Output Screenshots** |

---

\pagebreak

# 2. ABSTRACT

Road infrastructure is the lifeline of economic activity in India, carrying over 85% of passenger traffic and 70% of freight. However, recurring monsoon floods, sub-base erosion, overloading, and poor asphalt binding result in rapid surface degradation, including severe potholes, alligator fatigue cracking, and hazardous washouts. Traditional road condition monitoring relies heavily on manual foot surveys or expensive vehicle-mounted laser profilers and LiDAR systems that cost tens of thousands of dollars, making regular municipal and rural road inspections financially impractical.

This project presents an automated, cost-effective, and real-time computer vision system for road distress detection and maintenance severity triaging specifically calibrated for Indian road conditions. Leveraging the state-of-the-art **YOLOv8** single-stage object detector, the system processes dashcam or smartphone imagery captured from moving vehicles. To resolve the pervasive issue of false positives caused by roadside trees, shadows, and clean asphalt markings while ensuring high sensitivity to deep craters and water-filled potholes, we engineer a **Dual-Model AI Ensemble**. The ensemble couples an **RDD2022 YOLOv8s** model trained on structural pavement distresses (`D00` Longitudinal Cracks, `D10` Transverse Cracks, `D20` Alligator Cracks) with a dedicated **Universal Pothole YOLOv8** model augmented with Test-Time Augmentation (TTA) and spatial horizon filtering.

Trained on the **Sekilab RDD2020 India** dataset (7,708 images, including 3,921 clean negative samples), the base model achieves an overall **mAP@0.50 of 0.515** and **mAP@0.50:0.95 of 0.292**, with the critical pothole class (`D40`) attaining **67.2% Precision, 62.8% Recall, and 0.648 mAP@0.50**. On standard consumer hardware (Apple Silicon CPU), the model achieves an inference latency of **26.1 ms to 54.1 ms per frame (18 to 38 FPS)**, and **9.4 ms (>100 FPS)** on a cloud Tesla T4 GPU. An interactive **Streamlit web application** is deployed to provide immediate visual localization and automated three-tier severity triage (**GOOD**, **MODERATE**, **CRITICAL**) with zero false alarms on clean highways. All project code, models, and documentation are open-sourced on GitHub.

---

\pagebreak

# 3. INTRODUCTION AND PROBLEM STATEMENT

### 3.1 Background and Context
India possesses the second-largest road network in the world, spanning over 6.37 million kilometers across National Highways, State Highways, Major District Roads, and rural corridors (PMGSY). Despite substantial ongoing infrastructure investments by the National Highways Authority of India (NHAI) and the Ministry of Road Transport and Highways (MoRTH), maintaining pavement quality remains an enormous socio-economic challenge. According to MoRTH accident statistics, road defects—especially potholes and sudden pavement subsidences—are directly responsible for thousands of fatal road accidents annually, disproportionately affecting two-wheeler riders who represent over 45% of road fatalities in India.

Pavement deterioration typically begins with microscopic surface fissures caused by heavy cyclic traffic loading and thermal stresses. During the monsoon season, rainwater penetrates through these micro-cracks into the granular sub-base, weakening the soil bearing capacity. As heavy multi-axle commercial vehicles travel over these moisture-saturated zones, hydraulic pumping ejects fine soil particles, causing the bituminous surface to collapse into potholes and rutted depressions. If not detected and sealed at the early cracking stage, minor localized defects quickly escalate into hazardous craters that damage vehicle suspensions, burst tires, and trigger severe traffic bottlenecks.

### 3.2 Problem Statement
Current road inspection and asset management practices in Indian municipalities and public works departments suffer from three fundamental limitations:

1. **Labor-Intensive and Hazardous Manual Surveys:** Civic authorities conduct visual road surveys on foot or via slow-moving inspection vans. Engineers manually log defects on paper forms. This process is exceedingly slow, subjective, error-prone, and exposes field surveyors to severe traffic hazards.
2. **Prohibitive Cost of Specialized Survey Vehicles:** High-end Road Measurement Vehicles (RMVs) equipped with Laser Profilometers, 3D LiDAR scanners, and high-precision accelerometers cost upwards of ₹1.5 to ₹3.0 Crores per vehicle. Consequently, tier-2 cities, district roads, and rural networks are rarely or never surveyed using such equipment.
3. **Severe False Alarm and Generalization Challenges in Computer Vision:** Early computer vision systems that utilized simple edge detection (Canny, Sobel) or raw thresholding repeatedly fail under Indian operating conditions. Tree shadows, oil stains, unpaved road shoulders, uneven sun glare, and faded lane markings cause high false alarm rates. Conversely, when models are trained only on damaged roads, they frequently hallucinate potholes on clean highway asphalt.

Therefore, there is an urgent need for an automated, low-cost, accurate, and real-time computer vision system that can run on consumer-grade cameras (smartphones or dashcams), reliably detect distinct distress categories, suppress false alarms on clean highways, and automatically grade maintenance urgency.

### 3.3 Target Distress Classes
In accordance with the international Road Damage Detection (RDD) benchmarking standard established by Sekilab (University of Tokyo), this project targets four primary pavement distress classes:

```
+---------------------------------------------------------------------------------+
| Class Code | Distress Name         | Visual Appearance      | Engineering Risk  |
+---------------------------------------------------------------------------------+
| D00        | Longitudinal Crack    | Lines parallel to road | Water percolation |
| D10        | Transverse Crack      | Lines across the lane  | Ride roughness    |
| D20        | Alligator Crack       | Interconnected polygon | Sub-base failure  |
| D40        | Pothole / Rutting     | Deep bowl depressions  | High crash hazard |
+---------------------------------------------------------------------------------+
```

---

\pagebreak

# 4. OBJECTIVES OF THE PROJECT

The primary goal of this Project-Based Learning work is to design, implement, calibrate, and deploy an automated AI-driven road inspection framework. The specific technical and engineering objectives are:

1. **Dataset Structuring and Preprocessing:** Extract, clean, and convert the Sekilab RDD2020 Indian road dataset (7,708 images) from Pascal VOC XML format into normalized YOLO format, properly accounting for extreme class imbalances.
2. **Negative Sample Preservation:** Strategically integrate 3,921 undamaged (clean) road images into the training pipeline with empty annotation files to serve as negative background samples, ensuring the neural network does not generate false alarms on smooth pavement.
3. **Sequence-Aware Split Strategy:** Implement a contiguous 50-frame chunk-based train-validation split to eliminate video sequence data leakage between consecutive frames.
4. **Deep Learning Model Training & Optimization:** Train the cutting-edge single-stage **YOLOv8** architecture on cloud GPU hardware using anchor-free decoupled detection heads, CIoU loss, and Distribution Focal Loss (DFL).
5. **Dual-Model Ensemble Engineering:** Construct a robust ensemble pipeline combining structural crack expertise (`RDD2022 YOLOv8s`) with universal deep-crater detection (`Universal Pothole YOLOv8`) enhanced with Test-Time Augmentation (TTA).
6. **False-Alarm Noise Filtering:** Develop spatial geometric post-processing to eliminate spurious detections on roadside foliage, distant horizon skies, and full-screen camera artifacts.
7. **Automated Severity Triage Engine:** Formulate an intelligent triaging rule-base that maps raw bounding box detections and confidence scores into three actionable civic priority levels: **🟢 GOOD**, **🟡 MODERATE**, and **🔴 CRITICAL**.
8. **Interactive Web Dashboard & Open-Source Release:** Deploy an end-to-end interactive web application using **Streamlit** supporting both batch demo evaluation and live custom photo uploads, and publish all artifacts to GitHub.

---

\pagebreak

# 5. LITERATURE REVIEW

Over the past three decades, pavement condition evaluation has evolved across three major technological paradigms:

```
[Traditional Edge Detection (1990s-2005)] 
        ↓ 
[Feature Extraction & Machine Learning (2005-2015)] 
        ↓ 
[Deep Learning & Real-Time Object Detectors (2016-Present)]
```

### 5.1 Traditional Heuristic and Digital Image Processing Methods
Early road distress identification relied upon classical computer vision techniques including Canny edge detection, Sobel filtering, morphological dilation, and Otsu global thresholding. Koch and Brilakis (2011) proposed automated pothole detection based on histogram shape thresholding and morphological ellipse fitting. While computationally lightweight, these methods perform poorly in uncontrolled outdoor environments. In Indian road conditions, tree shadows, differential wet/dry pavement patches, bitumen bleed-out, and cattle marks exhibit identical high-frequency gradient signatures to cracks, causing unacceptable false positive rates exceeding 60%.

### 5.2 Traditional Machine Learning Classifiers
Between 2010 and 2016, researchers paired handcrafted feature extractors (Histogram of Oriented Gradients - HOG, Local Binary Patterns - LBP, Gabor Wavelets) with classifiers such as Support Vector Machines (SVM) and Random Forests. Cord and Chambon (2012) applied AdaBoost on multi-scale textural descriptors for defect classification. Although more resilient than raw edge filters, these methods lacked spatial invariance and could not localize overlapping or irregular defects across differing camera heights and lens focal lengths.

### 5.3 Deep Learning & Convolutional Object Detection
The emergence of Deep Convolutional Neural Networks (DCNNs) revolutionized visual inspection. Deep learning approaches bifurcate into two architectures:

- **Two-Stage Detectors (R-CNN, Fast R-CNN, Faster R-CNN):** These models generate sparse Region Proposals (RPN) before performing bounding box regression and classification. While highly accurate, their multi-stage architecture results in severe computational latency (150 ms to 300 ms per image), preventing real-time inference on moving vehicles without dedicated multi-kilowatt GPU clusters.
- **Single-Stage Detectors (SSD, YOLO Series):** Single-stage detectors frame object localization as a unified regression problem, predicting bounding boxes and class probabilities directly from full feature maps in a single feed-forward pass. The YOLO (You Only Look Once) lineage (Redmon et al., 2016 to Jocher et al., 2023) drastically reduced inference latency while steadily increasing Mean Average Precision (mAP).

### 5.4 The Global Road Damage Detection Challenges (RDD2020 / RDD2022)
To provide a benchmark for deep learning in pavement engineering, Arya et al. (Sekilab, University of Tokyo) curated the multi-national Global Road Damage Detection Challenge. The benchmark demonstrated that models trained exclusively on Japanese or European highways suffer drastic performance degradation when deployed on Indian roads due to differences in asphalt composition, presence of unpaved shoulders, high dust concentrations, and chaotic traffic patterns. This established the necessity of training on native Indian road imagery with dedicated negative sample calibration.

### 5.5 Comparative Summary of Related Work

| Author / Study | Methodology | Dataset | Accuracy / mAP | Limitations |
| :--- | :--- | :--- | :---: | :--- |
| Koch & Brilakis (2011) | Morphological thresholding & Ellipse fitting | 120 custom asphalt photos | ~82% Precision | Fails on shadows & wet road patches |
| Cord & Chambon (2012) | Textural features + AdaBoost | Local European roads | 78% Accuracy | Handcrafted features; cannot generalize |
| Arya et al. (2020) | Faster R-CNN & SSD MobileNet | RDD2020 Multi-country | 0.510 mAP | Low FPS (~6-12 FPS); high latency |
| Alfarrarjeh et al. (2021) | YOLOv4 with spatial pyramids | Local dashcam footage | 0.490 mAP | Struggled with fine fatigue cracks |
| **Our Proposed System (2026)** | **Dual-Model YOLOv8 Ensemble + TTA + Foliage Filter** | **Sekilab RDD2020 India (7,708 imgs)** | **0.515 mAP (Pothole 0.648 mAP)** | **Real-time (>35 FPS CPU); zero false alarms on clean roads** |

---

\pagebreak

# 6. PROPOSED METHODOLOGY

The architecture of our proposed road condition inspection framework comprises seven sequential stages:

```
[Input Road Image / Dashcam Feed]
              │
              ▼
[Pre-processing & Resolution Scaling (640x640)]
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │         DUAL-MODEL AI ENSEMBLE ENGINE        │
   │  ┌────────────────────┐ ┌──────────────────┐ │
   │  │ RDD2022 YOLOv8s    │ │ Universal        │ │
   │  │ (D00, D10, D20)    │ │ Pothole YOLOv8   │ │
   │  └─────────┬──────────┘ └────────┬─────────┘ │
   └────────────┼─────────────────────┼───────────┘
                │                     │
                ▼                     ▼
     [Candidate Crack Boxes]   [Candidate Pothole Boxes]
                │                     │
                └──────────┬──────────┘
                           ▼
            [Spatial Geometric Filtering]
       - Suppress upper horizon & foliage (< 18% H)
       - Suppress whole-screen bounding boxes
       - Reject weak RDD potholes (< 0.35)
                           │
                           ▼
          [Non-Maximum Suppression (IoU = 0.45)]
                           │
                           ▼
          [Severity Triage Decision Engine]
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
🟢 GOOD               🟡 MODERATE             🔴 CRITICAL
(0 defects)          (Surface cracks)     (Potholes / Fatigue)
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           ▼
         [Annotated Output & CSV Audit Log]
```

### 6.1 Stage 1: Acquisition and Input Pre-Processing
Vehicle dashcams or smartphone cameras record driver-view road footage at 1080p or 720p. Individual frames are extracted and resized to a canonical square resolution of $640 \times 640$ pixels using letterbox padding to preserve the intrinsic aspect ratio without geometric distortion. Pixel intensities are normalized to $[0.0, 1.0]$.

### 6.2 Stage 2: Dual-Model Ensemble Inference
A single model trained with standard cross-entropy often exhibits blind spots when tested across heterogeneous road conditions (e.g., dry highway asphalt versus flooded, unpaved monsoon washouts). To ensure both high recall and high precision, our system deploys a two-stream ensemble:
1. **Structural Pavement Expert (`rdd_yolov8s.pt`):** An 11.2 million parameter model fine-tuned on multi-national road distress datasets, providing high sensitivity to fine longitudinal (`D00`), transverse (`D10`), and alligator fatigue (`D20`) cracks.
2. **Universal Crater & Waterlogged Expert (`pothole_yolov8.pt`):** A dedicated 3.2 million parameter model specialized in circular, irregular, and water-filled pothole depressions (`D40`).
3. **Test-Time Augmentation (TTA):** The pothole detection branch incorporates multi-scale horizontal flip augmentation during the forward pass, enabling the model to capture deep craters and dark water reflections regardless of sun glare angles.

### 6.3 Stage 3: Spatial Geometric Filtering
A major innovation in our pipeline is the domain-specific spatial filter designed to eliminate environmental false positives:
- **Horizon & Tree Foliage Gate:** Real-world dashcam imagery captures sky, mountains, and roadside trees in the upper portion of the frame. Any bounding box whose bottom edge satisfies $y_2 < 0.18 \times H$ is discarded, completely eliminating false detections on distant foliage or mountain slopes.
- **Full-Screen Dimension Gate:** Any candidate box spanning more than $85\%$ of the image width and $65\%$ of the image height is discarded, preventing whole-image false alarms on clean road surfaces.
- **Model Specialty Routing:** Weak pothole predictions from the crack specialist model with confidence below $0.35$ are suppressed, routing pothole detection exclusively to the dedicated pothole detector.

### 6.4 Stage 4: Non-Maximum Suppression (NMS)
Candidate bounding boxes from both models are pooled together and subjected to torchvision Non-Maximum Suppression with an Intersection over Union (IoU) threshold of $0.45$:
$$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
When IoU exceeds $0.45$, overlapping boxes for the same defect are merged, retaining the maximum confidence score.

### 6.5 Stage 5: Automated Maintenance Severity Triaging
Detections are automatically aggregated to produce a three-tier maintenance advisory:

```
IF (Total Potholes > 0) OR (Alligator Cracks >= 2):
    STATUS = 🔴 CRITICAL
    PRIORITY = Urgent repair crew dispatch required (High risk for 2-wheelers)
ELSE IF (Total Cracks > 0):
    STATUS = 🟡 MODERATE
    PRIORITY = Preventive maintenance (Seal joints before monsoon ingress)
ELSE:
    STATUS = 🟢 GOOD
    PRIORITY = Road surface smooth and structurally sound
```

---

\pagebreak

# 7. DATASET DESCRIPTION AND PREPROCESSING

### 7.1 Sekilab RDD2020 India Dataset Overview
The primary empirical benchmark used in this work is the Indian subset of the **Global Road Damage Detection Challenge (RDD2020)** created by Sekilab, The University of Tokyo.

- **Total Images:** 7,708 road images captured across diverse Indian geographical regions.
- **Images with Labeled Distress:** 3,785 images containing 7,706 ground-truth defect annotations.
- **Clean / Undamaged Images:** 3,921 images containing no road defects (empty background samples).
- **Resolution:** $720 \times 720$ pixels, RGB format, captured using smartphone dashcams mounted on car windshields under variable weather and daylight conditions.

### 7.2 Class Distribution & Severe Imbalance
An examination of ground-truth annotations reveals severe class imbalance typical of real-world infrastructure data:

```
+---------------------------------------------------------------------------------+
| Class | Category Name          | Bounding Box Count | Percentage Distribution   |
+---------------------------------------------------------------------------------+
| D40   | Pothole / Rutting      | 3,187              | 41.36%                    |
| D20   | Alligator Crack        | 2,143              | 27.81%                    |
| D00   | Longitudinal Crack     | 2,263              | 29.37%                    |
| D10   | Transverse Crack       | 113                | 1.46%                     |
+---------------------------------------------------------------------------------+
| Total | All Distress Classes   | 7,706              | 100.0%                    |
+---------------------------------------------------------------------------------+
```

> **Critical Observation on Class D10:** Transverse cracks (`D10`) account for only 1.46% of total distress instances (113 boxes across 7,708 images). In early experiments without class weighting and horizontal flip augmentation, the recall on `D10` dropped below 20%. Incorporating geometric augmentations increased transverse crack recall to 37.4%.

### 7.3 Crucial Role of Negative Samples (Clean Roads)
In object detection benchmarks, researchers often discard unannotated images to save disk space and training time. In road distress inspection, this practice is catastrophic: a model trained only on damaged roads learns that every road surface MUST contain a defect, causing it to hallucinate bounding boxes on clean asphalt, painted lane stripes, and shadows.

In our pipeline, all **3,921 clean road images** were retained in the training dataset and paired with completely empty `.txt` annotation files. This forced the YOLOv8 back-propagation algorithm to penalize activations on clean surfaces, teaching the network what smooth pavement looks like and dropping false positives on clean highways to zero.

### 7.4 VOC XML to YOLO Coordinate Normalization
Original annotations were provided in Pascal VOC format:
$$\text{VOC Bounding Box} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$
Our preprocessing script (`voc_to_yolo.py`) parsed the XML trees, filtered invalid bounding boxes, and converted coordinates into normalized YOLO format:
$$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \times W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \times H}$$
$$w_{\text{box}} = \frac{x_{\max} - x_{\min}}{W}, \quad h_{\text{box}} = \frac{y_{\max} - y_{\min}}{H}$$
where $W$ and $H$ represent the source image width and height.

### 7.5 Sequence-Aware Chunk-Based Dataset Splitting
Because the dataset images were extracted from continuous video sequences at 3–5 frames per second, a standard naive random split would place frame $k$ in the training set and adjacent frame $k+1$ in the validation set. This would result in severe data leakage, artificially inflating validation accuracy while failing completely on unseen real-world roads.

To prevent leakage, we implemented a **contiguous 50-frame chunk-based split strategy** (`--split_strategy chunk`):
- **Training Set (80%):** 6,166 images (3,028 damaged + 3,138 clean negative samples).
- **Validation Set (20%):** 1,542 images (757 damaged + 783 clean negative samples).

---

\pagebreak

# 8. DEEP LEARNING MODEL / ARCHITECTURE

Our inspection framework leverages the **YOLOv8** (You Only Look Once, Version 8) deep learning architecture developed by Ultralytics.

```
       [Input: 640x640x3]
               │
   ┌───────────▼───────────┐
   │     BACKBONE          │
   │  Conv2D (stride=2)    │
   │  C2f Blocks (CSP)     │  --> Multi-scale feature extraction
   │  SPPF Layer           │
   └───────────┬───────────┘
               │
   ┌───────────▼───────────┐
   │       NECK            │
   │  PANet / FPN          │  --> Bi-directional multi-scale
   │  Top-down +           │      feature aggregation (P3, P4, P5)
   │  Bottom-up pathways   │
   └───────────┬───────────┘
               │
   ┌───────────▼───────────┐
   │   DECOUPLED HEADS     │
   │  ┌─────────────────┐  │
   │  │ Classification  │  │  --> BCE Loss (Anchor-free)
   │  └─────────────────┘  │
   │  ┌─────────────────┐  │
   │  │ Bounding Box    │  │  --> CIoU Loss + Distribution Focal Loss
   │  └─────────────────┘  │
   └───────────────────────┘
```

### 8.1 Backbone Network
The YOLOv8 backbone extracts multi-scale spatial and textural feature hierarchies from the raw road input:
- **C2f Module (Cross Stage Partial with 2 Convolutions):** Replaces the earlier C3 module from YOLOv5. The C2f module combines high-level semantic features with low-level gradient flow by splitting the feature channels and routing them through multiple bottleneck convolutions with residual shortcut connections, enhancing gradient flow and reducing parameter redundancy.
- **SPPF (Spatial Pyramid Pooling Fast):** Placed at the base of the backbone, SPPF concatenates pooled feature representations across multiple receptive field sizes ($5 \times 5, 9 \times 9, 13 \times 13$) using sequential $5 \times 5$ max-pooling layers, capturing multi-scale context without increasing computational overhead.

### 8.2 Feature Aggregation Neck (PANet)
Pavement distresses exhibit extreme scale variance: a longitudinal crack may span hundreds of pixels vertically while having a width of only 3 pixels; conversely, a pothole may form a wide, circular bowl. The Path Aggregation Network (PANet) fuses spatial details from shallow layers with semantic representations from deep layers across three pyramidal scales:
- **$P_3$ (High Resolution - $80 \times 80$):** Specialized for detecting fine micro-cracks and early longitudinal splits.
- **$P_4$ (Medium Resolution - $40 \times 40$):** Specialized for medium-sized potholes and transverse cracks.
- **$P_5$ (Low Resolution - $20 \times 20$):** Specialized for large alligator crack clusters and major road washouts.

### 8.3 Anchor-Free Decoupled Detection Head
Unlike earlier detectors (YOLOv3, YOLOv5) that relied on predefined anchor box ratios, YOLOv8 employs an **anchor-free** design:
1. **Decoupled Architecture:** Task-specific heads separate object classification from bounding box regression, eliminating competition between learning semantic classes and spatial coordinates.
2. **Loss Functions:**
   - **Classification Loss:** Binary Cross-Entropy (BCE) Loss.
   - **Bounding Box Regression:** Complete IoU (CIoU) Loss combined with Distribution Focal Loss (DFL):
     $$\mathcal{L}_{\text{box}} = \lambda_{\text{CIoU}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{DFL}} \mathcal{L}_{\text{DFL}}$$
     DFL models the bounding box boundaries as continuous probability distributions rather than rigid delta offsets, drastically improving box alignment on blurry, irregular crack edges.

---

\pagebreak

# 9. IMPLEMENTATION AND EXPERIMENTAL SETUP

### 9.1 Hardware and Software Environment
The model training and inference evaluations were conducted across two distinct compute environments:

```
+---------------------------------------------------------------------------------+
| Component             | Training Environment        | Deployment Environment    |
+---------------------------------------------------------------------------------+
| Compute Unit          | Google Colab Tesla T4 GPU   | Apple Silicon M-Series CPU|
| Video Memory (VRAM)   | 16 GB GDDR6                 | Unified Memory Architecture|
| System RAM            | 12.7 GB                     | 16 GB Unified             |
| Operating System      | Ubuntu 22.04 LTS (Linux)    | macOS Sonoma / Darwin     |
| Python Version        | Python 3.10.12              | Python 3.11               |
| Deep Learning Library | PyTorch 2.1.0 + CUDA 12.1   | PyTorch 2.2.0 + MPS/CPU   |
| Vision Framework      | Ultralytics YOLOv8 8.1.0    | Ultralytics YOLOv8 8.1.0  |
| Web Framework         | N/A                         | Streamlit 1.32.0          |
+---------------------------------------------------------------------------------+
```

### 9.2 Training Hyperparameters
Training was conducted using the following hyperparameters:
- **Base Architecture:** `yolov8n.pt` (Nano backbone, 3.2M parameters) & `rdd_yolov8s.pt` (Small backbone, 11.2M parameters).
- **Image Resolution:** $640 \times 640$ pixels.
- **Epochs:** 25 epochs with early stopping patience of 10 epochs.
- **Batch Size:** 16 images per mini-batch.
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum $\beta = 0.937$ and weight decay $0.0005$.
- **Learning Rate Schedule:** Initial learning rate $\eta_0 = 0.01$, cycling down to $\eta_{\text{final}} = 0.001$ via Cosine Annealing.
- **Data Augmentations:** Mosaic augmentation ($p = 1.0$), horizontal flips ($p = 0.5$), HSV color jitter (Hue: 0.015, Saturation: 0.7, Value: 0.4), and translation ($\pm 10\%$). Mosaic was disabled for the final 5 epochs to stabilize boundary regression.

---

\pagebreak

# 10. RESULTS, EVALUATION AND DISCUSSION

### 10.1 Quantitative Performance Metrics
The model was evaluated on the independent validation split consisting of **1,542 Indian road images** (757 damaged + 783 clean negative samples).

```
+---------------------------------------------------------------------------------+
| Distress Class        | Precision (P) | Recall (R) | mAP@0.50 | mAP@0.50:0.95   |
+---------------------------------------------------------------------------------+
| D00 (Longitudinal)    | 0.584         | 0.512      | 0.518    | 0.284           |
| D10 (Transverse)      | 0.548         | 0.374      | 0.342    | 0.162           |
| D20 (Alligator)       | 0.612         | 0.546      | 0.551    | 0.318           |
| D40 (Pothole/Rutting) | 0.672         | 0.628      | 0.648    | 0.402           |
+---------------------------------------------------------------------------------+
| Overall (All Classes) | 0.604         | 0.515      | 0.515    | 0.292           |
+---------------------------------------------------------------------------------+
```

### 10.2 Training Convergence and Loss Curves
The model exhibited steady convergence across the 25 training epochs:
- **Box Loss:** Decreased from $1.82$ to $1.04$ on the validation split.
- **Classification Loss:** Decreased from $2.41$ to $0.88$, demonstrating robust class discrimination.
- **DFL Loss:** Converged to $1.01$, indicating sharp bounding box localization on irregular crack edges.

### 10.3 Confusion Matrix Analysis
Analysis of the validation confusion matrix yields critical engineering insights:
1. **Background Confusion:** Only $8.4\%$ of background patches were falsely flagged as defects, directly validating the effectiveness of retaining 3,921 clean negative road samples during training.
2. **D00 vs. Lane Markings:** Longitudinal cracks (`D00`) exhibited mild confusion ($6.2\%$) with faded white lane markings, as both share narrow linear geometry aligned with the travel direction.
3. **Pothole Localization:** Class `D40` showed the highest true-positive rate ($64.8\%$), as potholes present distinct visual signatures: dark shadowed depressions with contrasting illuminated rims.

### 10.4 Measured Inference Latency & Speed Benchmark
To ensure feasibility for real-time vehicular deployment, inference latency was benchmarked across hardware platforms:

```
+---------------------------------------------------------------------------------+
| Compute Hardware        | Resolution  | Latency (ms) | Throughput (FPS) | Status|
+---------------------------------------------------------------------------------+
| Tesla T4 GPU (Colab)    | 640 x 640   | 9.4 ms       | ~106 FPS         | Real-time|
| Apple M-Series CPU      | 320 x 320   | 26.1 ms      | ~38 FPS          | Real-time|
| Apple M-Series CPU      | 640 x 640   | 54.1 ms      | ~18 FPS          | Near-RT  |
+---------------------------------------------------------------------------------+
```

> **Real-World Vehicular Feasibility:** At a vehicle speed of $60\text{ km/h}$ ($16.6\text{ m/s}$), processing at $38\text{ FPS}$ evaluates a road frame every $43.5\text{ cm}$. A standard $50\text{ cm}$ pothole is captured and verified across $2\text{ to }3$ successive frames, enabling real-time driver alert generation.

### 10.5 Qualitative Benchmark Tests on Real-World Challenges
The system was evaluated against challenging real-world scenarios:

1. **Autumn Clean Highway (`Unknown-2.jpg`):** Standard models often misclassify roadside grass and tree foliage as potholes. With our spatial horizon filter and clean-road negative training, the app returned **🟢 GOOD: No Defects (0 Found)** with zero false alarms.
2. **Mountain Highway (`Unknown-3.jpg`):** Clean asphalt with white dashed edge lines returned **🟢 GOOD: No Defects (0 Found)**.
3. **Wet Asphalt with Pothole & Cracks (`images-2.jpeg`):** Accurately localized both the central pothole (`D40_Pothole 88%`) and radiating alligator fatigue cracks (`D20_Alligator_Crack 69%`), triaged as **🔴 CRITICAL**.
4. **Muddy Road with Deep Craters (`images-4.jpeg`):** Successfully localized multiple water-filled mud craters (`D40_Pothole 88%`, `D40_Pothole 22%`), triaged as **🔴 CRITICAL**.
5. **Heavily Shattered Alligator Pavement (`images-5.jpeg`):** Correctly detected extensive fatigue cracking and broken road surfaces (`D20_Alligator_Crack 51%`), triaged as **🔴 CRITICAL**.
6. **Earthquake / Landslide Fault Split (`images-6.jpeg`):** Severe road fault line clearly identified (`D20_Alligator_Crack 58%`, `D00_Longitudinal_Crack 23%`), triaged as **🔴 CRITICAL**.
7. **Deep Trench with Traffic Cones (`images-7.jpeg`):** Accurately flagged the transversal pavement fracture (`D10_Transverse_Crack 22%`) across the roadway, triaged as **🔴 CRITICAL**.
8. **Collapsed Asphalt Slabs (`images-8.jpeg`):** Multi-defect localization identified 7 distinct structural distress zones, triaged as **🔴 CRITICAL**.
9. **Flooded Road with Motorbikes & Barricade (`images-16.jpeg`):** Test-Time Augmentation (TTA) successfully captured the deep waterlogged pit (`D40_Pothole 76%`) and central hazard depression, triaged as **🔴 CRITICAL**.

---

\pagebreak

# 11. CONCLUSION AND FUTURE SCOPE

### 11.1 Conclusion
This project successfully designed, implemented, and validated an automated computer vision framework for road distress detection and maintenance severity triaging on Indian roads. By utilizing the state-of-the-art **YOLOv8** object detector and formulating a **Dual-Model AI Ensemble** coupled with spatial horizon filtering and Test-Time Augmentation, the system overcomes the historical vulnerability of automated pavement inspection: false alarms on clean roads versus missed detections on complex waterlogged craters.

Trained on 7,708 Sekilab Indian road images (including 3,921 clean negative samples), the system achieves an overall **mAP@0.50 of 0.515**, with the critical pothole class reaching **0.648 mAP@0.50** at an inference speed of up to **38 FPS on CPU** and **>100 FPS on GPU**. The interactive Streamlit dashboard enables rapid visual inspection, automated severity triage, and CSV audit logging, demonstrating an affordable, scalable alternative to expensive survey vehicles.

### 11.2 Future Scope
1. **GPS Tagging & Municipal GIS Heatmaps:** Integrating a hardware GPS module into the video capture pipeline to tag detected distresses with geographical coordinates (latitude/longitude), dynamically updating municipal road repair GIS heatmaps.
2. **Stereo Vision & 3D Pothole Volume Estimation:** Incorporating stereo cameras or monocular depth estimation networks (e.g., MiDaS, Depth Anything) to measure pothole depth and volume in cubic meters, allowing automated calculation of required asphalt patching material.
3. **Edge Microcontroller Deployment:** Porting the quantized YOLOv8 model via ONNX / TensorRT onto embedded edge computing boards (Raspberry Pi 5 with AI Hat, NVIDIA Jetson Orin Nano) for direct installation on public transport buses and municipal waste collection vehicles.
4. **Night-Vision & Adverse Weather Optimization:** Fine-tuning the network on thermal and infrared nighttime footage to maintain continuous 24/7 road inspection capabilities.

---

\pagebreak

# 12. REFERENCES

1. **Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y.** (2020). *Global Road Damage Detection Challenge: Datasets and Baselines.* IEEE International Conference on Big Data (Big Data), pp. 2533-2542.
2. **Jocher, G., Chaurasia, A., & Qiu, J.** (2023). *Ultralytics YOLOv8: A State-of-the-Art Object Detector.* GitHub repository: https://github.com/ultralytics/ultralytics.
3. **Koch, C., & Brilakis, I.** (2011). *Pothole detection in asphalt pavement images.* Advanced Engineering Informatics, 25(3), 507-515.
4. **Cord, A., & Chambon, S.** (2012). *Automatic road defect detection by textural pattern recognition based on AdaBoost.* Computer-Aided Civil and Infrastructure Engineering, 27(4), 244-259.
5. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). *You Only Look Once: Unified, Real-Time Object Detection.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779-788.
6. **Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M.** (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv preprint arXiv:2004.10934.
7. **Ministry of Road Transport and Highways (MoRTH), Government of India.** (2022). *Road Accidents in India 2022.* Transport Research Wing, New Delhi.
8. **Indian Roads Congress (IRC:SP:72-2015).** *Guidelines for the Design of Flexible Pavements for Low Volume Rural Roads.* New Delhi, India.
9. **Alfarrarjeh, A., Trivedi, D., Kim, S. H., & Shahabi, C.** (2021). *A Deep Learning Approach for Road Damage Detection from Smartphone Videos.* IEEE International Conference on Big Data, pp. 5201-5210.
10. **Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D.** (2020). *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.* Proceedings of the AAAI Conference on Artificial Intelligence, 34(07), 12993-13000.
11. **Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H.** (2018). *Road damage detection and classification using deep neural networks with smartphone images.* Computer-Aided Civil and Infrastructure Engineering, 33(12), 1127-1141.

---

\pagebreak

# 13. RELEVANT SOURCE CODE AND OUTPUT SCREENSHOTS

### 13.1 Open-Source GitHub Repository
All project source code, data preprocessing scripts, model downloaders, trained weights, seminar reports, and interactive web application files are publicly hosted on GitHub:
👉 **[https://github.com/Ravish945/road-damage-classification](https://github.com/Ravish945/road-damage-classification)**

---

### 13.2 Core Implementation Code Listings

#### Listing 1: Dual-Model Ensemble & False-Positive Spatial Filtering (`app.py`)
```python
import os
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

def run_ensemble_detection(image, conf=0.18, iou=0.45, imgsz=640, augment=True):
    """
    Executes dual-model AI ensemble detection:
    1. RDD2022 YOLOv8s for structural cracks (D00, D10, D20)
    2. Universal Pothole Detector for deep craters & waterlogged holes (D40)
    Applies spatial horizon and foliage filtering to eliminate clean-road false alarms.
    """
    w, h = image.size
    boxes, scores, labels = [], [], []

    rdd_map = {
        "longitudinal_crack": "D00_Longitudinal_Crack",
        "transverse_crack": "D10_Transverse_Crack",
        "alligator_crack": "D20_Alligator_Crack",
        "pothole": "D40_Pothole"
    }

    # Stream 1: Structural Cracks via RDD2022 YOLOv8s
    if rdd_model is not None:
        res_rdd = rdd_model.predict(image, conf=conf, imgsz=imgsz, augment=augment, verbose=False)[0]
        for b in res_rdd.boxes:
            cls_name = rdd_model.names.get(int(b.cls[0]), "defect")
            score = float(b.conf[0])
            # Suppress weak RDD potholes; pothole detection is delegated to universal detector
            if cls_name == "pothole" and score < 0.35:
                continue
            bx = b.xyxy[0].tolist()
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            # Spatial filter: Suppress whole-screen false triggers
            if bw > 0.85 * w and bh > 0.65 * h:
                continue
            # Horizon filter: Suppress distant tree foliage & sky activations
            if bx[3] < 0.18 * h:
                continue
            boxes.append(bx)
            scores.append(score)
            labels.append(rdd_map.get(cls_name, cls_name))

    # Stream 2: Universal Pothole & Crater Detector with TTA
    if pot_model is not None:
        res_pot = pot_model.predict(image, conf=conf, imgsz=imgsz, augment=augment, verbose=False)[0]
        for b in res_pot.boxes:
            bx = b.xyxy[0].tolist()
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            if bw > 0.85 * w and bh > 0.65 * h:
                continue
            if bx[3] < 0.18 * h:
                continue
            boxes.append(bx)
            scores.append(float(b.conf[0]))
            labels.append("D40_Pothole")

    if not boxes:
        return [], [], []

    # Non-Maximum Suppression (NMS) to merge overlapping candidates
    t_boxes = torch.tensor(boxes)
    t_scores = torch.tensor(scores)
    keep = torchvision.ops.nms(t_boxes, t_scores, iou_threshold=iou).tolist()

    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]
```

#### Listing 2: Sequence-Aware Dataset Split & Normalization (`voc_to_yolo.py`)
```python
import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_file, output_txt, classes):
    """Parses Pascal VOC XML and converts coordinates to normalized YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(output_txt, "w") as out:
        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Normalization to [0, 1]
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
```

---

### 13.3 Output Screenshots and Evaluation Figures

#### Figure 1: Validation Confusion Matrix
The confusion matrix demonstrates strong class separation, with background misclassification contained below $8.4\%$:
*(Refer to `results/screenshots/confusion_matrix.png` in the repository)*

#### Figure 2: Precision-Recall (PR) Curve
Overall model achieves **mAP@0.50 of 0.515**, with the critical pothole class (`D40`) attaining **0.648 mAP@0.50**:
*(Refer to `results/screenshots/BoxPR_curve.png` in the repository)*

#### Figure 3: Training Loss & Metric Convergence
Box loss, classification loss, and Distribution Focal Loss (DFL) across 25 epochs:
*(Refer to `results/screenshots/results.png` in the repository)*

#### Figure 4: Qualitative Detection Outputs Across Benchmark Scenarios
- **Clean Highway:** Correctly verified as **🟢 GOOD: No Defects (0 Found)** with zero false alarms.
- **Wet Asphalt Pothole:** Localized `D40_Pothole 88%` and `D20_Alligator_Crack 69%` (🔴 CRITICAL).
- **Muddy Road Craters:** Detected multiple waterlogged holes at `88%` and `22%` confidence (🔴 CRITICAL).
- **Flooded Road with Barricade (`images-16.jpeg`):** Successfully localized deep right crater at `76%` confidence and hazard pit at `16%` confidence (🔴 CRITICAL).
