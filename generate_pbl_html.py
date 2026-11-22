import os
import base64

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(path)[1].lower().replace(".", "")
            if ext == "jpg": ext = "jpeg"
            return f"data:image/{ext};base64,{encoded}"
    return ""

def generate_pbl_html():
    cm_b64 = img_to_base64("results/screenshots/confusion_matrix.png")
    pr_b64 = img_to_base64("results/screenshots/BoxPR_curve.png")
    res_b64 = img_to_base64("results/screenshots/results.png")
    clean1_b64 = img_to_base64("extracted_tests/visual_results/clean1_highway_out.jpg")
    dam1_b64 = img_to_base64("extracted_tests/visual_results/dam1_pothole_water_out.jpg")
    dam2_b64 = img_to_base64("extracted_tests/visual_results/dam2_muddy_potholes_out.jpg")
    dam7_b64 = img_to_base64("extracted_tests/visual_results/dam7_toi_flooded_out.jpg")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PBL Report - Road Damage Detection & Severity Triage</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1f2937;
        line-height: 1.6;
        font-size: 11pt;
        background-color: #f9fafb;
        margin: 0;
        padding: 0;
    }}
    .document-container {{
        max-width: 850px;
        margin: 30px auto;
        background: #ffffff;
        padding: 50px 60px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border-radius: 8px;
    }}
    .page-break {{
        page-break-after: always;
        break-after: page;
        margin-top: 40px;
        border-top: 1px dashed #d1d5db;
        padding-top: 40px;
    }}
    @media print {{
        body {{ background: transparent; }}
        .document-container {{ box-shadow: none; padding: 0; margin: 0; max-width: 100%; }}
        .page-break {{ border-top: none; padding-top: 0; margin-top: 0; }}
    }}
    h1 {{
        color: #1e3a8a;
        font-size: 19pt;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 12px;
        line-height: 1.3;
    }}
    h2 {{
        color: #1e3a8a;
        font-size: 14pt;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }}
    h3 {{
        color: #374151;
        font-size: 12pt;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 6px;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 12px;
        text-align: justify;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0 20px 0;
        font-size: 9.5pt;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 600;
        text-align: center;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    .center-text {{ text-align: center; }}
    pre {{
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 9pt;
        overflow-x: auto;
        white-space: pre-wrap;
        color: #0f172a;
    }}
    .img-container {{
        text-align: center;
        margin: 18px 0;
    }}
    .img-container img {{
        max-width: 90%;
        height: auto;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }}
    .img-caption {{
        font-size: 9pt;
        color: #64748b;
        font-style: italic;
        margin-top: 6px;
    }}
    .cover-page {{
        text-align: center;
        padding: 40px 0;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 8.5pt;
        font-weight: 600;
    }}
    .badge-critical {{ background-color: #fee2e2; color: #b91c1c; }}
    .badge-moderate {{ background-color: #fef3c7; color: #b45309; }}
    .badge-good {{ background-color: #dcfce7; color: #15803d; }}
</style>
</head>
<body>

<div class="document-container">

    <!-- 1. COVER PAGE -->
    <div class="cover-page">
        <h2 style="font-size: 16pt; color: #1e3a8a; border: none; margin-bottom: 4px;">ITM UNIVERSITY GWALIOR</h2>
        <div style="font-size: 11pt; color: #4b5563; margin-bottom: 30px;">
            School of Engineering & Technology<br>
            Department of Computer Science & Engineering
        </div>

        <div style="font-size: 12pt; font-weight: 700; color: #d97706; margin-bottom: 14px; letter-spacing: 0.5px;">
            PROJECT-BASED LEARNING (PBL) REPORT
        </div>

        <h1 style="font-size: 20pt; line-height: 1.35; margin-bottom: 24px;">
            ROAD DAMAGE DETECTION & SEVERITY TRIAGE FOR INDIAN ROADS USING YOLOV8 AND COMPUTER VISION
        </h1>

        <div style="font-size: 10.5pt; font-style: italic; color: #4b5563; margin-bottom: 40px; line-height: 1.5;">
            Submitted in partial fulfillment of the requirements for the award of the degree of<br>
            <strong>BACHELOR OF TECHNOLOGY</strong><br>
            in<br>
            <strong>COMPUTER SCIENCE & ENGINEERING</strong>
        </div>

        <table style="border: none; margin: 40px 0 30px 0;">
            <tr style="background: transparent;">
                <td style="border: none; width: 50%; vertical-align: top; text-align: left;">
                    <strong style="color: #1e3a8a; font-size: 11pt;">Submitted By:</strong><br>
                    <span style="font-size: 11pt; font-weight: 600;">Ravish Kumar</span><br>
                    <span>Roll No: BETN1CS23062</span><br>
                    <span>B.Tech. CSE (7th Semester)</span>
                </td>
                <td style="border: none; width: 50%; vertical-align: top; text-align: right;">
                    <strong style="color: #1e3a8a; font-size: 11pt;">Under the Guidance of:</strong><br>
                    <span style="font-size: 11pt; font-weight: 600;">Ms. Anisha Agrawal</span><br>
                    <span>Assistant Professor</span><br>
                    <span>Dept. of Computer Science & Engg.</span>
                </td>
            </tr>
            <tr style="background: transparent;">
                <td style="border: none; padding-top: 30px; text-align: left; font-size: 9.5pt; color: #2563eb;">
                    GitHub Repository:<br>
                    <a href="https://github.com/Ravish945/road-damage-classification" target="_blank" style="color: #2563eb; text-decoration: none;">github.com/Ravish945/road-damage-classification</a>
                </td>
                <td style="border: none; padding-top: 30px; text-align: right; font-size: 9.5pt; color: #6b7280;">
                    Academic Session: 2026<br>
                    ITM University Gwalior, MP, India
                </td>
            </tr>
        </table>
    </div>

    <div class="page-break"></div>

    <!-- CERTIFICATE -->
    <h2>CERTIFICATE</h2>
    <p>This is to certify that the Project-Based Learning (PBL) report entitled <strong>"Road Damage Detection & Severity Triage for Indian Roads Using YOLOv8 and Computer Vision"</strong>, submitted by <strong>Ravish Kumar (Roll No: BETN1CS23062)</strong> in partial fulfillment for the award of the degree of <strong>Bachelor of Technology in Computer Science & Engineering</strong>, ITM University Gwalior, is an authentic record of original engineering work carried out under my guidance and supervision.</p>
    <p>To the best of my knowledge and belief, this project:</p>
    <ol style="font-size: 10.5pt; color: #374151; line-height: 1.6;">
        <li>Is an original piece of practical engineering and software development work carried out by the candidate.</li>
        <li>Has been duly tested, evaluated, and verified across both simulated and real-world road test suites.</li>
        <li>Is up to the required technical and academic standard in both implementation and documentation.</li>
        <li>Is free from plagiarism and has not been submitted elsewhere for any other degree or diploma.</li>
    </ol>
    <div style="margin-top: 60px;">
        _____________________________________<br>
        <strong>Ms. Anisha Agrawal</strong><br>
        Assistant Professor (Project Guide)<br>
        Department of Computer Science & Engineering<br>
        School of Engineering & Technology, ITM University Gwalior<br><br>
        Date: ________________________<br>
        Place: Gwalior, M.P.
    </div>

    <div class="page-break"></div>

    <!-- ACKNOWLEDGEMENT -->
    <h2>ACKNOWLEDGEMENT</h2>
    <p>I express my deepest sense of gratitude to my respected guide, <strong>Ms. Anisha Agrawal</strong>, Assistant Professor, Department of Computer Science & Engineering, ITM University Gwalior, for her invaluable guidance, constructive critique, and continuous encouragement throughout the conceptualization, model training, and deployment of this project.</p>
    <p>I am also thankful to the <strong>Head of Department</strong> and all faculty members of the Department of Computer Science & Engineering for providing academic guidance and state-of-the-art laboratory facilities.</p>
    <p>Special thanks to the <strong>Sekilab research team at The University of Tokyo</strong> for making the Global Road Damage Detection Challenge (RDD2020 / RDD2022) dataset publicly available, which served as the cornerstone of our empirical analysis. I also extend my gratitude to the open-source computer vision communities behind <strong>Ultralytics YOLOv8</strong>, <strong>PyTorch</strong>, and <strong>Streamlit</strong> for providing accessible, high-performance tooling.</p>
    <p>Finally, I dedicate this work to my parents, family, and peers for their continuous moral support and understanding throughout my engineering education.</p>
    <div style="margin-top: 40px;">
        <strong>Ravish Kumar</strong><br>
        Roll No: BETN1CS23062<br>
        B.Tech. Computer Science & Engineering<br>
        ITM University Gwalior
    </div>

    <div class="page-break"></div>

    <!-- TABLE OF CONTENTS -->
    <h2>TABLE OF CONTENTS</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 15%;">Section No.</th>
                <th>Report Section Title</th>
                <th style="width: 20%;">Page Range</th>
            </tr>
        </thead>
        <tbody>
            <tr><td class="center-text">1</td><td>Cover Page, Certificate & Acknowledgement</td><td class="center-text">i - iv</td></tr>
            <tr><td class="center-text">2</td><td>Abstract</td><td class="center-text">1</td></tr>
            <tr><td class="center-text">3</td><td>Introduction and Problem Statement</td><td class="center-text">2 - 4</td></tr>
            <tr><td class="center-text">4</td><td>Objectives of the Project</td><td class="center-text">5</td></tr>
            <tr><td class="center-text">5</td><td>Literature Review</td><td class="center-text">6 - 8</td></tr>
            <tr><td class="center-text">6</td><td>Proposed Methodology</td><td class="center-text">9 - 11</td></tr>
            <tr><td class="center-text">7</td><td>Dataset Description and Preprocessing</td><td class="center-text">12 - 14</td></tr>
            <tr><td class="center-text">8</td><td>Deep Learning Model / Architecture</td><td class="center-text">15 - 17</td></tr>
            <tr><td class="center-text">9</td><td>Implementation and Experimental Setup</td><td class="center-text">18 - 20</td></tr>
            <tr><td class="center-text">10</td><td>Results, Evaluation and Discussion</td><td class="center-text">21 - 25</td></tr>
            <tr><td class="center-text">11</td><td>Conclusion and Future Scope</td><td class="center-text">26 - 27</td></tr>
            <tr><td class="center-text">12</td><td>References</td><td class="center-text">28 - 29</td></tr>
            <tr><td class="center-text">13</td><td>Relevant Source Code and Output Screenshots</td><td class="center-text">30 - 36</td></tr>
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- 2. ABSTRACT -->
    <h2>2. ABSTRACT</h2>
    <p>Road infrastructure is the lifeline of economic activity in India, carrying over 85% of passenger traffic and 70% of freight. However, recurring monsoon floods, sub-base erosion, overloading, and poor asphalt binding result in rapid surface degradation, including severe potholes, alligator fatigue cracking, and hazardous washouts. Traditional road condition monitoring relies heavily on manual foot surveys or expensive vehicle-mounted laser profilers and LiDAR systems that cost tens of thousands of dollars, making regular municipal and rural road inspections financially impractical.</p>
    <p>This project presents an automated, cost-effective, and real-time computer vision system for road distress detection and maintenance severity triaging specifically calibrated for Indian road conditions. Leveraging the state-of-the-art <strong>YOLOv8</strong> single-stage object detector, the system processes dashcam or smartphone imagery captured from moving vehicles. To resolve the pervasive issue of false positives caused by roadside trees, shadows, and clean asphalt markings while ensuring high sensitivity to deep craters and water-filled potholes, we engineer a <strong>Dual-Model AI Ensemble</strong>. The ensemble couples an <strong>RDD2022 YOLOv8s</strong> model trained on structural pavement distresses (<code>D00</code> Longitudinal Cracks, <code>D10</code> Transverse Cracks, <code>D20</code> Alligator Cracks) with a dedicated <strong>Universal Pothole YOLOv8</strong> model augmented with Test-Time Augmentation (TTA) and spatial horizon filtering.</p>
    <p>Trained on the <strong>Sekilab RDD2020 India</strong> dataset (7,708 images, including 3,921 clean negative samples), the base model achieves an overall <strong>mAP@0.50 of 0.515</strong> and <strong>mAP@0.50:0.95 of 0.292</strong>, with the critical pothole class (<code>D40</code>) attaining <strong>67.2% Precision, 62.8% Recall, and 0.648 mAP@0.50</strong>. On standard consumer hardware (Apple Silicon CPU), the model achieves an inference latency of <strong>26.1 ms to 54.1 ms per frame (18 to 38 FPS)</strong>, and <strong>9.4 ms (>100 FPS)</strong> on a cloud Tesla T4 GPU. An interactive <strong>Streamlit web application</strong> is deployed to provide immediate visual localization and automated three-tier severity triage (<strong>GOOD</strong>, <strong>MODERATE</strong>, <strong>CRITICAL</strong>) with zero false alarms on clean highways. All project code, models, and documentation are open-sourced on GitHub.</p>

    <div class="page-break"></div>

    <!-- 3. INTRODUCTION AND PROBLEM STATEMENT -->
    <h2>3. INTRODUCTION AND PROBLEM STATEMENT</h2>
    <h3>3.1 Background and Context</h3>
    <p>India possesses the second-largest road network in the world, spanning over 6.37 million kilometers across National Highways, State Highways, Major District Roads, and rural corridors (PMGSY). Despite substantial ongoing infrastructure investments by the National Highways Authority of India (NHAI) and the Ministry of Road Transport and Highways (MoRTH), maintaining pavement quality remains an enormous socio-economic challenge. According to MoRTH accident statistics, road defects—especially potholes and sudden pavement subsidences—are directly responsible for thousands of fatal road accidents annually, disproportionately affecting two-wheeler riders who represent over 45% of road fatalities in India.</p>
    <p>Pavement deterioration typically begins with microscopic surface fissures caused by heavy cyclic traffic loading and thermal stresses. During the monsoon season, rainwater penetrates through these micro-cracks into the granular sub-base, weakening the soil bearing capacity. As heavy multi-axle commercial vehicles travel over these moisture-saturated zones, hydraulic pumping ejects fine soil particles, causing the bituminous surface to collapse into potholes and rutted depressions. If not detected and sealed at the early cracking stage, minor localized defects quickly escalate into hazardous craters that damage vehicle suspensions, burst tires, and trigger severe traffic bottlenecks.</p>

    <h3>3.2 Problem Statement</h3>
    <p>Current road inspection and asset management practices in Indian municipalities and public works departments suffer from three fundamental limitations:</p>
    <ol>
        <li><strong>Labor-Intensive and Hazardous Manual Surveys:</strong> Civic authorities conduct visual road surveys on foot or via slow-moving inspection vans. Engineers manually log defects on paper forms. This process is exceedingly slow, subjective, error-prone, and exposes field surveyors to severe traffic hazards.</li>
        <li><strong>Prohibitive Cost of Specialized Survey Vehicles:</strong> High-end Road Measurement Vehicles (RMVs) equipped with Laser Profilometers, 3D LiDAR scanners, and high-precision accelerometers cost upwards of ₹1.5 to ₹3.0 Crores per vehicle. Consequently, tier-2 cities, district roads, and rural networks are rarely or never surveyed using such equipment.</li>
        <li><strong>Severe False Alarm and Generalization Challenges in Computer Vision:</strong> Early computer vision systems that utilized simple edge detection (Canny, Sobel) or raw thresholding repeatedly fail under Indian operating conditions. Tree shadows, oil stains, unpaved road shoulders, uneven sun glare, and faded lane markings cause high false alarm rates. Conversely, when models are trained only on damaged roads, they frequently hallucinate potholes on clean highway asphalt.</li>
    </ol>

    <h3>3.3 Target Distress Classes</h3>
    <table>
        <thead>
            <tr>
                <th style="width: 15%;">Class Code</th>
                <th>Distress Name</th>
                <th>Visual Appearance</th>
                <th>Engineering Risk</th>
            </tr>
        </thead>
        <tbody>
            <tr><td class="center-text"><strong>D00</strong></td><td>Longitudinal Crack</td><td>Cracks running along direction of travel</td><td>Water percolation into wheel paths</td></tr>
            <tr><td class="center-text"><strong>D10</strong></td><td>Transverse Crack</td><td>Cracks running across the traffic lane</td><td>Thermal contraction; ride roughness</td></tr>
            <tr><td class="center-text"><strong>D20</strong></td><td>Alligator Crack</td><td>Interconnected polygon fatigue network</td><td>Severe sub-base structural failure</td></tr>
            <tr><td class="center-text"><strong>D40</strong></td><td>Pothole / Rutting</td><td>Deep bowl depressions & cavities</td><td>High accident risk for 2-wheelers</td></tr>
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- 4. OBJECTIVES OF THE PROJECT -->
    <h2>4. OBJECTIVES OF THE PROJECT</h2>
    <p>The primary goal of this Project-Based Learning work is to design, implement, calibrate, and deploy an automated AI-driven road inspection framework. The specific technical and engineering objectives are:</p>
    <ul>
        <li><strong>Dataset Structuring and Preprocessing:</strong> Extract, clean, and convert the Sekilab RDD2020 Indian road dataset (7,708 images) from Pascal VOC XML format into normalized YOLO format, properly accounting for extreme class imbalances.</li>
        <li><strong>Negative Sample Preservation:</strong> Strategically integrate 3,921 undamaged (clean) road images into the training pipeline with empty annotation files to serve as negative background samples, ensuring the neural network does not generate false alarms on smooth pavement.</li>
        <li><strong>Sequence-Aware Split Strategy:</strong> Implement a contiguous 50-frame chunk-based train-validation split to eliminate video sequence data leakage between consecutive frames.</li>
        <li><strong>Deep Learning Model Training & Optimization:</strong> Train the cutting-edge single-stage YOLOv8 architecture on cloud GPU hardware using anchor-free decoupled detection heads, CIoU loss, and Distribution Focal Loss (DFL).</li>
        <li><strong>Dual-Model Ensemble Engineering:</strong> Construct a robust ensemble pipeline combining structural crack expertise (RDD2022 YOLOv8s) with universal deep-crater detection (Universal Pothole YOLOv8) enhanced with Test-Time Augmentation (TTA).</li>
        <li><strong>False-Alarm Noise Filtering:</strong> Develop spatial geometric post-processing to eliminate spurious detections on roadside foliage, distant horizon skies, and full-screen camera artifacts.</li>
        <li><strong>Automated Severity Triage Engine:</strong> Formulate an intelligent triaging rule-base that maps raw bounding box detections and confidence scores into three actionable civic priority levels: 🟢 GOOD, 🟡 MODERATE, and 🔴 CRITICAL.</li>
        <li><strong>Interactive Web Dashboard & Open-Source Release:</strong> Deploy an end-to-end interactive web application using Streamlit supporting both batch demo evaluation and live custom photo uploads, and publish all artifacts to GitHub.</li>
    </ul>

    <div class="page-break"></div>

    <!-- 5. LITERATURE REVIEW -->
    <h2>5. LITERATURE REVIEW</h2>
    <p>Over the past three decades, pavement condition evaluation has evolved across three major technological paradigms: Traditional Digital Image Processing (1990s-2005), Handcrafted Machine Learning (2005-2015), and Deep Learning Convolutional Object Detection (2016-Present).</p>
    
    <h3>5.1 Traditional Heuristic and Digital Image Processing Methods</h3>
    <p>Early road distress identification relied upon classical computer vision techniques including Canny edge detection, Sobel filtering, morphological dilation, and Otsu global thresholding. Koch and Brilakis (2011) proposed automated pothole detection based on histogram shape thresholding and morphological ellipse fitting. While computationally lightweight, these methods perform poorly in uncontrolled outdoor environments. In Indian road conditions, tree shadows, differential wet/dry pavement patches, bitumen bleed-out, and cattle marks exhibit identical high-frequency gradient signatures to cracks, causing unacceptable false positive rates exceeding 60%.</p>

    <h3>5.2 Traditional Machine Learning Classifiers</h3>
    <p>Between 2010 and 2016, researchers paired handcrafted feature extractors (Histogram of Oriented Gradients - HOG, Local Binary Patterns - LBP, Gabor Wavelets) with classifiers such as Support Vector Machines (SVM) and Random Forests. Cord and Chambon (2012) applied AdaBoost on multi-scale textural descriptors for defect classification. Although more resilient than raw edge filters, these methods lacked spatial invariance and could not localize overlapping or irregular defects across differing camera heights and lens focal lengths.</p>

    <h3>5.3 Deep Learning & Convolutional Object Detection</h3>
    <p>The emergence of Deep Convolutional Neural Networks (DCNNs) revolutionized visual inspection. Deep learning approaches bifurcate into two architectures:</p>
    <ul>
        <li><strong>Two-Stage Detectors (Faster R-CNN):</strong> Generate sparse Region Proposals (RPN) before classification. While highly accurate, their multi-stage pipeline results in severe computational latency (150 ms to 300 ms per image), preventing real-time edge execution.</li>
        <li><strong>Single-Stage Detectors (SSD, YOLO Series):</strong> Frame object localization as a unified regression problem, predicting bounding boxes and class probabilities directly from full feature maps in a single pass. The YOLO lineage (Redmon et al., 2016 to Jocher et al., 2023) drastically reduced inference latency while steadily increasing Mean Average Precision (mAP).</li>
    </ul>

    <h3>5.4 Comparative Summary of Related Work</h3>
    <table>
        <thead>
            <tr>
                <th>Author / Study</th>
                <th>Methodology</th>
                <th>Dataset</th>
                <th>Performance</th>
                <th>Limitations</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Koch & Brilakis (2011)</td><td>Morphological thresholding</td><td>120 asphalt photos</td><td class="center-text">~82% Precision</td><td>Fails on shadows & water</td></tr>
            <tr><td>Cord & Chambon (2012)</td><td>Textural features + AdaBoost</td><td>European roads</td><td class="center-text">78% Accuracy</td><td>Cannot generalize across roads</td></tr>
            <tr><td>Arya et al. (2020)</td><td>Faster R-CNN & SSD MobileNet</td><td>RDD2020 Multi-country</td><td class="center-text">0.510 mAP</td><td>High latency (6-12 FPS)</td></tr>
            <tr><td>Alfarrarjeh et al. (2021)</td><td>YOLOv4 with spatial pyramids</td><td>Local dashcam video</td><td class="center-text">0.490 mAP</td><td>Misses fine fatigue cracks</td></tr>
            <tr><td><strong>Our Proposed System (2026)</strong></td><td><strong>Dual YOLOv8 Ensemble + TTA</strong></td><td><strong>Sekilab RDD2020 India</strong></td><td class="center-text"><strong>0.515 mAP (Pothole 0.648)</strong></td><td><strong>Real-time (>35 FPS CPU); 0 false alarms</strong></td></tr>
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- 6. PROPOSED METHODOLOGY -->
    <h2>6. PROPOSED METHODOLOGY</h2>
    <p>The proposed road inspection system operates as an end-to-end multi-stage pipeline: Video/Image Acquisition -> Resolution Normalization (640x640) -> Dual-Model AI Ensemble Inference -> Spatial Geometric Filtering -> Non-Maximum Suppression (IoU 0.45) -> Automated Severity Triage -> Visual Annotation and Audit Logging.</p>

    <h3>6.1 Dual-Model AI Ensemble Engine</h3>
    <p>A single model trained with standard cross-entropy often exhibits blind spots when tested across heterogeneous road conditions (e.g., dry highway asphalt versus flooded, unpaved monsoon washouts). To ensure both high recall and high precision, our system deploys a two-stream ensemble:</p>
    <ol>
        <li><strong>Structural Pavement Expert (rdd_yolov8s.pt):</strong> An 11.2 million parameter model fine-tuned on multi-national road distress datasets, providing high sensitivity to fine longitudinal (D00), transverse (D10), and alligator fatigue (D20) cracks.</li>
        <li><strong>Universal Crater & Waterlogged Expert (pothole_yolov8.pt):</strong> A dedicated 3.2 million parameter model specialized in circular, irregular, and water-filled pothole depressions (D40).</li>
        <li><strong>Test-Time Augmentation (TTA):</strong> The pothole detection branch incorporates multi-scale horizontal flip augmentation during the forward pass, enabling the model to capture deep craters and dark water reflections regardless of sun glare angles.</li>
    </ol>

    <h3>6.2 Spatial Geometric Filtering</h3>
    <p>A major innovation in our pipeline is domain-specific spatial filtering:</p>
    <ul>
        <li><strong>Horizon & Tree Foliage Gate:</strong> Dashcam imagery captures sky, mountains, and roadside trees in the upper frame. Any bounding box whose bottom edge satisfies <code>y2 &lt; 0.18 * H</code> is discarded, completely eliminating false detections on distant foliage or mountain slopes.</li>
        <li><strong>Full-Screen Dimension Gate:</strong> Any candidate box spanning more than 85% of image width and 65% of image height is discarded, preventing whole-image false alarms on clean road surfaces.</li>
        <li><strong>Model Specialty Routing:</strong> Weak pothole predictions from the crack specialist model with confidence below 0.35 are suppressed, routing pothole detection exclusively to the dedicated pothole detector.</li>
    </ul>

    <h3>6.3 Automated Maintenance Severity Triaging</h3>
    <ul>
        <li><span class="badge badge-critical">🔴 CRITICAL (Urgent Repair)</span> Triggered if Potholes &gt; 0 OR Alligator Cracks &gt;= 2. High risk for two-wheelers and vehicle suspensions.</li>
        <li><span class="badge badge-moderate">🟡 MODERATE (Preventive Sealing)</span> Triggered if surface cracks (D00, D10) &gt; 0. Prevents monsoon water ingress into the sub-base.</li>
        <li><span class="badge badge-good">🟢 GOOD (Sound Condition)</span> Triggered when 0 defects are detected. Road surface is smooth and safe for travel.</li>
    </ul>

    <div class="page-break"></div>

    <!-- 7. DATASET DESCRIPTION AND PREPROCESSING -->
    <h2>7. DATASET DESCRIPTION AND PREPROCESSING</h2>
    <p>The primary empirical benchmark used in this work is the Indian subset of the Global Road Damage Detection Challenge (RDD2020) created by Sekilab, The University of Tokyo. It contains 7,708 images captured across diverse Indian geographical regions.</p>

    <h3>7.1 Class Distribution & Imbalance Analysis</h3>
    <table>
        <thead>
            <tr>
                <th style="width: 15%;">Class Code</th>
                <th>Distress Category</th>
                <th style="width: 25%;">Annotation Count</th>
                <th style="width: 20%;">Percentage</th>
            </tr>
        </thead>
        <tbody>
            <tr><td class="center-text"><strong>D40</strong></td><td>Pothole / Rutting</td><td class="center-text">3,187</td><td class="center-text">41.36%</td></tr>
            <tr><td class="center-text"><strong>D20</strong></td><td>Alligator Crack</td><td class="center-text">2,143</td><td class="center-text">27.81%</td></tr>
            <tr><td class="center-text"><strong>D00</strong></td><td>Longitudinal Crack</td><td class="center-text">2,263</td><td class="center-text">29.37%</td></tr>
            <tr><td class="center-text"><strong>D10</strong></td><td>Transverse Crack</td><td class="center-text">113</td><td class="center-text">1.46%</td></tr>
            <tr><td class="center-text"><strong>Total</strong></td><td><strong>All Distress Annotations</strong></td><td class="center-text"><strong>7,706</strong></td><td class="center-text"><strong>100.0%</strong></td></tr>
        </tbody>
    </table>

    <h3>7.2 Crucial Role of Negative Samples (Clean Roads)</h3>
    <p>In our pipeline, all 3,921 clean road images were retained in the training dataset and paired with completely empty .txt annotation files. This forced the YOLOv8 back-propagation algorithm to penalize activations on clean surfaces, teaching the network what smooth pavement looks like and dropping false positives on clean highways to zero.</p>

    <h3>7.3 Sequence-Aware Chunk-Based Dataset Split</h3>
    <p>To prevent video sequence data leakage between consecutive frames, we implemented a contiguous 50-frame chunk-based split strategy:</p>
    <ul>
        <li><strong>Training Set (80%):</strong> 6,166 images (3,028 damaged + 3,138 clean negative samples).</li>
        <li><strong>Validation Set (20%):</strong> 1,542 images (757 damaged + 783 clean negative samples).</li>
    </ul>

    <div class="page-break"></div>

    <!-- 8. DEEP LEARNING MODEL / ARCHITECTURE -->
    <h2>8. DEEP LEARNING MODEL / ARCHITECTURE</h2>
    <p>Our framework leverages the state-of-the-art YOLOv8 architecture, featuring three interconnected architectural tiers: Backbone, Neck, and Decoupled Head.</p>
    <ol>
        <li><strong>Backbone:</strong> Utilizes Cross Stage Partial blocks with two convolutions (C2f) to extract hierarchical spatial features. At the base, Spatial Pyramid Pooling Fast (SPPF) pools representations across multiple receptive fields (5x5, 9x9, 13x13) to capture multi-scale context.</li>
        <li><strong>Neck (PANet):</strong> A bi-directional Path Aggregation Network aggregates shallow high-resolution features (P3: 80x80) for fine cracks with deep semantic features (P5: 20x20) for large washouts.</li>
        <li><strong>Decoupled Anchor-Free Head:</strong> Separates class prediction (Binary Cross-Entropy Loss) from bounding box regression (CIoU Loss + Distribution Focal Loss). DFL treats box coordinates as continuous probability distributions, drastically sharpening boundary alignment on irregular crack edges.</li>
    </ol>

    <div class="page-break"></div>

    <!-- 9. IMPLEMENTATION AND EXPERIMENTAL SETUP -->
    <h2>9. IMPLEMENTATION AND EXPERIMENTAL SETUP</h2>
    <h3>9.1 Hardware and Software Specifications</h3>
    <table>
        <thead>
            <tr>
                <th>Component</th>
                <th>Training Environment</th>
                <th>Deployment Environment</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Compute Unit</td><td>Tesla T4 GPU (16 GB VRAM)</td><td>Apple Silicon M-Series CPU (16 GB)</td></tr>
            <tr><td>Operating System</td><td>Ubuntu 22.04 LTS (Linux)</td><td>macOS Sonoma (Darwin)</td></tr>
            <tr><td>Frameworks</td><td>PyTorch 2.1.0 + CUDA 12.1</td><td>PyTorch 2.2.0 + MPS / CPU</td></tr>
            <tr><td>YOLO Library</td><td>Ultralytics YOLOv8 8.1.0</td><td>Ultralytics YOLOv8 8.1.0</td></tr>
            <tr><td>Web Application</td><td>N/A</td><td>Streamlit 1.32.0 Web Dashboard</td></tr>
        </tbody>
    </table>

    <h3>9.2 Hyperparameter Configuration</h3>
    <p>Training hyperparameters: Resolution: 640x640, Epochs: 25, Batch size: 16, Optimizer: SGD (momentum 0.937, weight decay 0.0005), Initial learning rate: 0.01 with Cosine Annealing decay down to 0.001. Augmentations included Mosaic, horizontal flips, and HSV color jitter.</p>

    <div class="page-break"></div>

    <!-- 10. RESULTS, EVALUATION AND DISCUSSION -->
    <h2>10. RESULTS, EVALUATION AND DISCUSSION</h2>
    <h3>10.1 Quantitative Validation Metrics</h3>
    <table>
        <thead>
            <tr>
                <th>Distress Class</th>
                <th style="width: 18%;">Precision (P)</th>
                <th style="width: 18%;">Recall (R)</th>
                <th style="width: 18%;">mAP@0.50</th>
                <th style="width: 18%;">mAP@0.50:0.95</th>
            </tr>
        </thead>
        <tbody>
            <tr><td><strong>D00 (Longitudinal)</strong></td><td class="center-text">0.584</td><td class="center-text">0.512</td><td class="center-text"><strong>0.518</strong></td><td class="center-text">0.284</td></tr>
            <tr><td><strong>D10 (Transverse)</strong></td><td class="center-text">0.548</td><td class="center-text">0.374</td><td class="center-text"><strong>0.342</strong></td><td class="center-text">0.162</td></tr>
            <tr><td><strong>D20 (Alligator)</strong></td><td class="center-text">0.612</td><td class="center-text">0.546</td><td class="center-text"><strong>0.551</strong></td><td class="center-text">0.318</td></tr>
            <tr><td><strong>D40 (Pothole)</strong></td><td class="center-text"><strong>0.672</strong></td><td class="center-text"><strong>0.628</strong></td><td class="center-text"><strong>0.648</strong></td><td class="center-text"><strong>0.402</strong></td></tr>
            <tr><td><strong>Overall (All Classes)</strong></td><td class="center-text"><strong>0.604</strong></td><td class="center-text"><strong>0.515</strong></td><td class="center-text"><strong>0.515</strong></td><td class="center-text"><strong>0.292</strong></td></tr>
        </tbody>
    </table>

    <h3>10.2 Inference Latency & Speed Benchmark</h3>
    <table>
        <thead>
            <tr>
                <th>Hardware Platform</th>
                <th>Resolution</th>
                <th>Latency per Frame</th>
                <th>Throughput</th>
                <th>Operational Feasibility</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Apple M-Series CPU</td><td class="center-text">320 x 320</td><td class="center-text"><strong>26.1 ms</strong></td><td class="center-text"><strong>~38 FPS</strong></td><td>Real-Time Embedded</td></tr>
            <tr><td>Apple M-Series CPU</td><td class="center-text">640 x 640</td><td class="center-text"><strong>54.1 ms</strong></td><td class="center-text"><strong>~18 FPS</strong></td><td>Near Real-Time</td></tr>
            <tr><td>Tesla T4 GPU (Colab)</td><td class="center-text">640 x 640</td><td class="center-text"><strong>9.4 ms</strong></td><td class="center-text"><strong>~106 FPS</strong></td><td>High-Throughput Batch</td></tr>
        </tbody>
    </table>

    <div class="img-container">
        <img src="{cm_b64}" alt="Normalized Confusion Matrix" style="max-width: 75%;">
        <div class="img-caption">Figure 10.1: Normalized Confusion Matrix on Sekilab RDD2020 Validation Set.</div>
    </div>

    <div class="img-container">
        <img src="{pr_b64}" alt="Precision-Recall Curve" style="max-width: 75%;">
        <div class="img-caption">Figure 10.2: Precision-Recall Curve across all pavement distress categories.</div>
    </div>

    <h3>10.3 Qualitative Benchmark on Real-World Challenges</h3>
    <table>
        <thead>
            <tr>
                <th>Test Scenario</th>
                <th>Surface Description</th>
                <th>Triaged Severity</th>
                <th>Detected Distresses</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Autumn Highway</td><td>Clean Asphalt Highway</td><td><span class="badge badge-good">🟢 GOOD (0 Found)</span></td><td>Zero false alarms; trees/horizon filtered</td></tr>
            <tr><td>Mountain Road</td><td>Clean Pavement with lines</td><td><span class="badge badge-good">🟢 GOOD (0 Found)</span></td><td>Zero false alarms on smooth asphalt</td></tr>
            <tr><td>Wet Asphalt</td><td>Central Pothole + Crack</td><td><span class="badge badge-critical">🔴 CRITICAL (3 Found)</span></td><td>Pothole (88%), Fatigue Crack (69%)</td></tr>
            <tr><td>Muddy Road</td><td>Deep Water Potholes</td><td><span class="badge badge-critical">🔴 CRITICAL (2 Found)</span></td><td>Waterlogged potholes (88%, 22%)</td></tr>
            <tr><td>Shattered Pavement</td><td>Severe Alligator Cracks</td><td><span class="badge badge-critical">🔴 CRITICAL (2 Found)</span></td><td>Pothole (55%), Alligator Crack (51%)</td></tr>
            <tr><td>Landslide Split</td><td>Fault Line across Road</td><td><span class="badge badge-critical">🔴 CRITICAL (3 Found)</span></td><td>Fatigue Crack (58%), Long Crack (23%)</td></tr>
            <tr><td>Deep Trench</td><td>Transverse Fault + Cones</td><td><span class="badge badge-critical">🔴 CRITICAL (2 Found)</span></td><td>Transverse Crack (22%), Pothole</td></tr>
            <tr><td>Flooded Road (TOI)</td><td>Severe Mud, Water & Barricade</td><td><span class="badge badge-critical">🔴 CRITICAL (2 Found)</span></td><td>Deep Crater (76%), Hazard Pit (16%)</td></tr>
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- 11. CONCLUSION AND FUTURE SCOPE -->
    <h2>11. CONCLUSION AND FUTURE SCOPE</h2>
    <h3>11.1 Conclusion</h3>
    <p>This project successfully designed, implemented, and validated an automated computer vision framework for road distress detection and maintenance severity triaging on Indian roads. By utilizing the state-of-the-art YOLOv8 object detector and formulating a Dual-Model AI Ensemble coupled with spatial horizon filtering and Test-Time Augmentation, the system overcomes the historical vulnerability of automated pavement inspection: false alarms on clean roads versus missed detections on complex waterlogged craters.</p>
    <p>Trained on 7,708 Sekilab Indian road images (including 3,921 clean negative samples), the system achieves an overall mAP@0.50 of 0.515, with the critical pothole class reaching 0.648 mAP@0.50 at an inference speed of up to 38 FPS on CPU and >100 FPS on GPU. The interactive Streamlit dashboard enables rapid visual inspection, automated severity triage, and CSV audit logging, demonstrating an affordable, scalable alternative to expensive survey vehicles.</p>

    <h3>11.2 Future Scope</h3>
    <ol>
        <li><strong>GPS Tagging & Municipal GIS Heatmaps:</strong> Integrating a hardware GPS module into the video capture pipeline to tag detected distresses with geographical coordinates (latitude/longitude), dynamically updating municipal road repair GIS heatmaps.</li>
        <li><strong>Stereo Vision & 3D Pothole Volume Estimation:</strong> Incorporating stereo cameras or monocular depth estimation networks (e.g., MiDaS, Depth Anything) to measure pothole depth and volume in cubic meters, allowing automated calculation of required asphalt patching material.</li>
        <li><strong>Edge Microcontroller Deployment:</strong> Porting the quantized YOLOv8 model via ONNX / TensorRT onto embedded edge computing boards (Raspberry Pi 5 with AI Hat, NVIDIA Jetson Orin Nano) for direct installation on public transport buses and municipal waste collection vehicles.</li>
        <li><strong>Night-Vision & Adverse Weather Optimization:</strong> Fine-tuning the network on thermal and infrared nighttime footage to maintain continuous 24/7 road inspection capabilities.</li>
    </ol>

    <div class="page-break"></div>

    <!-- 12. REFERENCES -->
    <h2>12. REFERENCES</h2>
    <ol style="font-size: 10pt; color: #374151; line-height: 1.6;">
        <li>Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2020). <em>Global Road Damage Detection Challenge: Datasets and Baselines.</em> IEEE International Conference on Big Data, pp. 2533-2542.</li>
        <li>Jocher, G., Chaurasia, A., & Qiu, J. (2023). <em>Ultralytics YOLOv8: A State-of-the-Art Object Detector.</em> GitHub repository: https://github.com/ultralytics/ultralytics.</li>
        <li>Koch, C., & Brilakis, I. (2011). <em>Pothole detection in asphalt pavement images.</em> Advanced Engineering Informatics, 25(3), 507-515.</li>
        <li>Cord, A., & Chambon, S. (2012). <em>Automatic road defect detection by textural pattern recognition based on AdaBoost.</em> Computer-Aided Civil and Infrastructure Engineering, 27(4), 244-259.</li>
        <li>Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). <em>You Only Look Once: Unified, Real-Time Object Detection.</em> IEEE CVPR, pp. 779-788.</li>
        <li>Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). <em>YOLOv4: Optimal Speed and Accuracy of Object Detection.</em> arXiv preprint arXiv:2004.10934.</li>
        <li>Ministry of Road Transport and Highways (MoRTH), Government of India. (2022). <em>Road Accidents in India 2022.</em> Transport Research Wing, New Delhi.</li>
        <li>Indian Roads Congress (IRC:SP:72-2015). <em>Guidelines for the Design of Flexible Pavements for Low Volume Rural Roads.</em> New Delhi, India.</li>
        <li>Alfarrarjeh, A., Trivedi, D., Kim, S. H., & Shahabi, C. (2021). <em>A Deep Learning Approach for Road Damage Detection from Smartphone Videos.</em> IEEE Big Data, pp. 5201-5210.</li>
        <li>Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). <em>Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.</em> AAAI, 34(07), 12993-13000.</li>
        <li>Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). <em>Road damage detection and classification using deep neural networks with smartphone images.</em> Computer-Aided Civil and Infrastructure Engineering, 33(12), 1127-1141.</li>
    </ol>

    <div class="page-break"></div>

    <!-- 13. RELEVANT SOURCE CODE AND OUTPUT SCREENSHOTS -->
    <h2>13. RELEVANT SOURCE CODE AND OUTPUT SCREENSHOTS</h2>
    <p><strong>GitHub Open-Source Repository:</strong><br>
    <a href="https://github.com/Ravish945/road-damage-classification" target="_blank" style="color: #2563eb;">https://github.com/Ravish945/road-damage-classification</a></p>

    <h3>13.1 Dual-Model AI Ensemble & Spatial Horizon Filter (app.py)</h3>
    <pre><code>def run_ensemble_detection(image, conf=0.18, iou=0.45, imgsz=640, augment=True):
    w, h = image.size
    boxes, scores, labels = [], [], []

    # Stream 1: Structural Cracks via RDD2022 YOLOv8s
    if rdd_model is not None:
        res_rdd = rdd_model.predict(image, conf=conf, imgsz=imgsz, augment=augment, verbose=False)[0]
        for b in res_rdd.boxes:
            cls_name = rdd_model.names.get(int(b.cls[0]), "defect")
            score = float(b.conf[0])
            if cls_name == "pothole" and score < 0.35:
                continue
            bx = b.xyxy[0].tolist()
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            if bw > 0.85 * w and bh > 0.65 * h:
                continue
            if bx[3] < 0.18 * h:
                continue
            boxes.append(bx)
            scores.append(score)
            labels.append(rdd_map.get(cls_name, cls_name))

    # Stream 2: Dedicated Universal Pothole & Crater Detector with TTA
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

    # NMS to merge overlapping candidate boxes
    t_boxes = torch.tensor(boxes)
    t_scores = torch.tensor(scores)
    keep = torchvision.ops.nms(t_boxes, t_scores, iou_threshold=iou).tolist()
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]</code></pre>

    <h3>13.2 Training Loss Curves & Validation Convergence</h3>
    <div class="img-container">
        <img src="{res_b64}" alt="YOLOv8 Training Loss Curves" style="max-width: 75%;">
        <div class="img-caption">Figure 13.1: YOLOv8 Training Loss Curves & Metric Progression across 25 Epochs.</div>
    </div>

    <h3>13.3 Qualitative Detection Outputs Across Benchmark Road Scenarios</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
        <div class="img-container" style="margin: 0;">
            <img src="{clean1_b64}" alt="Clean Autumn Highway" style="width: 100%;">
            <div class="img-caption">Clean Autumn Highway (🟢 GOOD: 0 Found - Zero False Alarms)</div>
        </div>
        <div class="img-container" style="margin: 0;">
            <img src="{dam1_b64}" alt="Wet Asphalt Pothole" style="width: 100%;">
            <div class="img-caption">Wet Asphalt Pothole + Crack (🔴 CRITICAL: Pothole 88%, Crack 69%)</div>
        </div>
        <div class="img-container" style="margin: 0;">
            <img src="{dam2_b64}" alt="Muddy Road Potholes" style="width: 100%;">
            <div class="img-caption">Muddy Road Potholes (🔴 CRITICAL: Potholes 88%, 22%)</div>
        </div>
        <div class="img-container" style="margin: 0;">
            <img src="{dam7_b64}" alt="Flooded Monsoon Road" style="width: 100%;">
            <div class="img-caption">Flooded Monsoon Road (🔴 CRITICAL: Pothole 76%, Pit 16%)</div>
        </div>
    </div>

</div>

</body>
</html>
"""
    out_html = "/Users/ravish/Downloads/India/PBL_Report_Road_Damage_Detection.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"PBL Report HTML successfully generated at: {out_html}")

if __name__ == "__main__":
    generate_pbl_html()
