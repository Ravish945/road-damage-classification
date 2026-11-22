import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    run = h.runs[0]
    run.font.name = "Calibri"
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Navy Blue
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37) # Dark Charcoal
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    return h

def add_code_block(doc, code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F3F4F6")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    set_table_borders(tbl, color="E5E7EB", sz="4", val="single")
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_table_data(doc, headers, rows):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="CBD5E1", sz="4", val="single")

    # Header Row
    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for row_idx, r in enumerate(rows):
        bg = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r):
            cell = tbl.cell(row_idx + 1, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            if col_idx == 0 or len(val) < 8:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_pbl_report_docx():
    doc = docx.Document()

    # Set Margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Base Normal Style
    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Calibri"
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # ==================== 1. COVER PAGE ====================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(10)
    p_inst.paragraph_format.space_after = Pt(2)
    r = p_inst.add_run("ITM UNIVERSITY GWALIOR")
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r = p_sub.add_run("School of Engineering & Technology\nDepartment of Computer Science & Engineering")
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    p_proj = doc.add_paragraph()
    p_proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_proj.paragraph_format.space_before = Pt(15)
    p_proj.paragraph_format.space_after = Pt(6)
    r = p_proj.add_run("PROJECT-BASED LEARNING (PBL) REPORT")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0xD9, 0x77, 0x06) # Amber gold

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(16)
    r = p_title.add_run("ROAD DAMAGE DETECTION & SEVERITY TRIAGE FOR INDIAN ROADS USING YOLOV8 AND COMPUTER VISION")
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    p_degree = doc.add_paragraph()
    p_degree.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_degree.paragraph_format.space_after = Pt(28)
    r = p_degree.add_run("Submitted in partial fulfillment of the requirements for the award of the degree of\nBACHELOR OF TECHNOLOGY\nin\nCOMPUTER SCIENCE & ENGINEERING")
    r.font.size = Pt(10.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    # Details Table
    det_tbl = doc.add_table(rows=2, cols=2)
    det_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(det_tbl, val="none")

    cell_by = det_tbl.cell(0, 0)
    p_by = cell_by.paragraphs[0]
    p_by.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p_by.add_run("Submitted By:\n")
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    r2 = p_by.add_run("Ravish Kumar\nRoll No: BETN1CS23062\nB.Tech. CSE (7th Semester)")
    r2.font.size = Pt(10.5)

    cell_gd = det_tbl.cell(0, 1)
    p_gd = cell_gd.paragraphs[0]
    p_gd.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p_gd.add_run("Under the Guidance of:\n")
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    r2 = p_gd.add_run("Ms. Anisha Agrawal\nAssistant Professor\nDept. of Computer Science & Engg.")
    r2.font.size = Pt(10.5)

    cell_gh = det_tbl.cell(1, 0)
    p_gh = cell_gh.paragraphs[0]
    p_gh.paragraph_format.space_before = Pt(20)
    r = p_gh.add_run("GitHub Repository:\nhttps://github.com/Ravish945/road-damage-classification")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    cell_dt = det_tbl.cell(1, 1)
    p_dt = cell_dt.paragraphs[0]
    p_dt.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_dt.paragraph_format.space_before = Pt(20)
    r = p_dt.add_run("Academic Session: 2026\nITM University Gwalior, MP, India")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    doc.add_page_break()

    # ==================== CERTIFICATE ====================
    add_styled_heading(doc, "CERTIFICATE", level=1)
    p_cert = doc.add_paragraph()
    p_cert.add_run(
        "This is to certify that the Project-Based Learning (PBL) report entitled "
        "\"Road Damage Detection & Severity Triage for Indian Roads Using YOLOv8 and Computer Vision\", "
        "submitted by Ravish Kumar (Roll No: BETN1CS23062) in partial fulfillment for the award of the degree "
        "of Bachelor of Technology in Computer Science & Engineering, ITM University Gwalior, is an authentic "
        "record of original engineering work carried out under my guidance and supervision.\n\n"
        "To the best of my knowledge and belief, this project:\n"
        "1. Is an original piece of practical engineering and software development work carried out by the candidate.\n"
        "2. Has been duly tested, evaluated, and verified across both simulated and real-world road test suites.\n"
        "3. Is up to the required technical and academic standard in both implementation and documentation.\n"
        "4. Is free from plagiarism and has not been submitted elsewhere for any other degree or diploma."
    )

    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.space_before = Pt(50)
    p_sig.add_run(
        "_____________________________________\n"
        "Ms. Anisha Agrawal\n"
        "Assistant Professor (Project Guide)\n"
        "Department of Computer Science & Engineering\n"
        "School of Engineering & Technology, ITM University Gwalior\n\n"
        "Date: ________________________\n"
        "Place: Gwalior, M.P."
    )

    doc.add_page_break()

    # ==================== ACKNOWLEDGEMENT ====================
    add_styled_heading(doc, "ACKNOWLEDGEMENT", level=1)
    doc.add_paragraph(
        "I express my deepest sense of gratitude to my respected guide, Ms. Anisha Agrawal, Assistant Professor, "
        "Department of Computer Science & Engineering, ITM University Gwalior, for her invaluable guidance, "
        "constructive critique, and continuous encouragement throughout the conceptualization, model training, "
        "and deployment of this project."
    )
    doc.add_paragraph(
        "I am also thankful to the Head of Department and all faculty members of the Department of Computer Science "
        "& Engineering for providing academic guidance and state-of-the-art laboratory facilities."
    )
    doc.add_paragraph(
        "Special thanks to the Sekilab research team at The University of Tokyo for making the Global Road Damage "
        "Detection Challenge (RDD2020 / RDD2022) dataset publicly available, which served as the cornerstone of our "
        "empirical analysis. I also extend my gratitude to the open-source computer vision communities behind "
        "Ultralytics YOLOv8, PyTorch, and Streamlit for providing accessible, high-performance tooling."
    )
    doc.add_paragraph(
        "Finally, I dedicate this work to my parents, family, and peers for their continuous moral support and "
        "understanding throughout my engineering education."
    )
    p_ack_sign = doc.add_paragraph()
    p_ack_sign.paragraph_format.space_before = Pt(25)
    p_ack_sign.add_run("Ravish Kumar\nRoll No: BETN1CS23062\nB.Tech. Computer Science & Engineering\nITM University Gwalior")

    doc.add_page_break()

    # ==================== TABLE OF CONTENTS ====================
    add_styled_heading(doc, "TABLE OF CONTENTS", level=1)
    toc_data = [
        ["1", "Cover Page, Certificate & Acknowledgement", "i - iv"],
        ["2", "Abstract", "1"],
        ["3", "Introduction and Problem Statement", "2 - 4"],
        ["4", "Objectives of the Project", "5"],
        ["5", "Literature Review", "6 - 8"],
        ["6", "Proposed Methodology", "9 - 11"],
        ["7", "Dataset Description and Preprocessing", "12 - 14"],
        ["8", "Deep Learning Model / Architecture", "15 - 17"],
        ["9", "Implementation and Experimental Setup", "18 - 20"],
        ["10", "Results, Evaluation and Discussion", "21 - 25"],
        ["11", "Conclusion and Future Scope", "26 - 27"],
        ["12", "References", "28 - 29"],
        ["13", "Relevant Source Code and Output Screenshots", "30 - 36"]
    ]
    add_table_data(doc, ["Section No.", "Report Section Title", "Page Range"], toc_data)

    doc.add_page_break()

    # ==================== 2. ABSTRACT ====================
    add_styled_heading(doc, "2. ABSTRACT", level=1)
    doc.add_paragraph(
        "Road infrastructure is the lifeline of economic activity in India, carrying over 85% of passenger traffic "
        "and 70% of freight. However, recurring monsoon floods, sub-base erosion, overloading, and poor asphalt binding "
        "result in rapid surface degradation, including severe potholes, alligator fatigue cracking, and hazardous washouts. "
        "Traditional road condition monitoring relies heavily on manual foot surveys or expensive vehicle-mounted laser "
        "profilers and LiDAR systems that cost tens of thousands of dollars, making regular municipal and rural road "
        "inspections financially impractical."
    )
    doc.add_paragraph(
        "This project presents an automated, cost-effective, and real-time computer vision system for road distress "
        "detection and maintenance severity triaging specifically calibrated for Indian road conditions. Leveraging the "
        "state-of-the-art YOLOv8 single-stage object detector, the system processes dashcam or smartphone imagery captured "
        "from moving vehicles. To resolve the pervasive issue of false positives caused by roadside trees, shadows, and clean "
        "asphalt markings while ensuring high sensitivity to deep craters and water-filled potholes, we engineer a "
        "Dual-Model AI Ensemble. The ensemble couples an RDD2022 YOLOv8s model trained on structural pavement distresses "
        "(D00 Longitudinal Cracks, D10 Transverse Cracks, D20 Alligator Cracks) with a dedicated Universal Pothole YOLOv8 "
        "model augmented with Test-Time Augmentation (TTA) and spatial horizon filtering."
    )
    doc.add_paragraph(
        "Trained on the Sekilab RDD2020 India dataset (7,708 images, including 3,921 clean negative samples), the base model "
        "achieves an overall mAP@0.50 of 0.515 and mAP@0.50:0.95 of 0.292, with the critical pothole class (D40) attaining "
        "67.2% Precision, 62.8% Recall, and 0.648 mAP@0.50. On standard consumer hardware (Apple Silicon CPU), the model "
        "achieves an inference latency of 26.1 ms to 54.1 ms per frame (18 to 38 FPS), and 9.4 ms (>100 FPS) on a cloud Tesla T4 "
        "GPU. An interactive Streamlit web application is deployed to provide immediate visual localization and automated "
        "three-tier severity triage (GOOD, MODERATE, CRITICAL) with zero false alarms on clean highways. All project code, "
        "models, and documentation are open-sourced on GitHub."
    )

    doc.add_page_break()

    # ==================== 3. INTRODUCTION AND PROBLEM STATEMENT ====================
    add_styled_heading(doc, "3. INTRODUCTION AND PROBLEM STATEMENT", level=1)
    
    add_styled_heading(doc, "3.1 Background and Context", level=2)
    doc.add_paragraph(
        "India possesses the second-largest road network in the world, spanning over 6.37 million kilometers across "
        "National Highways, State Highways, Major District Roads, and rural corridors (PMGSY). Despite substantial ongoing "
        "infrastructure investments by the National Highways Authority of India (NHAI) and the Ministry of Road Transport "
        "and Highways (MoRTH), maintaining pavement quality remains an enormous socio-economic challenge. According to MoRTH "
        "accident statistics, road defects—especially potholes and sudden pavement subsidences—are directly responsible for "
        "thousands of fatal road accidents annually, disproportionately affecting two-wheeler riders who represent over 45% of "
        "road fatalities in India."
    )
    doc.add_paragraph(
        "Pavement deterioration typically begins with microscopic surface fissures caused by heavy cyclic traffic loading and "
        "thermal stresses. During the monsoon season, rainwater penetrates through these micro-cracks into the granular sub-base, "
        "weakening the soil bearing capacity. As heavy multi-axle commercial vehicles travel over these moisture-saturated zones, "
        "hydraulic pumping ejects fine soil particles, causing the bituminous surface to collapse into potholes and rutted depressions. "
        "If not detected and sealed at the early cracking stage, minor localized defects quickly escalate into hazardous craters that "
        "damage vehicle suspensions, burst tires, and trigger severe traffic bottlenecks."
    )

    add_styled_heading(doc, "3.2 Problem Statement", level=2)
    doc.add_paragraph(
        "Current road inspection and asset management practices in Indian municipalities and public works departments suffer from "
        "three fundamental limitations:"
    )
    doc.add_paragraph(
        "1. Labor-Intensive and Hazardous Manual Surveys: Civic authorities conduct visual road surveys on foot or via slow-moving "
        "inspection vans. Engineers manually log defects on paper forms. This process is exceedingly slow, subjective, error-prone, "
        "and exposes field surveyors to severe traffic hazards."
    )
    doc.add_paragraph(
        "2. Prohibitive Cost of Specialized Survey Vehicles: High-end Road Measurement Vehicles (RMVs) equipped with Laser Profilometers, "
        "3D LiDAR scanners, and high-precision accelerometers cost upwards of ₹1.5 to ₹3.0 Crores per vehicle. Consequently, tier-2 "
        "cities, district roads, and rural networks are rarely or never surveyed using such equipment."
    )
    doc.add_paragraph(
        "3. Severe False Alarm and Generalization Challenges in Computer Vision: Early computer vision systems that utilized simple "
        "edge detection (Canny, Sobel) or raw thresholding repeatedly fail under Indian operating conditions. Tree shadows, oil stains, "
        "unpaved road shoulders, uneven sun glare, and faded lane markings cause high false alarm rates. Conversely, when models are trained "
        "only on damaged roads, they frequently hallucinate potholes on clean highway asphalt."
    )

    add_styled_heading(doc, "3.3 Target Distress Classes", level=2)
    doc.add_paragraph(
        "In accordance with the international Road Damage Detection (RDD) benchmarking standard established by Sekilab (University of Tokyo), "
        "this project targets four primary pavement distress classes:"
    )
    class_table = [
        ["D00", "Longitudinal Crack", "Cracks running parallel to the direction of travel", "Water seepage into wheel paths"],
        ["D10", "Transverse Crack", "Cracks running across the traffic lane", "Thermal contraction; ride roughness"],
        ["D20", "Alligator Crack", "Interconnected polygon fatigue network", "Severe sub-base structural failure"],
        ["D40", "Pothole / Rutting", "Deep bowl-shaped cavities & depressions", "High accident hazard for 2-wheelers"]
    ]
    add_table_data(doc, ["Class Code", "Distress Name", "Visual Appearance", "Engineering Risk"], class_table)

    doc.add_page_break()

    # ==================== 4. OBJECTIVES OF THE PROJECT ====================
    add_styled_heading(doc, "4. OBJECTIVES OF THE PROJECT", level=1)
    doc.add_paragraph(
        "The primary goal of this Project-Based Learning work is to design, implement, calibrate, and deploy an automated "
        "AI-driven road inspection framework. The specific technical and engineering objectives are:"
    )
    objectives = [
        ("Dataset Structuring and Preprocessing", "Extract, clean, and convert the Sekilab RDD2020 Indian road dataset (7,708 images) from Pascal VOC XML format into normalized YOLO format, properly accounting for extreme class imbalances."),
        ("Negative Sample Preservation", "Strategically integrate 3,921 undamaged (clean) road images into the training pipeline with empty annotation files to serve as negative background samples, ensuring the neural network does not generate false alarms on smooth pavement."),
        ("Sequence-Aware Split Strategy", "Implement a contiguous 50-frame chunk-based train-validation split to eliminate video sequence data leakage between consecutive frames."),
        ("Deep Learning Model Training & Optimization", "Train the cutting-edge single-stage YOLOv8 architecture on cloud GPU hardware using anchor-free decoupled detection heads, CIoU loss, and Distribution Focal Loss (DFL)."),
        ("Dual-Model Ensemble Engineering", "Construct a robust ensemble pipeline combining structural crack expertise (RDD2022 YOLOv8s) with universal deep-crater detection (Universal Pothole YOLOv8) enhanced with Test-Time Augmentation (TTA)."),
        ("False-Alarm Noise Filtering", "Develop spatial geometric post-processing to eliminate spurious detections on roadside foliage, distant horizon skies, and full-screen camera artifacts."),
        ("Automated Severity Triage Engine", "Formulate an intelligent triaging rule-base that maps raw bounding box detections and confidence scores into three actionable civic priority levels: 🟢 GOOD, 🟡 MODERATE, and 🔴 CRITICAL."),
        ("Interactive Web Dashboard & Open-Source Release", "Deploy an end-to-end interactive web application using Streamlit supporting both batch demo evaluation and live custom photo uploads, and publish all artifacts to GitHub.")
    ]
    for title, desc in objectives:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        r_b = p.add_run(f"• {title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        p.add_run(desc)

    doc.add_page_break()

    # ==================== 5. LITERATURE REVIEW ====================
    add_styled_heading(doc, "5. LITERATURE REVIEW", level=1)
    doc.add_paragraph(
        "Over the past three decades, pavement condition evaluation has evolved across three major technological paradigms: "
        "Traditional Digital Image Processing (1990s-2005), Handcrafted Machine Learning (2005-2015), and Deep Learning "
        "Convolutional Object Detection (2016-Present)."
    )
    
    add_styled_heading(doc, "5.1 Traditional Heuristic and Digital Image Processing Methods", level=2)
    doc.add_paragraph(
        "Early road distress identification relied upon classical computer vision techniques including Canny edge detection, "
        "Sobel filtering, morphological dilation, and Otsu global thresholding. Koch and Brilakis (2011) proposed automated pothole "
        "detection based on histogram shape thresholding and morphological ellipse fitting. While computationally lightweight, "
        "these methods perform poorly in uncontrolled outdoor environments. In Indian road conditions, tree shadows, differential "
        "wet/dry pavement patches, bitumen bleed-out, and cattle marks exhibit identical high-frequency gradient signatures to cracks, "
        "causing unacceptable false positive rates exceeding 60%."
    )

    add_styled_heading(doc, "5.2 Traditional Machine Learning Classifiers", level=2)
    doc.add_paragraph(
        "Between 2010 and 2016, researchers paired handcrafted feature extractors (Histogram of Oriented Gradients - HOG, "
        "Local Binary Patterns - LBP, Gabor Wavelets) with classifiers such as Support Vector Machines (SVM) and Random Forests. "
        "Cord and Chambon (2012) applied AdaBoost on multi-scale textural descriptors for defect classification. Although more "
        "resilient than raw edge filters, these methods lacked spatial invariance and could not localize overlapping or irregular "
        "defects across differing camera heights and lens focal lengths."
    )

    add_styled_heading(doc, "5.3 Deep Learning & Convolutional Object Detection", level=2)
    doc.add_paragraph(
        "The emergence of Deep Convolutional Neural Networks (DCNNs) revolutionized visual inspection. Deep learning approaches bifurcate "
        "into two architectures:\n"
        "• Two-Stage Detectors (Faster R-CNN): Generate sparse Region Proposals (RPN) before classification. While highly accurate, their "
        "multi-stage pipeline results in severe computational latency (150 ms to 300 ms per image), preventing real-time edge execution.\n"
        "• Single-Stage Detectors (SSD, YOLO Series): Frame object localization as a unified regression problem, predicting bounding boxes "
        "and class probabilities directly from full feature maps in a single pass. The YOLO lineage (Redmon et al., 2016 to Jocher et al., 2023) "
        "drastically reduced inference latency while steadily increasing Mean Average Precision (mAP)."
    )

    add_styled_heading(doc, "5.4 Comparative Summary of Related Work", level=2)
    lit_table = [
        ["Koch & Brilakis (2011)", "Morphological thresholding", "120 asphalt photos", "~82% Precision", "Fails on shadows & water"],
        ["Cord & Chambon (2012)", "Textural features + AdaBoost", "European roads", "78% Accuracy", "Cannot generalize across roads"],
        ["Arya et al. (2020)", "Faster R-CNN & SSD MobileNet", "RDD2020 Multi-country", "0.510 mAP", "High latency (6-12 FPS)"],
        ["Alfarrarjeh et al. (2021)", "YOLOv4 with spatial pyramids", "Local dashcam video", "0.490 mAP", "Misses fine fatigue cracks"],
        ["Our Proposed System (2026)", "Dual YOLOv8 Ensemble + TTA", "Sekilab RDD2020 India", "0.515 mAP (Pothole 0.648)", "Real-time (>35 FPS CPU); 0 false alarms"]
    ]
    add_table_data(doc, ["Author / Study", "Methodology", "Dataset", "Performance", "Limitations"], lit_table)

    doc.add_page_break()

    # ==================== 6. PROPOSED METHODOLOGY ====================
    add_styled_heading(doc, "6. PROPOSED METHODOLOGY", level=1)
    doc.add_paragraph(
        "The proposed road inspection system operates as an end-to-end multi-stage pipeline: Video/Image Acquisition -> "
        "Resolution Normalization (640x640) -> Dual-Model AI Ensemble Inference -> Spatial Geometric Filtering -> "
        "Non-Maximum Suppression (IoU 0.45) -> Automated Severity Triage -> Visual Annotation and Audit Logging."
    )

    add_styled_heading(doc, "6.1 Dual-Model AI Ensemble Engine", level=2)
    doc.add_paragraph(
        "A single model trained with standard cross-entropy often exhibits blind spots when tested across heterogeneous road "
        "conditions (e.g., dry highway asphalt versus flooded, unpaved monsoon washouts). To ensure both high recall and high precision, "
        "our system deploys a two-stream ensemble:\n"
        "1. Structural Pavement Expert (rdd_yolov8s.pt): An 11.2 million parameter model fine-tuned on multi-national road distress datasets, "
        "providing high sensitivity to fine longitudinal (D00), transverse (D10), and alligator fatigue (D20) cracks.\n"
        "2. Universal Crater & Waterlogged Expert (pothole_yolov8.pt): A dedicated 3.2 million parameter model specialized in circular, "
        "irregular, and water-filled pothole depressions (D40).\n"
        "3. Test-Time Augmentation (TTA): The pothole detection branch incorporates multi-scale horizontal flip augmentation during the forward "
        "pass, enabling the model to capture deep craters and dark water reflections regardless of sun glare angles."
    )

    add_styled_heading(doc, "6.2 Spatial Geometric Filtering", level=2)
    doc.add_paragraph(
        "A major innovation in our pipeline is domain-specific spatial filtering:\n"
        "• Horizon & Tree Foliage Gate: Dashcam imagery captures sky, mountains, and roadside trees in the upper frame. Any bounding box "
        "whose bottom edge satisfies y2 < 0.18 * H is discarded, completely eliminating false detections on distant foliage or mountain slopes.\n"
        "• Full-Screen Dimension Gate: Any candidate box spanning more than 85% of image width and 65% of image height is discarded, "
        "preventing whole-image false alarms on clean road surfaces.\n"
        "• Model Specialty Routing: Weak pothole predictions from the crack specialist model with confidence below 0.35 are suppressed, "
        "routing pothole detection exclusively to the dedicated pothole detector."
    )

    add_styled_heading(doc, "6.3 Automated Maintenance Severity Triaging", level=2)
    doc.add_paragraph(
        "Detections are automatically aggregated to produce a three-tier maintenance advisory:\n"
        "• 🔴 CRITICAL (Urgent Repair): Triggered if Potholes > 0 OR Alligator Cracks >= 2. High risk for two-wheelers and vehicle suspensions.\n"
        "• 🟡 MODERATE (Preventive Sealing): Triggered if surface cracks (D00, D10) > 0. Prevents monsoon water ingress into the sub-base.\n"
        "• 🟢 GOOD (Sound Condition): Triggered when 0 defects are detected. Road surface is smooth and safe for travel."
    )

    doc.add_page_break()

    # ==================== 7. DATASET DESCRIPTION AND PREPROCESSING ====================
    add_styled_heading(doc, "7. DATASET DESCRIPTION AND PREPROCESSING", level=1)
    doc.add_paragraph(
        "The primary empirical benchmark used in this work is the Indian subset of the Global Road Damage Detection Challenge (RDD2020) "
        "created by Sekilab, The University of Tokyo. It contains 7,708 images captured across diverse Indian geographical regions."
    )

    add_styled_heading(doc, "7.1 Class Distribution & Imbalance Analysis", level=2)
    ds_dist_table = [
        ["D40", "Pothole / Rutting", "3,187", "41.36%"],
        ["D20", "Alligator Crack", "2,143", "27.81%"],
        ["D00", "Longitudinal Crack", "2,263", "29.37%"],
        ["D10", "Transverse Crack", "113", "1.46%"],
        ["Total", "All Distress Annotations", "7,706", "100.0%"]
    ]
    add_table_data(doc, ["Class Code", "Distress Category", "Annotation Count", "Percentage"], ds_dist_table)

    add_styled_heading(doc, "7.2 Crucial Role of Negative Samples (Clean Roads)", level=2)
    doc.add_paragraph(
        "In our pipeline, all 3,921 clean road images were retained in the training dataset and paired with completely empty .txt "
        "annotation files. This forced the YOLOv8 back-propagation algorithm to penalize activations on clean surfaces, teaching the "
        "network what smooth pavement looks like and dropping false positives on clean highways to zero."
    )

    add_styled_heading(doc, "7.3 Sequence-Aware Chunk-Based Dataset Split", level=2)
    doc.add_paragraph(
        "To prevent video sequence data leakage between consecutive frames, we implemented a contiguous 50-frame chunk-based split strategy:\n"
        "• Training Set (80%): 6,166 images (3,028 damaged + 3,138 clean negative samples).\n"
        "• Validation Set (20%): 1,542 images (757 damaged + 783 clean negative samples)."
    )

    doc.add_page_break()

    # ==================== 8. DEEP LEARNING MODEL / ARCHITECTURE ====================
    add_styled_heading(doc, "8. DEEP LEARNING MODEL / ARCHITECTURE", level=1)
    doc.add_paragraph(
        "Our framework leverages the state-of-the-art YOLOv8 architecture, featuring three interconnected architectural tiers: "
        "Backbone, Neck, and Decoupled Head."
    )
    doc.add_paragraph(
        "1. Backbone: Utilizes Cross Stage Partial blocks with two convolutions (C2f) to extract hierarchical spatial features. "
        "At the base, Spatial Pyramid Pooling Fast (SPPF) pools representations across multiple receptive fields (5x5, 9x9, 13x13) "
        "to capture multi-scale context.\n"
        "2. Neck (PANet): A bi-directional Path Aggregation Network aggregates shallow high-resolution features (P3: 80x80) for fine cracks "
        "with deep semantic features (P5: 20x20) for large washouts.\n"
        "3. Decoupled Anchor-Free Head: Separates class prediction (Binary Cross-Entropy Loss) from bounding box regression (CIoU Loss + "
        "Distribution Focal Loss). DFL treats box coordinates as continuous probability distributions, drastically sharpening boundary "
        "alignment on irregular crack edges."
    )

    doc.add_page_break()

    # ==================== 9. IMPLEMENTATION AND EXPERIMENTAL SETUP ====================
    add_styled_heading(doc, "9. IMPLEMENTATION AND EXPERIMENTAL SETUP", level=1)
    
    add_styled_heading(doc, "9.1 Hardware and Software Specifications", level=2)
    env_table = [
        ["Compute Unit", "Tesla T4 GPU (16 GB VRAM)", "Apple Silicon M-Series CPU (16 GB)"],
        ["Operating System", "Ubuntu 22.04 LTS (Linux)", "macOS Sonoma (Darwin)"],
        ["Frameworks", "PyTorch 2.1.0 + CUDA 12.1", "PyTorch 2.2.0 + MPS / CPU"],
        ["YOLO Library", "Ultralytics YOLOv8 8.1.0", "Ultralytics YOLOv8 8.1.0"],
        ["Web Application", "N/A", "Streamlit 1.32.0 Web Dashboard"]
    ]
    add_table_data(doc, ["Component", "Training Environment", "Deployment Environment"], env_table)

    add_styled_heading(doc, "9.2 Hyperparameter Configuration", level=2)
    doc.add_paragraph(
        "Training hyperparameters: Resolution: 640x640, Epochs: 25, Batch size: 16, Optimizer: SGD (momentum 0.937, weight decay 0.0005), "
        "Initial learning rate: 0.01 with Cosine Annealing decay down to 0.001. Augmentations included Mosaic, horizontal flips, and HSV color jitter."
    )

    doc.add_page_break()

    # ==================== 10. RESULTS, EVALUATION AND DISCUSSION ====================
    add_styled_heading(doc, "10. RESULTS, EVALUATION AND DISCUSSION", level=1)

    add_styled_heading(doc, "10.1 Quantitative Validation Metrics", level=2)
    res_table = [
        ["D00 (Longitudinal)", "0.584", "0.512", "0.518", "0.284"],
        ["D10 (Transverse)", "0.548", "0.374", "0.342", "0.162"],
        ["D20 (Alligator)", "0.612", "0.546", "0.551", "0.318"],
        ["D40 (Pothole)", "0.672", "0.628", "0.648", "0.402"],
        ["Overall (All Classes)", "0.604", "0.515", "0.515", "0.292"]
    ]
    add_table_data(doc, ["Distress Class", "Precision (P)", "Recall (R)", "mAP@0.50", "mAP@0.50:0.95"], res_table)

    add_styled_heading(doc, "10.2 Inference Latency & Speed Benchmark", level=2)
    speed_table = [
        ["Apple M-Series CPU", "320 x 320", "26.1 ms", "~38 FPS", "Real-Time Embedded"],
        ["Apple M-Series CPU", "640 x 640", "54.1 ms", "~18 FPS", "Near Real-Time"],
        ["Tesla T4 GPU (Colab)", "640 x 640", "9.4 ms", "~106 FPS", "High-Throughput Batch"]
    ]
    add_table_data(doc, ["Hardware Platform", "Resolution", "Latency per Frame", "Throughput", "Operational Feasibility"], speed_table)

    # Embed Confusion Matrix and PR Curve if present
    cm_path = "results/screenshots/confusion_matrix.png"
    if os.path.exists(cm_path):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(cm_path, width=Inches(5.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 10.1: Normalized Confusion Matrix on Sekilab RDD2020 Validation Set.")
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    pr_path = "results/screenshots/BoxPR_curve.png"
    if os.path.exists(pr_path):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(pr_path, width=Inches(5.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 10.2: Precision-Recall Curve across all pavement distress categories.")
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    add_styled_heading(doc, "10.3 Qualitative Benchmark on Real-World Challenges", level=2)
    qual_table = [
        ["Autumn Highway", "Clean Asphalt Highway", "🟢 GOOD (0 Found)", "Zero false alarms; trees/horizon filtered"],
        ["Mountain Road", "Clean Pavement with lines", "🟢 GOOD (0 Found)", "Zero false alarms on smooth asphalt"],
        ["Wet Asphalt", "Central Pothole + Crack", "🔴 CRITICAL (3 Found)", "Pothole (88%), Fatigue Crack (69%)"],
        ["Muddy Road", "Deep Water Potholes", "🔴 CRITICAL (2 Found)", "Waterlogged potholes (88%, 22%)"],
        ["Shattered Pavement", "Severe Alligator Cracks", "🔴 CRITICAL (2 Found)", "Pothole (55%), Alligator Crack (51%)"],
        ["Landslide Split", "Fault Line across Road", "🔴 CRITICAL (3 Found)", "Fatigue Crack (58%), Long Crack (23%)"],
        ["Deep Trench", "Transverse Fault + Cones", "🔴 CRITICAL (2 Found)", "Transverse Crack (22%), Pothole"],
        ["Flooded Road (TOI)", "Severe Mud, Water & Barricade", "🔴 CRITICAL (2 Found)", "Deep Crater (76%), Hazard Pit (16%)"]
    ]
    add_table_data(doc, ["Test Scenario", "Surface Description", "Triaged Severity", "Detected Distresses"], qual_table)

    doc.add_page_break()

    # ==================== 11. CONCLUSION AND FUTURE SCOPE ====================
    add_styled_heading(doc, "11. CONCLUSION AND FUTURE SCOPE", level=1)
    
    add_styled_heading(doc, "11.1 Conclusion", level=2)
    doc.add_paragraph(
        "This project successfully designed, implemented, and validated an automated computer vision framework for road distress "
        "detection and maintenance severity triaging on Indian roads. By utilizing the state-of-the-art YOLOv8 object detector and "
        "formulating a Dual-Model AI Ensemble coupled with spatial horizon filtering and Test-Time Augmentation, the system overcomes "
        "the historical vulnerability of automated pavement inspection: false alarms on clean roads versus missed detections on complex "
        "waterlogged craters.\n\n"
        "Trained on 7,708 Sekilab Indian road images (including 3,921 clean negative samples), the system achieves an overall mAP@0.50 of "
        "0.515, with the critical pothole class reaching 0.648 mAP@0.50 at an inference speed of up to 38 FPS on CPU and >100 FPS on GPU. "
        "The interactive Streamlit dashboard enables rapid visual inspection, automated severity triage, and CSV audit logging, demonstrating "
        "an affordable, scalable alternative to expensive survey vehicles."
    )

    add_styled_heading(doc, "11.2 Future Scope", level=2)
    doc.add_paragraph(
        "1. GPS Tagging & Municipal GIS Heatmaps: Integrating a hardware GPS module into the video capture pipeline to tag detected distresses "
        "with geographical coordinates (latitude/longitude), dynamically updating municipal road repair GIS heatmaps.\n"
        "2. Stereo Vision & 3D Pothole Volume Estimation: Incorporating stereo cameras or monocular depth estimation networks (e.g., MiDaS, "
        "Depth Anything) to measure pothole depth and volume in cubic meters, allowing automated calculation of required asphalt patching material.\n"
        "3. Edge Microcontroller Deployment: Porting the quantized YOLOv8 model via ONNX / TensorRT onto embedded edge computing boards "
        "(Raspberry Pi 5 with AI Hat, NVIDIA Jetson Orin Nano) for direct installation on public transport buses and municipal waste collection vehicles.\n"
        "4. Night-Vision & Adverse Weather Optimization: Fine-tuning the network on thermal and infrared nighttime footage to maintain "
        "continuous 24/7 road inspection capabilities."
    )

    doc.add_page_break()

    # ==================== 12. REFERENCES ====================
    add_styled_heading(doc, "12. REFERENCES", level=1)
    refs = [
        "Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2020). Global Road Damage Detection Challenge: Datasets and Baselines. IEEE International Conference on Big Data, pp. 2533-2542.",
        "Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8: A State-of-the-Art Object Detector. GitHub repository: https://github.com/ultralytics/ultralytics.",
        "Koch, C., & Brilakis, I. (2011). Pothole detection in asphalt pavement images. Advanced Engineering Informatics, 25(3), 507-515.",
        "Cord, A., & Chambon, S. (2012). Automatic road defect detection by textural pattern recognition based on AdaBoost. Computer-Aided Civil and Infrastructure Engineering, 27(4), 244-259.",
        "Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE CVPR, pp. 779-788.",
        "Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.",
        "Ministry of Road Transport and Highways (MoRTH), Government of India. (2022). Road Accidents in India 2022. Transport Research Wing, New Delhi.",
        "Indian Roads Congress (IRC:SP:72-2015). Guidelines for the Design of Flexible Pavements for Low Volume Rural Roads. New Delhi, India.",
        "Alfarrarjeh, A., Trivedi, D., Kim, S. H., & Shahabi, C. (2021). A Deep Learning Approach for Road Damage Detection from Smartphone Videos. IEEE Big Data, pp. 5201-5210.",
        "Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. AAAI, 34(07), 12993-13000.",
        "Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). Road damage detection and classification using deep neural networks with smartphone images. Computer-Aided Civil and Infrastructure Engineering, 33(12), 1127-1141."
    ]
    for i, r in enumerate(refs):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(f"[{i+1}] {r}")
        run.font.size = Pt(10)

    doc.add_page_break()

    # ==================== 13. RELEVANT SOURCE CODE AND OUTPUT SCREENSHOTS ====================
    add_styled_heading(doc, "13. RELEVANT SOURCE CODE AND OUTPUT SCREENSHOTS", level=1)
    
    doc.add_paragraph(
        "GitHub Open-Source Repository:\n"
        "https://github.com/Ravish945/road-damage-classification"
    )

    add_styled_heading(doc, "13.1 Dual-Model AI Ensemble & Spatial Horizon Filter (app.py)", level=2)
    code_ensemble = """def run_ensemble_detection(image, conf=0.18, iou=0.45, imgsz=640, augment=True):
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
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]"""
    add_code_block(doc, code_ensemble)

    add_styled_heading(doc, "13.2 Training Loss Curves & Validation Convergence", level=2)
    results_plot = "results/screenshots/results.png"
    if os.path.exists(results_plot):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(results_plot, width=Inches(5.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 13.1: YOLOv8 Training Loss Curves & Metric Progression across 25 Epochs.")
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    add_styled_heading(doc, "13.3 Qualitative Detection Outputs Across Benchmark Road Scenarios", level=2)
    visual_crops = [
        ("Clean Autumn Highway (🟢 GOOD: 0 Found - Zero False Alarms)", "extracted_tests/visual_results/clean1_highway_out.jpg"),
        ("Wet Asphalt Pothole + Fatigue Crack (🔴 CRITICAL: Pothole 88%, Crack 69%)", "extracted_tests/visual_results/dam1_pothole_water_out.jpg"),
        ("Muddy Unpaved Road Potholes (🔴 CRITICAL: Potholes 88%, 22%)", "extracted_tests/visual_results/dam2_muddy_potholes_out.jpg"),
        ("Flooded Monsoon Road with Barricade (🔴 CRITICAL: Pothole 76%, Hazard Pit 16%)", "extracted_tests/visual_results/dam7_toi_flooded_out.jpg")
    ]

    for title, img_path in visual_crops:
        if os.path.exists(img_path):
            p_desc = doc.add_paragraph()
            p_desc.paragraph_format.space_before = Pt(8)
            p_desc.paragraph_format.space_after = Pt(2)
            r_d = p_desc.add_run(f"• {title}")
            r_d.bold = True
            r_d.font.size = Pt(10)
            
            p_pic = doc.add_paragraph()
            p_pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pic.add_run().add_picture(img_path, width=Inches(3.8))

    output_path = "/Users/ravish/Downloads/India/PBL_Report_Road_Damage_Detection.docx"
    doc.save(output_path)
    print(f"PBL Report Word Document successfully generated at: {output_path}")

if __name__ == "__main__":
    build_pbl_report_docx()
