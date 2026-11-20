"""
app.py
Interactive Web Application for Road Damage Detection & Severity Assessment.

Author: Ravish Patel
Project: Road Damage Detection PBL (Indian Road Conditions)
Framework: Streamlit & Ultralytics YOLOv8
"""

import os
import glob
from PIL import Image
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Road Damage Detection - PBL",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-box {
        text-align: center;
        padding: 10px;
        background: #FFFFFF;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Friendly mapping
DAMAGE_INFO = {
    "D00_Longitudinal_Crack": {
        "title": "Longitudinal Crack (D00)",
        "desc": "Cracks running parallel to the direction of traffic flow, typically caused by pavement fatigue or joint deterioration.",
        "color": "#3B82F6"
    },
    "D10_Transverse_Crack": {
        "title": "Transverse Crack (D10)",
        "desc": "Cracks perpendicular to the road centerline, often caused by thermal shrinkage or underlying reflective cracks.",
        "color": "#10B981"
    },
    "D20_Alligator_Crack": {
        "title": "Alligator Crack (D20)",
        "desc": "Interconnected crack network resembling reptile skin, indicating severe structural failure in the road base.",
        "color": "#F59E0B"
    },
    "D40_Pothole": {
        "title": "Pothole (D40)",
        "desc": "Bowl-shaped depression in the pavement surface resulting from water ingress and traffic wear. Critical road hazard.",
        "color": "#EF4444"
    }
}


def load_model(weights_path):
    """Load YOLO model lazily with caching."""
    from ultralytics import YOLO
    if not os.path.exists(weights_path):
        # Fallback for preview/testing if model is not yet trained
        return YOLO("yolov8n.pt")
    return YOLO(weights_path)


def compute_severity(detection_counts):
    """Assess severity based on detected damages."""
    potholes = detection_counts.get("D40_Pothole", 0)
    alligators = detection_counts.get("D20_Alligator_Crack", 0)
    cracks = detection_counts.get("D00_Longitudinal_Crack", 0) + detection_counts.get("D10_Transverse_Crack", 0)

    if potholes > 0 or alligators >= 2:
        return "CRITICAL", "Immediate structural repair / pothole filling required to prevent accidents.", "#EF4444"
    elif alligators > 0 or cracks >= 2:
        return "MODERATE", "Surface patching and seal coating recommended before deterioration spreads.", "#F59E0B"
    elif cracks > 0:
        return "LOW", "Minor surface cracks detected. Periodic monitoring suggested.", "#3B82F6"
    else:
        return "GOOD", "Road surface is in good condition with no detected damages.", "#10B981"


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/color/96/000000/road.png", width=70)
st.sidebar.title("Configuration")

# Model path selection
default_weights = "models/best.pt"
if not os.path.exists(default_weights):
    weights_choice = st.sidebar.text_input("Model Weights Path", "models/best.pt")
    st.sidebar.info("Tip: After training via Google Colab, place your `best.pt` file into the `models/` directory.")
else:
    weights_choice = default_weights
    st.sidebar.success(f"Loaded trained model: `{weights_choice}`")

# Confidence and IoU sliders
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
iou_threshold = st.sidebar.slider("IoU NMS Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Project Details")
st.sidebar.markdown("**Student:** Ravish Patel")
st.sidebar.markdown("**Project:** Road Damage PBL")
st.sidebar.markdown("**Dataset:** RDD India (Crowdsourced)")
st.sidebar.markdown("**Architecture:** Ultralytics YOLOv8")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">Automated Road Damage Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Computer Vision system for Indian road distress identification and severity analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Image Inspection", "📊 Dataset & Damage Classes", "📘 Architecture & Methodology"])

with tab1:
    col_input, col_display = st.columns([1, 2])

    with col_input:
        st.subheader("Input Source")
        input_type = st.radio("Choose image source:", ["Select Demo Sample", "Upload Custom Image"])

        image_to_process = None

        if input_type == "Select Demo Sample":
            samples = sorted(glob.glob("demo_samples/*.jpg"))
            if samples:
                sample_names = [os.path.basename(s) for s in samples]
                selected_sample = st.selectbox("Select sample image:", sample_names)
                image_to_process = Image.open(os.path.join("demo_samples", selected_sample))
            else:
                st.warning("No images found in demo_samples directory.")
        else:
            uploaded_file = st.file_uploader("Upload road image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)

        run_btn = st.button("🚀 Analyze Road Surface", type="primary", use_container_width=True)

    with col_display:
        st.subheader("Analysis & Detection Results")
        if image_to_process is not None:
            if run_btn:
                with st.spinner("Processing image through YOLOv8 model..."):
                    try:
                        model = load_model(weights_choice)
                        # Run inference
                        results = model.predict(
                            source=image_to_process,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            save=False
                        )[0]

                        # Render detected image
                        annotated_array = results.plot()
                        annotated_img = Image.fromarray(annotated_array[..., ::-1]) # BGR to RGB

                        # Calculate stats
                        counts = {}
                        detection_details = []
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names.get(cls_id, f"Class_{cls_id}")
                            conf_val = float(box.conf[0])
                            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                            counts[cls_name] = counts.get(cls_name, 0) + 1
                            detection_details.append({
                                "Damage Type": DAMAGE_INFO.get(cls_name, {}).get("title", cls_name),
                                "Confidence": f"{conf_val * 100:.1f}%",
                                "Coordinates [xmin, ymin, xmax, ymax]": str(coords)
                            })

                        # Severity assessment
                        sev_status, sev_msg, sev_color = compute_severity(counts)

                        # Display before / after images
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(image_to_process, caption="Original Input Road Image", use_container_width=True)
                        with c2:
                            st.image(annotated_img, caption="Detected Distresses with Bounding Boxes", use_container_width=True)

                        # Display Severity Alert
                        st.markdown(f"""
                        <div style="background-color: {sev_color}15; border-left: 5px solid {sev_color}; padding: 12px; border-radius: 6px; margin: 15px 0;">
                            <h4 style="margin: 0; color: {sev_color};">Road Health Status: {sev_status}</h4>
                            <p style="margin: 5px 0 0 0; color: #374151;">{sev_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Detection breakdown
                        st.markdown("#### Detected Damages Breakdown")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Potholes (D40)", counts.get("D40_Pothole", 0))
                        m2.metric("Alligator Cracks (D20)", counts.get("D20_Alligator_Crack", 0))
                        m3.metric("Longitudinal (D00)", counts.get("D00_Longitudinal_Crack", 0))
                        m4.metric("Transverse (D10)", counts.get("D10_Transverse_Crack", 0))

                        # Table of detections
                        if detection_details:
                            st.markdown("#### Detailed Detection Coordinates")
                            st.dataframe(detection_details, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
            else:
                st.image(image_to_process, caption="Selected Image (Click 'Analyze Road Surface' to run)", use_container_width=True)
        else:
            st.info("Please select a demo sample image or upload a photo of a road surface to begin.")


with tab2:
    st.subheader("Road Damage Taxonomy (RDD India)")
    st.write("This project addresses the standard road distress classifications defined in the Global Road Damage Detection benchmark:")

    for code, info in DAMAGE_INFO.items():
        st.markdown(f"""
        <div style="background:#FFFFFF; border-left: 6px solid {info['color']}; padding: 12px 18px; margin-bottom: 12px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <strong style="font-size: 1.1rem; color: #111827;">{info['title']}</strong>
            <p style="margin: 4px 0 0 0; color: #4B5563; font-size: 0.95rem;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Summary (India Sub-dataset)")
    st.markdown("""
    - **Total Training Images:** 7,706 images
    - **Images with Labeled Damage:** 3,785 images
    - **Background Images (No Damage):** 3,921 images (used to minimize false positives)
    - **Annotation Standard:** Pascal VOC XML format converted into normalized YOLO coordinate space.
    - **Primary Damage Distribution:** Potholes (`D40`) and Alligator Cracks (`D20`) dominate Indian road conditions.
    """)


with tab3:
    st.subheader("System Architecture & Engineering Methodology")
    st.markdown("""
    ### 1. Computer Vision Pipeline
    1. **Data Ingestion:** High-resolution road surface captures from dashboard/smart phone cameras.
    2. **Format Conversion:** Pascal VOC XML bounding box parsing `(xmin, ymin, xmax, ymax)` converted to normalized relative coordinates `(x_center, y_center, w, h)`.
    3. **Data Augmentation:** Mosaic augmentation, random horizontal flips, and HSV color space variations during training to simulate diverse weather and lighting.
    4. **Feature Extraction:** YOLOv8 Backbone with CSPDarknet structure and Path Aggregation Network (PANet) neck for multi-scale feature fusion.
    5. **Loss Functions:**
       - **CIoU Loss:** Complete Intersection over Union for accurate bounding box localization.
       - **Distribution Focal Loss (DFL):** High precision boundary regression.
       - **BCE Loss:** Binary Cross-Entropy for class classification.
    6. **Inference & Post-processing:** Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.

    ### 2. Practical Application
    Mounted on public transit buses or municipal inspection vehicles, this system eliminates manual surveyor risk and delivers automated daily road health logs with GPS-ready coordinates.
    """)
