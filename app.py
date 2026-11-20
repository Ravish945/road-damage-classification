"""
Streamlit Web Dashboard for Road Damage Detection.
Provides visual demonstration of defect localization, triage severity assessment,
and inspection logging for municipal road maintenance.
"""

import os
import glob
import csv
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Standard damage taxonomy and contextual information
DAMAGE_INFO = {
    "D00_Longitudinal_Crack": {
        "title": "Longitudinal Crack (D00)",
        "desc": "Cracks parallel to traffic flow along wheel paths or joints. Caused by repeated wheel load stress or asphalt joint fatigue.",
        "color": "#3B82F6"
    },
    "D10_Transverse_Crack": {
        "title": "Transverse Crack (D10)",
        "desc": "Cracks perpendicular to the road centerline. Commonly caused by thermal contraction and reflective sub-base cracking.",
        "color": "#10B981"
    },
    "D20_Alligator_Crack": {
        "title": "Alligator Crack (D20)",
        "desc": "Interconnected polygon fatigue cracks resembling reptile skin. Indicates structural weakness in the road foundation.",
        "color": "#F59E0B"
    },
    "D40_Pothole": {
        "title": "Pothole (D40)",
        "desc": "Deep structural cavities caused by water seepage and heavy traffic wear. Immediate hazard for two-wheelers and passenger vehicles.",
        "color": "#EF4444"
    }
}


def load_model(weights_path):
    """
    Loads custom-trained YOLO weights.
    Raises FileNotFoundError if weights do not exist (no silent fallback to COCO weights).
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at '{weights_path}'. Train the model first using train.py "
            f"or verify your checkpoint path."
        )
    from ultralytics import YOLO
    return YOLO(weights_path)


def compute_severity(detection_counts):
    """
    Heuristic triage index aligned with IRC:82-2015 pavement maintenance guidelines:
    - CRITICAL: Potholes (D40) or multiple alligator fatigue cracks (D20). Immediate repair required.
    - MODERATE: Isolated alligator crack or multiple longitudinal/transverse cracks. Requires crack sealing.
    - LOW: Minor single cracks. Scheduled for periodic monitoring.
    - GOOD: No distress detected.
    """
    potholes = detection_counts.get("D40_Pothole", 0)
    alligators = detection_counts.get("D20_Alligator_Crack", 0)
    cracks = detection_counts.get("D00_Longitudinal_Crack", 0) + detection_counts.get("D10_Transverse_Crack", 0)

    if potholes > 0 or alligators >= 2:
        return "CRITICAL", "Immediate patching / pothole filling required to prevent accidents and water infiltration.", "#EF4444"
    elif alligators > 0 or cracks >= 2:
        return "MODERATE", "Surface crack sealing recommended before water ingress deteriorates the sub-base.", "#F59E0B"
    elif cracks > 0:
        return "LOW", "Minor surface cracks detected. Include in periodic road monitoring schedule.", "#3B82F6"
    else:
        return "GOOD", "Road surface is in acceptable condition with no significant distress.", "#10B981"


def log_inspection_event(image_name, counts, severity_status):
    """Appends inspection result to a local CSV audit log."""
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inspection_log.csv")
    file_exists = os.path.exists(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Image_Name", "Potholes_D40", "Alligator_D20", "Longitudinal_D00", "Transverse_D10", "Severity_Status"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            counts.get("D40_Pothole", 0),
            counts.get("D20_Alligator_Crack", 0),
            counts.get("D00_Longitudinal_Crack", 0),
            counts.get("D10_Transverse_Crack", 0),
            severity_status
        ])


# --- SIDEBAR CONTROLS ---
st.sidebar.title("Model Settings")

default_weights = "models/best.pt"
weights_choice = st.sidebar.text_input("Model Weights Path", default_weights)

weights_exist = os.path.exists(weights_choice)
if weights_exist:
    st.sidebar.success(f"Loaded: `{weights_choice}`")
else:
    st.sidebar.error(f"Weights file not found at `{weights_choice}`.")
    st.sidebar.info("Run `python train.py` to generate `models/best.pt`.")

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
iou_threshold = st.sidebar.slider("IoU NMS Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown("- **Task:** Road Damage Detection & Classification")
st.sidebar.markdown("- **Dataset:** RDD India (Crowdsourced)")
st.sidebar.markdown("- **Framework:** Ultralytics YOLOv8 & PyTorch")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">Road Damage Detection & Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Computer vision pipeline for automated defect localization and maintenance triage</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Image Inspection", "📋 Inspection Audit Log", "📊 Damage Taxonomy", "📘 Engineering Methodology"])

with tab1:
    col_input, col_display = st.columns([1, 2])

    with col_input:
        st.subheader("Input Image")
        input_type = st.radio("Source:", ["Select Demo Sample", "Upload Custom Image"])

        image_to_process = None
        selected_name = "custom_upload.jpg"

        if input_type == "Select Demo Sample":
            samples = sorted(glob.glob("demo_samples/*.jpg"))
            if samples:
                sample_names = [os.path.basename(s) for s in samples]
                selected_sample = st.selectbox("Choose demo image:", sample_names)
                selected_name = selected_sample
                image_to_process = Image.open(os.path.join("demo_samples", selected_sample))
            else:
                st.warning("No images found in demo_samples directory.")
        else:
            uploaded_file = st.file_uploader("Upload road photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_name = uploaded_file.name
                image_to_process = Image.open(uploaded_file)

        run_btn = st.button("Run Damage Analysis", type="primary", use_container_width=True, disabled=not weights_exist)
        if not weights_exist:
            st.warning("Analysis disabled until valid model weights are provided.")

    with col_display:
        st.subheader("Analysis & Detection")
        if image_to_process is not None:
            if run_btn and weights_exist:
                with st.spinner("Executing YOLOv8 forward pass..."):
                    try:
                        model = load_model(weights_choice)
                        results = model.predict(
                            source=image_to_process,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            save=False
                        )[0]

                        annotated_array = results.plot()
                        annotated_img = Image.fromarray(annotated_array[..., ::-1])

                        counts = {}
                        detection_details = []
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names.get(cls_id, f"Class_{cls_id}")
                            conf_val = float(box.conf[0])
                            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                            counts[cls_name] = counts.get(cls_name, 0) + 1
                            detection_details.append({
                                "Distress Class": DAMAGE_INFO.get(cls_name, {}).get("title", cls_name),
                                "Confidence": f"{conf_val * 100:.1f}%",
                                "Bounding Box [xmin, ymin, xmax, ymax]": str(coords)
                            })

                        sev_status, sev_msg, sev_color = compute_severity(counts)

                        # Log event to CSV
                        log_inspection_event(selected_name, counts, sev_status)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(image_to_process, caption=f"Input: {selected_name}", use_container_width=True)
                        with c2:
                            st.image(annotated_img, caption="Predicted Defects (Bounding Boxes)", use_container_width=True)

                        st.markdown(f"""
                        <div style="background-color: {sev_color}15; border-left: 5px solid {sev_color}; padding: 12px; border-radius: 6px; margin: 12px 0;">
                            <h4 style="margin: 0; color: {sev_color};">Triage Status: {sev_status}</h4>
                            <p style="margin: 4px 0 0 0; color: #374151;">{sev_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### Distress Breakdown")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Potholes (D40)", counts.get("D40_Pothole", 0))
                        m2.metric("Alligator Cracks (D20)", counts.get("D20_Alligator_Crack", 0))
                        m3.metric("Longitudinal (D00)", counts.get("D00_Longitudinal_Crack", 0))
                        m4.metric("Transverse (D10)", counts.get("D10_Transverse_Crack", 0))

                        if detection_details:
                            st.markdown("#### Detected Coordinates")
                            st.dataframe(pd.DataFrame(detection_details), use_container_width=True)
                        else:
                            st.info("No road distress detected above the confidence threshold.")

                    except Exception as e:
                        st.error(f"Inference error: {str(e)}")
            else:
                st.image(image_to_process, caption=f"Selected: {selected_name}", use_container_width=True)
        else:
            st.info("Select a demo sample or upload a road surface photo to begin inspection.")


with tab2:
    st.subheader("Inspection Audit Log")
    log_file = "results/inspection_log.csv"
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.dataframe(df_log.tail(20), use_container_width=True)
        csv_data = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Audit Log (CSV)",
            data=csv_data,
            file_name="road_inspection_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No inspections logged yet. Run an analysis on an image to generate audit entries.")


with tab3:
    st.subheader("Distress Classification Standards")
    st.write("Classes defined in the Global Road Damage Detection Challenge (RDD2020/2022):")

    for code, info in DAMAGE_INFO.items():
        st.markdown(f"""
        <div style="background:#FFFFFF; border-left: 5px solid {info['color']}; padding: 10px 16px; margin-bottom: 10px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">
            <strong style="color: #111827;">{info['title']}</strong>
            <p style="margin: 3px 0 0 0; color: #4B5563; font-size: 0.92rem;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)


with tab4:
    st.subheader("Architecture & Implementation Details")
    st.markdown("""
    ### Technical Overview
    - **Detection Engine:** YOLOv8n (Anchor-free single-stage detector).
    - **Input Normalization:** 640×640 or 320×320 dynamic resolution with letterboxing.
    - **Bounding Box Loss:** CIoU (Complete Intersection over Union) + DFL (Distribution Focal Loss).
    - **Negative Sampling:** Clean road images are included without labels to minimize false alarms on shadows and manholes.
    - **Localization Output:** Pixel coordinates `[xmin, ymin, xmax, ymax]`. For municipal operations, integration with hardware GPS receivers over serial NMEA streams allows attaching latitude/longitude to the CSV audit log.
    """)
