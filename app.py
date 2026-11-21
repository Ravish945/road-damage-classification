import os
import glob
import json
import csv
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Road Damage Detection & Triage",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("⚙️ Inspection Settings")
model_path = st.sidebar.text_input("Model Checkpoint", "models/best.pt")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.01, 0.80, 0.15, 0.01)
auto_sensitivity = st.sidebar.checkbox("Auto-Sensitivity (Smart Boost)", value=True, help="Automatically detects subtle defects and out-of-domain desktop/web photos if standard threshold yields 0.")
st.sidebar.caption("📌 Recommended: 0.15 for balanced detection across potholes & cracks.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Target Classes")
st.sidebar.markdown("""
- **D00**: Longitudinal Crack
- **D10**: Transverse Crack
- **D20**: Alligator (Fatigue) Crack
- **D40**: Pothole / Rutting
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** PBL - Indian Road Distress AI  \n**Author:** Ravish Patel")

# Load model with caching
@st.cache_resource
def load_yolo(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_yolo(model_path)
if model is None:
    st.error(f"Model file `{model_path}` not found. Please train first or place weights in models/.")
    st.stop()


def smart_detect(yolo_model, img, base_conf=0.15, enable_boost=True):
    """
    Runs multi-tier inference:
    1. Evaluates at user's base confidence threshold.
    2. If 0 detections and boost is enabled, steps down adaptively with multi-scale TTA
       to catch fine cracks and out-of-domain web/macro photos.
    """
    res = yolo_model.predict(img, conf=base_conf, imgsz=640, verbose=False)[0]
    if len(res.boxes) > 0 or not enable_boost:
        return res, False, base_conf

    # Adaptive fallback levels
    for fallback_conf in [0.08, 0.04, 0.015]:
        res_fallback = yolo_model.predict(img, conf=fallback_conf, imgsz=640, augment=True, verbose=False)[0]
        if len(res_fallback.boxes) > 0:
            return res_fallback, True, fallback_conf

    return res, False, base_conf


# Header
st.markdown('<div class="main-header">🛣️ Road Damage Detection & Severity Triage</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated computer vision inspection on Indian roads using YOLOv8 with real-time maintenance priority classification</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_inspect, tab_metrics, tab_logs = st.tabs(["🔍 Inspection Studio", "📊 Model Performance & Curves", "📋 Inspection Log"])

with tab_inspect:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Select or Upload Image")
        source = st.radio("Input Source", ["Demo Gallery (Categorized)", "Upload Custom Photo (Desktop/Web)"], horizontal=True, label_visibility="collapsed")

        image = None
        img_name = "test.jpg"

        if "Demo Gallery" in source:
            catalog_file = "demo_samples/catalog.json"
            catalog = {}
            if os.path.exists(catalog_file):
                with open(catalog_file) as f:
                    catalog = json.load(f)

            cat_options = ["All Verified Samples"] + list(catalog.keys())
            chosen_cat = st.selectbox("Damage Category Filter:", cat_options)

            if chosen_cat == "All Verified Samples" or chosen_cat not in catalog:
                file_pool = sorted(glob.glob("demo_samples/*.jpg"))
            else:
                file_pool = [os.path.join("demo_samples", f) for f in catalog[chosen_cat] if os.path.exists(os.path.join("demo_samples", f))]

            if file_pool:
                options = [os.path.basename(f) for f in file_pool]
                default_idx = options.index("India_000389.jpg") if "India_000389.jpg" in options else 0
                chosen = st.selectbox("Pick a road image:", options, index=default_idx)
                image_path = os.path.join("demo_samples", chosen)
                image = Image.open(image_path).convert("RGB")
                img_name = chosen
            else:
                st.warning("No sample images in this category.")
        else:
            uploaded = st.file_uploader("Upload road image from your computer", type=["jpg", "jpeg", "png", "webp"])
            st.caption("💡 Supports all formats. Auto-Sensitivity is enabled by default to boost detection on desktop and web-downloaded photos.")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                img_name = uploaded.name

        if image is not None:
            st.image(image, caption=f"Input: {img_name}", use_container_width=True)
            run_btn = st.button("🚀 Run AI Road Inspection", type="primary")
        else:
            run_btn = False

    with col_output:
        st.subheader("2. AI Analysis & Severity Triage")

        if image is not None and run_btn:
            with st.spinner("Analyzing road surface..."):
                t_start = time.time()
                results, is_adaptive, effective_conf = smart_detect(
                    model,
                    image,
                    base_conf=conf_thresh,
                    enable_boost=auto_sensitivity
                )
                latency_ms = (time.time() - t_start) * 1000

                # Render annotated bounding boxes
                annotated = Image.fromarray(results.plot()[..., ::-1])
                st.image(annotated, caption="YOLOv8 Detection Output", use_container_width=True)

                # Show adaptive notification if triggered
                if is_adaptive:
                    st.info(f"⚡ **Auto-Sensitivity Active:** Detected defects using adaptive threshold ({effective_conf*100:.1f}%) with multi-scale enhancement. Ideal for desktop & web-downloaded photos.")

                # Aggregate detection statistics
                counts = {}
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    name = model.names.get(cls_id, f"Class_{cls_id}")
                    counts[name] = counts.get(name, 0) + 1

                potholes = counts.get("D40_Pothole", 0)
                alligators = counts.get("D20_Alligator_Crack", 0)
                cracks = counts.get("D00_Longitudinal_Crack", 0) + counts.get("D10_Transverse_Crack", 0)
                total_defects = sum(counts.values())

                # Quick Metric Cards
                m1, m2, m3 = st.columns(3)
                with m1:
                    if potholes > 0 or alligators >= 2:
                        st.error(f"🔴 CRITICAL\n{potholes} Pothole(s), {alligators} Fatigue")
                    elif counts:
                        st.warning(f"🟡 MODERATE\n{total_defects} Surface Crack(s)")
                    else:
                        st.success("🟢 GOOD\nNo Defects")

                with m2:
                    st.metric("Total Distresses", f"{total_defects} found")

                with m3:
                    st.metric("Inference Latency", f"{latency_ms:.1f} ms", f"{1000/max(1, latency_ms):.0f} FPS")

                # Actionable Maintenance Recommendation
                if potholes > 0 or alligators >= 2:
                    st.error(
                        f"**Urgent Maintenance Required:** {potholes} pothole(s) and {alligators} alligator crack(s) detected. "
                        "High risk of vehicle tire blowout or suspension damage. Immediate asphalt patching team dispatch recommended."
                    )
                elif counts:
                    st.warning(
                        "**Preventive Maintenance Required:** Surface cracks detected. "
                        "Schedule joint sealing and crack filling before the monsoon season to prevent water percolation into the sub-base."
                    )
                else:
                    st.success("**Pavement in Good Condition:** No structural distresses detected above threshold. Road is within standard safety limits.")

                # Breakdown Table
                if counts:
                    st.markdown("#### Detailed Breakdown:")
                    b_cols = st.columns(len(counts))
                    for i, (k, v) in enumerate(counts.items()):
                        with b_cols[i]:
                            st.info(f"**{k}**: {v}")

                # Save to inspection CSV log
                os.makedirs("results", exist_ok=True)
                with open("results/inspection_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        img_name,
                        f"{latency_ms:.1f}ms",
                        total_defects,
                        potholes,
                        dict(counts)
                    ])
        else:
            st.info("👈 Select or upload an image and click **Run AI Road Inspection** to see detections, bounding boxes, and maintenance severity triage.")


with tab_metrics:
    st.subheader("Model Validation & Training Benchmark")
    st.write("Quantitative results trained for 25 epochs on Google Colab (Tesla T4 GPU) on Sekilab RDD2020 India Dataset.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        cm_path = "results/screenshots/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Validation Confusion Matrix", use_container_width=True)
    with col_m2:
        pr_path = "results/screenshots/BoxPR_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Curve (mAP@0.50 = 0.515)", use_container_width=True)

    res_path = "results/screenshots/results.png"
    if os.path.exists(res_path):
        st.image(res_path, caption="Training Loss Curves & Convergence", use_container_width=True)


with tab_logs:
    st.subheader("Inspection History Log")
    log_file = "results/inspection_log.csv"
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        try:
            df = pd.read_csv(log_file, names=["Timestamp", "Image Name", "Latency", "Total Defects", "Potholes", "Details"])
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "📥 Download Inspection Log (CSV)",
                data=open(log_file).read(),
                file_name="road_inspection_log.csv",
                mime="text/csv"
            )
        except Exception:
            st.write("Could not parse inspection log.")
    else:
        st.write("No inspections logged yet. Run a detection in the **Inspection Studio** tab to populate the log.")
