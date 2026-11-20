import os
import glob
import csv
from datetime import datetime
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️")

st.title("Road Damage Detection & Severity Triage")
st.write("Upload a road photo or pick a sample image to detect potholes and cracks using YOLOv8.")

# Sidebar settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model weights", "models/best.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

# Load model with proper Streamlit caching
@st.cache_resource
def get_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = get_model(model_path)
if model is None:
    st.error(f"Could not find model weights at `{model_path}`. Please run `python train.py` first.")
    st.stop()

# Input source: upload or sample
source_type = st.radio("Choose input:", ["Demo Sample", "Upload Image"], horizontal=True)

image = None
image_name = "test.jpg"

if source_type == "Demo Sample":
    sample_files = sorted(glob.glob("demo_samples/*.jpg"))
    if sample_files:
        chosen = st.selectbox("Select sample image:", [os.path.basename(f) for f in sample_files])
        image_path = os.path.join("demo_samples", chosen)
        image = Image.open(image_path)
        image_name = chosen
    else:
        st.warning("No images found in demo_samples folder.")
else:
    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        image_name = uploaded.name

# Run detection
if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection")
        if st.button("Run Detection", type="primary"):
            with st.spinner("Detecting..."):
                results = model.predict(image, conf=conf_thresh, save=False)[0]
                annotated = Image.fromarray(results.plot()[..., ::-1])
                st.image(annotated, use_container_width=True)

                # Count detections per class
                counts = {}
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    name = model.names.get(cls_id, f"Class {cls_id}")
                    counts[name] = counts.get(name, 0) + 1

                # Severity logic based on defects
                potholes = counts.get("D40_Pothole", 0)
                alligators = counts.get("D20_Alligator_Crack", 0)

                if potholes > 0 or alligators >= 2:
                    st.error(f"Severity: CRITICAL ({potholes} potholes, {alligators} alligator cracks) - immediate repair recommended")
                elif counts:
                    st.warning("Severity: MODERATE - surface cracks detected, schedule for resealing")
                else:
                    st.success("Severity: GOOD - no road damages detected")

                # Show breakdown
                if counts:
                    st.write("**Detected damages:**")
                    for k, v in counts.items():
                        st.write(f"- {k}: {v}")

                    # Append to inspection log
                    os.makedirs("results", exist_ok=True)
                    with open("results/inspection_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, counts])
