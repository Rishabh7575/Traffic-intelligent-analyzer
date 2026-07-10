from __future__ import division

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from util.parser import load_classes
from util.streamlit_pipeline import (
    recommend_signal,
    resolve_path,
    run_inference_on_image,
    run_inference_on_video,
    load_model,
    summarize_lanes,
)


st.set_page_config(page_title="Intelligent Traffic Management System", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("config/yolov3.cfg", "weights/yolov3.weights", model_resolution="416")


@st.cache_resource(show_spinner=False)
def get_classes():
    return load_classes(str(resolve_path("data/idd.names")))


def load_image_from_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def save_temp_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


st.title("Intelligent Traffic Management System")
st.caption("YOLOv3-based vehicle detection with lane-wise statistics and signal recommendation.")

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.01)
    nms_thresh = st.slider("NMS threshold", 0.05, 0.95, 0.30, 0.01)
    input_mode = st.radio("Input type", ["Image", "Video"], index=0)


try:
    model = get_model()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

classes = get_classes()

upload_label = "Upload an image" if input_mode == "Image" else "Upload a video"
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"], accept_multiple_files=False)

# Try and Catch block is used to handle exceptions that may occur during the loading of the model. If a RuntimeError occurs, an error message is displayed and the application stops execution.
if uploaded_file:
    if input_mode == "Image":
        try:
            image = load_image_from_upload(uploaded_file)
            if image is None:
                st.error("Unable to read the uploaded image.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()
            status.write("Running vehicle detection...")
            annotated, detections, vehicle_count, lane_counts, class_counts = run_inference_on_image(
                image, model, classes, confidence, nms_thresh
            )
            progress.progress(100)

            signal = recommend_signal(lane_counts)

            left, right = st.columns([1.2, 1])
            with left:
                st.subheader("Processed Image")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
                _, buffer = cv2.imencode(".jpg", annotated)
                st.download_button(
                    label="Download processed image",
                    data=buffer.tobytes(),
                    file_name=f"{Path(uploaded_file.name).stem}_annotated.jpg",
                    mime="image/jpeg",
                )

            with right:
                st.subheader("Statistics")
                st.metric("Total vehicles", vehicle_count)
                st.metric("Recommended lane", signal["lane"] if signal["lane"] is not None else "N/A")
                st.metric("Recommended open time", f"{signal['seconds']} sec")
                st.write(signal["message"])
                st.write("Lane-wise counts")
                st.table(summarize_lanes(lane_counts))
                st.write("Vehicle classes")
                st.table([{"class": key, "count": int(value)} for key, value in class_counts.items()] or [{"class": "None", "count": 0}])
        except Exception as error:
            st.error(f"Image detection failed: {error}")

    else:
        temp_video = save_temp_upload(uploaded_file)
        try:
            progress = st.progress(0)
            status = st.empty()
            status.write("Running vehicle detection on video...")

            def update_progress(value):
                progress.progress(min(max(value, 0.0), 1.0))

            output_video_path, vehicle_count, lane_counts, class_counts = run_inference_on_video(
                temp_video,
                model,
                classes,
                confidence,
                nms_thresh,
                progress_callback=update_progress,
            )

            signal = recommend_signal(lane_counts)
            video_bytes = Path(output_video_path).read_bytes()
            progress.progress(100)

            left, right = st.columns([1.2, 1])
            with left:
                st.subheader("Processed Video")
                st.video(video_bytes)
                st.download_button(
                    label="Download processed video",
                    data=video_bytes,
                    file_name=Path(output_video_path).name,
                    mime="video/mp4",
                )
# statistical exaggeration will be don
            with right:
                st.subheader("Statistics")
                st.metric("Total vehicles", vehicle_count)
                st.metric("Recommended lane", signal["lane"] if signal["lane"] is not None else "N/A")
                st.metric("Recommended open time", f"{signal['seconds']} sec")
                st.write(signal["message"])
                st.write("Lane-wise counts")
                st.table(summarize_lanes(lane_counts))
                st.write("Vehicle classes")
                st.table([{"class": key, "count": int(value)} for key, value in class_counts.items()] or [{"class": "None", "count": 0}])
        except Exception as error:
            st.error(f"Video detection failed: {error}")
        finally:
            if temp_video.exists():
                temp_video.unlink(missing_ok=True)

else:
    st.info("Upload an image or video to start detection.")