from __future__ import division

import hashlib
import sys
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from .image_processor import preparing_image
from .model import Darknet
from .parser import load_classes
from .utils import non_max_suppression


BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_URLS = [
    "https://data.pjreddie.com/files/yolov3.weights",
    "https://sourceforge.net/projects/yolov3.mirror/files/v8/yolov3.weights/download",
]
EXPECTED_WEIGHTS_SIZE = 248007048
EXPECTED_WEIGHTS_SHA256 = "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
VEHICLE_CLASSES = {"car", "motorbike", "truck", "bicycle", "autorickshaw"}


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights_file(weights_path):
    if not weights_path.exists():
        return False, "missing"
    actual_size = weights_path.stat().st_size
    if actual_size != EXPECTED_WEIGHTS_SIZE:
        return False, f"expected {EXPECTED_WEIGHTS_SIZE} bytes, found {actual_size} bytes"
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256 != EXPECTED_WEIGHTS_SHA256:
        return False, "sha256 mismatch"
    return True, "ok"


def format_progress_bar(current, total, width=30):
    if not total:
        return f"{current / (1024 * 1024):.1f} MB"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total) * 100
    return f"[{bar}] {percent:6.2f}%"


def download_with_progress(url, destination_path, progress_callback=None):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, open(destination_path, "wb") as file_handle:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size and total_size.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total_size)


def ensure_weights_file(weights_path, status_callback=None, progress_callback=None):
    weights_path = Path(weights_path)
    is_valid, reason = verify_weights_file(weights_path)
    if is_valid:
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        weights_path.unlink()

    temp_path = weights_path.with_suffix(weights_path.suffix + ".part")
    last_error = reason

    for url in WEIGHTS_URLS:
        try:
            if status_callback:
                status_callback(f"Downloading yolov3.weights from {url}")
            if temp_path.exists():
                temp_path.unlink()
            download_with_progress(url, temp_path, progress_callback)
            temp_path.replace(weights_path)
            is_valid, reason = verify_weights_file(weights_path)
            if is_valid:
                return weights_path
            last_error = reason
            if weights_path.exists():
                weights_path.unlink()
        except (HTTPError, URLError, OSError, ValueError) as error:
            last_error = str(error)
            if temp_path.exists():
                temp_path.unlink()

    manual_urls = "\n".join(f"- {url}" for url in WEIGHTS_URLS)
    raise RuntimeError(
        "Failed to download valid yolov3.weights.\n"
        f"Last error: {last_error}\n"
        "Manual download URLs:\n"
        f"{manual_urls}\n"
        "Place the file at: weights/yolov3.weights"
    )


def load_model(cfg_path, weights_path, model_resolution="416"):
    model = Darknet(str(resolve_path(cfg_path)))
    model.hyperparams["height"] = model_resolution
    ensure_weights_file(resolve_path(weights_path))
    model.load_weights(str(resolve_path(weights_path)))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def decode_detection_tensor(prediction, classes):
    detections = []
    if prediction is None or isinstance(prediction, int):
        return detections

    for row in prediction:
        class_id = int(row[-1])
        label = classes[class_id]
        detections.append(
            {
                "label": label,
                "confidence": float(row[5]),
                "bbox": [float(row[1]), float(row[2]), float(row[3]), float(row[4])],
            }
        )
    return detections


def lane_index_from_bbox(bbox, frame_width, lane_count=4):
    center_x = (bbox[0] + bbox[2]) / 2.0
    lane_width = frame_width / float(lane_count)
    lane_index = int(center_x // lane_width) + 1
    return max(1, min(lane_count, lane_index))


def run_inference_on_image(image_bgr, model, classes, confidence, nms_thresh, input_size=416):
    if image_bgr is None:
        raise ValueError("Invalid image provided")

    original_height, original_width = image_bgr.shape[:2]
    batch = preparing_image(image_bgr, input_size)
    if torch.cuda.is_available():
        batch = batch.cuda()

    with torch.no_grad():
        prediction = model(Variable(batch))

    prediction = non_max_suppression(prediction, confidence, len(classes), nms_conf=nms_thresh)
    detections = decode_detection_tensor(prediction, classes)
    annotated = image_bgr.copy()

    vehicle_count = 0
    lane_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    class_counts = Counter()

    if isinstance(prediction, int) or prediction is None:
        return annotated, detections, vehicle_count, lane_counts, class_counts

    for detection in prediction:
        class_id = int(detection[-1])
        label = classes[class_id]
        if label not in VEHICLE_CLASSES:
            continue

        vehicle_count += 1
        class_counts[label] += 1
        bbox = detection[1:5].tolist()
        lane_index = lane_index_from_bbox(bbox, original_width)
        lane_counts[lane_index] += 1

        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} {float(detection[5]):.2f}"
        cv2.putText(annotated, label_text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated, detections, vehicle_count, lane_counts, class_counts


def summarize_lanes(lane_counts):
    summary = []
    for lane_id in sorted(lane_counts):
        summary.append({"lane": lane_id, "count": int(lane_counts[lane_id])})
    return summary


def recommend_signal(lane_counts):
    lane_counts_list = [count for count in lane_counts.values() if count > 0]
    if not lane_counts_list:
        return {"lane": None, "seconds": 20, "message": "No vehicles detected. Keep default timing."}

    recommended_lane = max(lane_counts, key=lane_counts.get)
    average_count = sum(lane_counts_list) / len(lane_counts_list)
    if average_count > 50:
        seconds = 75 if int(max(lane_counts_list)) > 75 else int(max(lane_counts_list)) + 20
    elif average_count > 20:
        seconds = 55
    else:
        seconds = 20

    return {
        "lane": int(recommended_lane),
        "seconds": int(seconds),
        "message": f"Open lane {recommended_lane} for approximately {seconds} seconds.",
    }


def run_inference_on_video(video_path, model, classes, confidence, nms_thresh, input_size=416, progress_callback=None):
    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open video file")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_path = Path(video_path).with_name(Path(video_path).stem + "_annotated.mp4")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    total_vehicle_count = 0
    lane_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    class_counts = Counter()
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated, detections, vehicle_count, frame_lane_counts, frame_class_counts = run_inference_on_image(
                frame, model, classes, confidence, nms_thresh, input_size=input_size
            )
            total_vehicle_count += vehicle_count
            for lane_id, count in frame_lane_counts.items():
                lane_counts[lane_id] += count
            class_counts.update(frame_class_counts)
            writer.write(annotated)
            frame_index += 1
            if progress_callback and total_frames:
                progress_callback(frame_index / total_frames)
    finally:
        capture.release()
        writer.release()

    return output_path, total_vehicle_count, lane_counts, class_counts