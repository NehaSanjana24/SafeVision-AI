"""Image, video, and camera-assisted PPE screening."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.detector import infer_frame, read_dataset_classes
from core.ui import get_class_names, get_model, render_detection_controls, require_page_auth, save_scan_result
from core.violations import evaluate_violations, is_negative_ppe_label
from core.video_utils import process_video

require_page_auth()
st.title("Visual Processing and Safety Detection")
st.caption("Review PPE detections, keep annotated evidence, and record each scan.")

try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.info("Add custom weights at `models/best.pt` or allow the default `yolov8n.pt` model to download on first use.")
    st.stop()

class_names = get_class_names(model)
confidence, iou, required_classes = render_detection_controls(class_names)
negative_mode = any(is_negative_ppe_label(name) for name in class_names)
unsupported_classes = [name for name in read_dataset_classes() if name not in class_names]
if negative_mode:
    st.info("Negative PPE labels are present in the dataset. Detections of these classes will be treated as violations.")
elif not required_classes:
    st.warning("Select at least one required PPE class in the sidebar to evaluate compliance.")

image_tab, video_tab, webcam_tab = st.tabs(["Image Upload", "Video Upload", "Webcam"])

with image_tab:
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image_upload")
    if image_file:
        try:
            original = Image.open(image_file).convert("RGB")
            frame = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
            annotated, detections = infer_frame(model, frame, confidence, iou)
            violations = evaluate_violations(
                detections, required_classes, negative_mode=negative_mode, class_names=class_names
            )
            person_count = sum(item["class_name"].lower() == "person" for item in detections)
            left, right = st.columns(2)
            left.image(original, caption="Original image", width="stretch")
            right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated image", width="stretch")
            if violations:
                st.error(f"Safety alert: {len(violations)} violation(s) detected.")
            if detections:
                st.dataframe(
                    pd.DataFrame(
                        [{"Class": item["class_name"], "Confidence": f"{item['confidence']:.1%}", "Bounding box": item["bbox"]} for item in detections]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No objects were detected at the current confidence threshold.")
            if st.button("Save image scan", type="primary", key="save_image_scan"):
                scan_id = save_scan_result(st.session_state.user_id, "image", image_file.name, person_count, violations, annotated)
                st.success(f"Scan #{scan_id} saved.")
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

with video_tab:
    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"], key="video_upload")
    stride = st.sidebar.slider("Video frame stride", 1, 30, 5, 1, help="Run inference every Nth frame; all frames are kept in the output.")
    if video_file:
        if st.button("Process video", type="primary", key="process_video"):
            temporary_input = None
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as file:
                    file.write(video_file.getbuffer())
                    temporary_input = Path(file.name)
                output_path = Path("data/outputs") / f"{Path(video_file.name).stem}_annotated.mp4"
                progress = st.progress(0, text="Preparing video...")
                preview = st.empty()

                def update_preview(frame, current, total):
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Processing frame {current:,} of {total:,}", width="stretch")
                    progress.progress(min(current / max(total, 1), 1.0), text=f"Processing frame {current:,} of {total:,}")

                output_path, frame_count, person_count, violations = process_video(
                    temporary_input,
                    output_path,
                    model,
                    confidence,
                    iou,
                    required_classes,
                    frame_stride=stride,
                    negative_mode=negative_mode,
                    preview_callback=update_preview,
                    class_names=class_names,
                )
                progress.progress(1.0, text=f"Processed {frame_count:,} frames")
                if violations:
                    st.error(f"Safety alert: {len(violations)} violation detection(s) found across sampled frames.")
                scan_id = save_scan_result(st.session_state.user_id, "video", video_file.name, person_count, violations)
                st.success(f"Video scan #{scan_id} saved with {len(violations)} violation(s).")
                st.video(str(output_path))
                st.download_button("Download annotated H.264 video", output_path.read_bytes(), file_name=output_path.name, mime="video/mp4")
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                if temporary_input:
                    temporary_input.unlink(missing_ok=True)

with webcam_tab:
    camera_image = st.camera_input("Capture a site image")
    if camera_image:
        try:
            original = Image.open(camera_image).convert("RGB")
            frame = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
            annotated, detections = infer_frame(model, frame, confidence, iou)
            violations = evaluate_violations(
                detections, required_classes, negative_mode=negative_mode, class_names=class_names
            )
            person_count = sum(item["class_name"].lower() == "person" for item in detections)
            left, right = st.columns(2)
            left.image(original, caption="Camera capture", width="stretch")
            right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated capture", width="stretch")
            if violations:
                st.error(f"Safety alert: {len(violations)} violation(s) detected.")
            if st.button("Save camera scan", type="primary", key="save_camera_scan"):
                scan_id = save_scan_result(st.session_state.user_id, "webcam", "camera_capture.jpg", person_count, violations, annotated)
                st.success(f"Scan #{scan_id} saved.")
        except Exception as exc:
            st.error(f"Camera processing failed: {exc}")