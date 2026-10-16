"""Video decoding, sampled-frame annotation, and H.264 output encoding."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2

from core.detector import infer_frame
from core.violations import evaluate_violations


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model: Any,
    confidence: float,
    iou: float,
    required_classes: list[str],
    frame_stride: int = 5,
    negative_mode: bool = False,
    preview_callback: Callable[[Any, int, int], None] | None = None,
    class_names: list[str] | None = None,
) -> tuple[Path, int, int, list[dict[str, Any]]]:
    """Annotate every output frame while running inference at the selected stride."""
    input_path, output_path = Path(input_path), Path(output_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened. Try another MP4, AVI, or MOV file.")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video contains no readable frames.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_path.parent, delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Could not initialize the video encoder.")

    stride = max(1, int(frame_stride))
    frame_number = 0
    annotated = None
    total_persons = 0
    violations: list[dict[str, Any]] = []
    evidence_dir = Path(__file__).resolve().parents[1] / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_number % stride == 0 or annotated is None:
                annotated, detections = infer_frame(model, frame, confidence, iou)
                total_persons += sum(item["class_name"].lower() == "person" for item in detections)
                found = evaluate_violations(
                    detections,
                    required_classes,
                    negative_mode=negative_mode,
                    class_names=class_names,
                )
                evidence_path = None
                if found:
                    evidence_path = evidence_dir / f"video_{uuid.uuid4().hex}.jpg"
                    if not cv2.imwrite(str(evidence_path), annotated):
                        evidence_path = None
                for item in found:
                    item["frame_number"] = frame_number
                    item["evidence_path"] = str(evidence_path) if evidence_path else None
                violations.extend(found)
                if preview_callback and (frame_number == 0 or frame_number % max(stride * 10, 1) == 0):
                    preview_callback(annotated, frame_number + 1, total_frames)
            writer.write(annotated)
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    if frame_number == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("The video contains no readable frames.")
    ffmpeg = shutil.which("ffmpeg")
    try:
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary_path), "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            shutil.move(str(temporary_path), output_path)
    except (subprocess.CalledProcessError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not encode browser-compatible H.264 video: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path, frame_number, total_persons, violations
