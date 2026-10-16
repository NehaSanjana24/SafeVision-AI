"""Ultralytics model loading, class metadata, and frame inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def configured_model_path() -> Path:
    return Path(os.getenv("YOLO_MODEL_PATH", ROOT_DIR / "models" / "best.pt"))


def read_dataset_classes() -> list[str]:
    config_path = ROOT_DIR / "data" / "data.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        names = config.get("names", [])
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
        if isinstance(names, list):
            return [str(name) for name in names]
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    return []


def load_yolo_model():
    """Load custom weights when present, otherwise Ultralytics' small pretrained model."""
    from ultralytics import YOLO

    weights = configured_model_path()
    if weights.is_file():
        return YOLO(str(weights))
    fallback = os.getenv("YOLO_FALLBACK_MODEL", "yolov8n.pt")
    return YOLO(fallback)


def model_class_names(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return read_dataset_classes()


def infer_frame(
    model: Any,
    frame_bgr,
    confidence: float = 0.35,
    iou: float = 0.45,
) -> tuple[Any, list[dict[str, Any]]]:
    """Infer one BGR frame and return its annotated image and normalized detections."""
    result = model.predict(frame_bgr, conf=confidence, iou=iou, verbose=False)[0]
    annotated = result.plot()
    names = result.names
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(class_name),
                    "confidence": float(box.conf.item()),
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                }
            )
    if annotated is None or not hasattr(annotated, "shape"):
        raise RuntimeError("The model returned an invalid annotated frame.")
    if annotated.shape[:2] != frame_bgr.shape[:2]:
        annotated = cv2.resize(annotated, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return annotated, detections
