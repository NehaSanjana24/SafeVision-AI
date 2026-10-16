"""Train a YOLOv8n PPE detector and install its best weights for the app."""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DATA_CONFIG = ROOT_DIR / "data" / "data.yaml"
WEIGHTS_DIR = ROOT_DIR / "models"


def main() -> None:
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_CONFIG}")
    model = YOLO("yolov8n.pt")
    result = model.train(data=str(DATA_CONFIG), epochs=50, imgsz=640, project=str(ROOT_DIR / "runs" / "detect"), name="ppe")
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training completed without best weights at {best_weights}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best_weights, destination)
    print(f"Best weights copied to {destination}")


if __name__ == "__main__":
    main()