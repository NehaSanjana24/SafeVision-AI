# SafeVision AI

AI-assisted PPE monitoring built with Streamlit, Ultralytics YOLOv8, OpenCV, SQLite, and Plotly. It supports image and video scans, browser camera captures, evidence-backed alerts, dashboards, and PDF/CSV exports.

## Run in GitHub Codespaces

Open this repository in a Codespace. The dev container installs the system video libraries and Python packages, using CPU-only PyTorch wheels because Codespaces has no GPU, then forwards port `8501`. Start the app from the repository root:

```bash
streamlit run app.py
```

For a local Python 3.11 environment:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run creates `data/safety_monitoring.db`, the evidence directory, and a bootstrap administrator. Development defaults are username `admin` and password `ChangeMe123!`. Set `DEFAULT_ADMIN_USERNAME`, `DEFAULT_ADMIN_EMAIL`, and especially `DEFAULT_ADMIN_PASSWORD` before the first run to choose your own credentials. The bootstrap password is only used if no admin account exists yet. New self-service accounts are supervisors.

## Detection model

Place trained Ultralytics weights at `models/best.pt`, or set `YOLO_MODEL_PATH` to another `.pt` file. When custom weights are absent, the app tries to load/download Ultralytics `yolov8n.pt` (`YOLO_FALLBACK_MODEL` can change this). The generic COCO model is useful for exercising the app but is **not a PPE detector**; real PPE checks require weights trained for your label set. The app reads labels from model metadata and `data/data.yaml`. Negative classes such as `no-helmet` are reported directly; positive PPE datasets flag a person when a selected PPE class is not found inside that person's box.

`data/data.yaml` documents the expected YOLO dataset layout and training class order (`helmet`, `no-helmet`, `no-vest`, `person`, `vest`, `goggles`). The currently installed checkpoint has only the first five classes; the app warns that goggles detection is unavailable until you train weights containing class 5. Confirm names/order against your annotation IDs, place images/labels in the configured folders, and run `python train.py` to activate goggles detection.

## Training

Run training where an NVIDIA GPU is available, such as Google Colab or Kaggle. Upload this project and your dataset, install `requirements.txt`, update `data/data.yaml`, and run:

```bash
python train.py
```

The script fine-tunes `yolov8n.pt` for 50 epochs at image size 640 and copies the best checkpoint to `models/best.pt`. Codespaces generally has no GPU, so training there is slow; train in Colab/Kaggle, then download the resulting `models/best.pt` into the project. Keep the weights private if the dataset or model is proprietary.

## Features

- Passwords use bcrypt; SQLite stores users, scans, violation detections, and evidence paths.
- Image, video, and browser camera capture use configurable confidence/IoU thresholds and required PPE classes.
- Video processing samples frames for inference, preserves full-length output, and uses ffmpeg to encode browser-compatible H.264.
- The dashboard has interactive Plotly trends, filters, a demo-data switch, and CSV export.
- Reports include period KPIs, breakdown charts, evidence thumbnails, PDF, and CSV exports.
- Alert history supports search/filtering and administrator-only deletion.

## Configuration and data

SQLite lives at `data/safety_monitoring.db` by default. Set `SAFETY_DB_PATH` to use another location. Annotated evidence is written under `evidence/`; generated video is under `data/outputs/`. Back up both the database and evidence directory if you need durable records.

For production, set the bootstrap admin variables before the first launch, use persistent storage for SQLite and evidence, restrict access to the Streamlit service, and configure HTTPS through a reverse proxy. This app is an assistive monitoring tool: review detections before taking safety or employment actions.

## Streamlit Community Cloud

Create a Community Cloud app pointing to `app.py`; the repository includes `requirements.txt` and `.streamlit/config.toml`. Community Cloud does not provide durable local storage: its SQLite database, evidence files, and large model weights can be lost on rebuild or restart. For production deployment, use a persistent database/object store and load weights from durable or secured storage. Add `DEFAULT_ADMIN_PASSWORD` and other secrets in the app's Secrets settings before first launch. GPU inference is not generally available on Community Cloud.