"""Shared Streamlit helpers for authenticated safety-monitoring pages."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st

from core.detector import load_yolo_model, model_class_names, read_dataset_classes
from core.violations import is_negative_ppe_label
from db.database import fetch_frame, save_scan

ROOT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT_DIR / "evidence"


def require_page_auth() -> None:
    """Protect a page even when it is rerun independently by Streamlit navigation."""
    if not st.session_state.get("authenticated"):
        from auth.auth import initialize_default_admin, login_screen
        from db.database import init_database

        init_database()
        initialize_default_admin()
        login_screen()
        st.stop()
    with st.sidebar:
        st.markdown("### 🦺 SafeVision")
        st.caption(
            f"Signed in as **{st.session_state.get('username', 'user')}** · "
            f"{st.session_state.get('role', 'supervisor').title()}"
        )
        if st.button("↪  Sign out", width="stretch", key="sidebar_logout"):
            from auth.auth import logout

            logout()
            st.rerun()


@st.cache_resource(show_spinner="Loading the safety detection model...")
def get_model():
    return load_yolo_model()


def get_class_names(model: Any | None = None) -> list[str]:
    names = model_class_names(model) if model is not None else read_dataset_classes()
    return names


def render_detection_controls(class_names: list[str]) -> tuple[float, float, list[str]]:
    st.sidebar.subheader("Detection controls")
    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    available = [name for name in class_names if name.lower() != "person" and not is_negative_ppe_label(name)]
    defaults = [name for name in available if any(term in name.lower() for term in ("helmet", "vest", "mask", "goggle"))]
    required = st.sidebar.multiselect("Required PPE", available, default=defaults[:4])
    return confidence, iou, required


def save_scan_result(
    user_id: int,
    source_type: str,
    filename: str,
    total_persons: int,
    violations: list[dict[str, Any]],
    annotated_bgr=None,
) -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stored: list[dict[str, Any]] = []
    shared_evidence = None
    if violations and annotated_bgr is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shared_evidence = EVIDENCE_DIR / f"scan_{stamp}.jpg"
        if not cv2.imwrite(str(shared_evidence), annotated_bgr):
            shared_evidence = None
    for item in violations:
        stored.append(
            {
                "violation_type": item.get("violation_type", item.get("class_name", "Safety violation")),
                "confidence": item.get("confidence", 0),
                "frame_number": item.get("frame_number"),
                "bbox": json.dumps(item.get("bbox", [])),
                "evidence_path": item.get("evidence_path") or (str(shared_evidence) if shared_evidence else None),
            }
        )
    violating_subjects = {
        (item.get("frame_number"), tuple(round(float(value), 1) for value in item.get("bbox", [])))
        for item in violations
    }
    compliance = 100.0 if total_persons <= 0 else max(
        0.0, 100.0 * (total_persons - min(total_persons, len(violating_subjects))) / total_persons
    )
    return save_scan(user_id, source_type, filename, total_persons, len(violations), compliance, stored)


def load_scan_data(start_date: date | None = None, end_date: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    filters: list[str] = []
    params: list[Any] = []
    if start_date:
        filters.append("date(s.timestamp) >= date(?)")
        params.append(start_date.isoformat())
    if end_date:
        filters.append("date(s.timestamp) <= date(?)")
        params.append(end_date.isoformat())
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    scans = fetch_frame(
        f"""SELECT s.*, u.username FROM scans s JOIN users u ON u.id = s.user_id {where}
            ORDER BY s.timestamp DESC""",
        tuple(params),
    )
    violations = fetch_frame(
        f"""SELECT v.*, s.timestamp AS scan_timestamp, s.source_type, s.filename, s.user_id,
                   u.username, s.compliance_pct
            FROM violations v JOIN scans s ON s.id = v.scan_id
            JOIN users u ON u.id = s.user_id {where}
            ORDER BY v.timestamp DESC""",
        tuple(params),
    )
    return scans, violations


def empty_range() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=29), today


def date_range_filter(label: str = "Date range") -> tuple[date, date]:
    start, end = empty_range()
    selected = st.sidebar.date_input(label, value=(start, end))
    if isinstance(selected, tuple) and len(selected) == 2:
        return selected[0], selected[1]
    return selected, selected


def format_timestamp(value: Any) -> str:
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)
